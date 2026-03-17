"""Handler for manually encoding a prompt on GPU (single-GPU mode)."""

from __future__ import annotations

import gc
import logging
import torch
from threading import RLock
from typing import TYPE_CHECKING

from handlers.base import StateHandlerBase
from runtime_config.model_download_specs import resolve_model_path
from services.services_utils import sync_device
from state.app_state_types import TextEncodingResult

if TYPE_CHECKING:
    from handlers.pipelines_handler import PipelinesHandler
    from handlers.text_handler import TextHandler
    from runtime_config.runtime_config import RuntimeConfig
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class EncodePromptHandler(StateHandlerBase):
    """Encodes a prompt on GPU with only Gemma loaded — no transformer in VRAM."""

    def __init__(
        self,
        state: AppState,
        lock: RLock,
        config: RuntimeConfig,
        pipelines_handler: PipelinesHandler,
        text_handler: TextHandler,
    ) -> None:
        super().__init__(state, lock, config)
        self._pipelines = pipelines_handler
        self._text = text_handler

    def is_single_gpu_local_mode(self) -> bool:
        """True when manual prompt encoding is relevant.

        Only needed in single-GPU + local encoder mode.  Multi-GPU keeps
        Gemma on cuda:1 permanently so the workflow is automatic.
        """
        if self.state.app_settings.use_multi_gpu:
            return False
        return self._text.should_use_local_encoding()

    def get_encoded_prompt(self) -> str | None:
        te = self.state.text_encoder
        return te.encoded_prompt if te is not None else None

    def encode_prompt(self, prompt: str) -> None:
        """Eject video pipeline, load Gemma to GPU, encode prompt, cache, unload Gemma.

        Always stores the result under the key (prompt.strip(), False) because
        local encoding never applies prompt enhancement.
        """
        te = self.state.text_encoder
        if te is None:
            raise RuntimeError("Text encoder state not initialised")

        gemma_root = self._text.resolve_gemma_root()
        if not gemma_root:
            raise RuntimeError(
                "Local text encoder not available. "
                "Download the text encoder or check Settings."
            )

        # ── 1. Free VRAM ────────────────────────────────────────────────────
        self._pipelines.unload_gpu_pipeline()
        logger.info("GPU pipeline ejected; VRAM freed for prompt encoding")

        # ── 2. Obtain GemmaTextEncoder on CPU ──────────────────────────────
        # Clear api_embeddings so patched_text_encoder returns the real encoder
        # (not DummyTextEncoder).  We will overwrite api_embeddings at the end.
        with self._lock:
            te.api_embeddings = None

        encoder = te.cached_encoder
        if encoder is None:
            logger.info("Cold-start: loading Gemma text encoder from disk to CPU…")
            # Ensure the ModelLedger.text_encoder() monkey-patch is active so
            # the patched version handles FP8 quantisation and caching.
            self._pipelines._install_text_patches_if_needed()

            checkpoint_path = str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "checkpoint"
                )
            )
            from ltx_pipelines.utils.model_ledger import ModelLedger

            ledger = ModelLedger(
                dtype=torch.bfloat16,
                device=torch.device("cpu"),   # patch loads to CPU then applies FP8
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
            )
            # patched_text_encoder: loads weights → CPU, applies FP8, stores in
            # te.cached_encoder, returns the encoder.
            encoder = ledger.text_encoder()  # type: ignore[assignment]

            if encoder is None or type(encoder).__name__ == "DummyTextEncoder":
                raise RuntimeError(
                    "Failed to load Gemma text encoder — "
                    "DummyTextEncoder returned unexpectedly during cold-start"
                )

        # ── 3. Move Gemma to GPU (VRAM is now free) ─────────────────────────
        device = self.config.device
        logger.info("Moving Gemma to %s for encoding", device)
        encoder.to(device)
        sync_device(device)

        v: torch.Tensor
        a: torch.Tensor | None
        try:
            # ── 4. Encode using the unpatched encode_text directly ───────────
            # We bypass patched_encode_text (which is designed for the pipeline
            # call-site) and call the original ltx_core function ourselves.
            from ltx_core.text_encoders.gemma.encoders.base_encoder import (
                encode_text as _raw_encode_text,
            )

            with torch.inference_mode():
                raw = _raw_encode_text(encoder, [prompt])

            v_raw, a_raw = raw[0]
            v = v_raw.to(device)
            a = a_raw.to(device) if a_raw is not None else None

        finally:
            # ── 5. Return Gemma to CPU and free VRAM ─────────────────────────
            logger.info("Returning Gemma to CPU; freeing VRAM")
            encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # ── 6. Cache the result ──────────────────────────────────────────────
        encoded = TextEncodingResult(video_context=v, audio_context=a)
        prompt_key = (prompt.strip(), False)   # False: no prompt enhancement

        with self._lock:
            max_size = self.state.app_settings.prompt_cache_size
            if max_size > 0:
                if prompt_key in te.prompt_cache:
                    del te.prompt_cache[prompt_key]
                elif len(te.prompt_cache) >= max_size:
                    oldest = next(iter(te.prompt_cache))
                    del te.prompt_cache[oldest]
                te.prompt_cache[prompt_key] = encoded

            # api_embeddings set so that if generate runs immediately the
            # DummyTextEncoder path is taken and the cached result is used.
            te.api_embeddings = encoded
            te.encoded_prompt = prompt.strip()

        logger.info(
            "Prompt encoded on GPU and cached. Gemma returned to CPU. VRAM freed."
        )

    def enhance_prompt(self, prompt: str) -> str:
        """Run Gemma's enhance_t2v on the prompt and return the enhanced text.

        Unlike encode_prompt, this only calls the LM for text expansion — no
        embeddings are computed or cached.  In multi-GPU mode the text encoder
        is already resident on cuda:1 so this is fast.  In single-GPU mode the
        pipeline is ejected and Gemma is moved to GPU for the call.
        """
        from typing import Any, cast

        te = self.state.text_encoder
        if te is None:
            raise RuntimeError("Text encoder state not initialised")

        gemma_root = self._text.resolve_gemma_root()
        if not gemma_root:
            raise RuntimeError(
                "Local text encoder not available. "
                "Download the text encoder or check Settings."
            )

        encoder = te.cached_encoder
        if encoder is None:
            logger.info("Cold-start: loading Gemma text encoder for prompt enhancement…")
            self._pipelines._install_text_patches_if_needed()
            checkpoint_path = str(
                resolve_model_path(
                    self.models_dir, self.config.model_download_specs, "checkpoint"
                )
            )
            from ltx_pipelines.utils.model_ledger import ModelLedger
            ledger = ModelLedger(
                dtype=torch.bfloat16,
                device=torch.device("cpu"),
                checkpoint_path=checkpoint_path,
                gemma_root_path=gemma_root,
            )
            encoder = ledger.text_encoder()
            if encoder is None or type(encoder).__name__ == "DummyTextEncoder":
                raise RuntimeError("Failed to load Gemma text encoder")

        if not hasattr(encoder, "enhance_t2v"):
            raise RuntimeError(
                "Text encoder does not support local prompt enhancement "
                "(enhance_t2v missing — is Gemma downloaded?)."
            )

        # Multi-GPU: encoder already resident on cuda:1, use as-is.
        # Single-GPU: eject the video pipeline and move encoder to GPU.
        is_multi_gpu = self.state.app_settings.use_multi_gpu
        moved_to_gpu = False
        if not is_multi_gpu:
            self._pipelines.unload_gpu_pipeline()
            device = self.config.device
            logger.info("Moving Gemma to %s for prompt enhancement", device)
            encoder.to(device)
            sync_device(device)
            moved_to_gpu = True

        try:
            with torch.inference_mode():
                enhanced = cast(Any, encoder).enhance_t2v(prompt)
            enhanced_str = str(enhanced)
            logger.info(
                "Prompt enhanced: %r -> %r",
                prompt[:60],
                enhanced_str[:60],
            )
            return enhanced_str
        finally:
            if moved_to_gpu:
                logger.info("Returning Gemma to CPU; freeing VRAM")
                encoder.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
