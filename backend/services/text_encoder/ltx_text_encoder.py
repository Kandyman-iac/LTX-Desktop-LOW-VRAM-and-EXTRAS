"""Text encoder patching and API embedding service.

Multi-GPU strategy (when transformer_device != device):
  self.device           — video GPU (cuda:0): transformer, VAE, upsampler
  self.transformer_device — text GPU (cuda:1): Gemma text encoder, resident permanently

The text encoder is loaded onto the text GPU once and stays there.
Encoded embeddings are transferred to the video GPU before the transformer
consumes them, so no cross-device tensor errors occur during denoising.

Single-GPU fallback: both self.device and self.transformer_device are cuda:0,
the text encoder is loaded/offloaded as before.
"""

from __future__ import annotations

import logging
import pickle
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch

from services.http_client.http_client import HTTPClient
from services.services_utils import PromptInput, TensorOrNone, sync_device
from state.app_state_types import CachedTextEncoder, TextEncodingResult

if TYPE_CHECKING:
    from state.app_state_types import AppState

logger = logging.getLogger(__name__)


class LTXTextEncoder:
    """Stateless text encoding operations with idempotent monkey-patching.

    When transformer_device != device (multi-GPU):
      - Text encoder lives permanently on transformer_device (cuda:1).
      - Embeddings are transferred to device (cuda:0) for the transformer.

    When transformer_device == device (single-GPU):
      - Text encoder is loaded/offloaded on the single device as normal.
    """

    def __init__(self, device: torch.device, http: HTTPClient, ltx_api_base_url: str, transformer_device: torch.device | None = None) -> None:
        # device           = video GPU — where transformer/VAE run
        # transformer_device = text GPU — where Gemma lives permanently (multi-GPU)
        self.device = device
        self.transformer_device = transformer_device or device
        self.http = http
        self.ltx_api_base_url = ltx_api_base_url
        self._model_ledger_patched = False
        self._encode_text_patched = False

        if self.transformer_device != self.device:
            logger.info(
                "Multi-GPU text encoder: text_device=%s, video_device=%s",
                self.transformer_device, self.device,
            )

    def install_patches(self, state_getter: Callable[[], AppState]) -> None:
        self._install_model_ledger_patch(state_getter)
        self._install_encode_text_patch(state_getter)

    def _install_model_ledger_patch(self, state_getter: Callable[[], AppState]) -> None:
        if self._model_ledger_patched:
            return

        try:
            from ltx_pipelines.utils import ModelLedger
            from ltx_pipelines.utils import helpers as ltx_utils

            original_text_encoder = ModelLedger.text_encoder
            original_cleanup_memory = ltx_utils.cleanup_memory

            def _quantize_linear_weights_fp8(module: object) -> None:
                """Cast all Linear weights to float8_e4m3fn and patch forward to upcast."""
                for child in module.modules():  # type: ignore[union-attr]
                    if not isinstance(child, torch.nn.Linear):
                        continue
                    child.weight.data = child.weight.data.to(torch.float8_e4m3fn)
                    if child.bias is not None:  # pyright: ignore[reportUnnecessaryComparison]
                        child.bias.data = child.bias.data.to(torch.float8_e4m3fn)

                    def _make_upcast_forward(lin: torch.nn.Linear) -> Callable[..., torch.Tensor]:
                        def _fwd(x: torch.Tensor, **kw: object) -> torch.Tensor:
                            w = lin.weight.to(x.dtype)
                            b = lin.bias.to(x.dtype) if lin.bias is not None else None  # pyright: ignore[reportUnnecessaryComparison]
                            return torch.nn.functional.linear(x, w, b)
                        return _fwd

                    child.forward = _make_upcast_forward(child)  # type: ignore[assignment]

            def patched_text_encoder(self_model_ledger: ModelLedger) -> object:
                state = state_getter()
                te_state = state.text_encoder
                if te_state is None:
                    return original_text_encoder(self_model_ledger)

                if te_state.api_embeddings is not None:
                    return DummyTextEncoder()

                if te_state.cached_encoder is not None:
                    # Already resident on transformer_device — return as-is.
                    # In multi-GPU mode the encoder never leaves this device.
                    return te_state.cached_encoder

                # First call: load to CPU via ModelLedger default, then move
                # permanently to transformer_device (text GPU in multi-GPU mode).
                saved_device = self_model_ledger.device
                self_model_ledger.device = torch.device("cpu")
                try:
                    te_state.cached_encoder = cast(
                        CachedTextEncoder, original_text_encoder(self_model_ledger)
                    )
                finally:
                    self_model_ledger.device = saved_device

                _quantize_linear_weights_fp8(te_state.cached_encoder)

                te_state.cached_encoder.to(self.transformer_device)
                sync_device(self.transformer_device)
                logger.info("Text encoder loaded onto %s (permanent)", self.transformer_device)
                return te_state.cached_encoder

            def patched_cleanup_memory() -> None:
                # Multi-GPU: only flush the video device (self.device) — the text
                # encoder on self.transformer_device is intentionally kept resident.
                # Single-GPU: same behaviour as before (flush and free).
                if self.transformer_device != self.device and torch.cuda.is_available():
                    torch.cuda.synchronize(self.device)
                    with torch.cuda.device(self.device):
                        torch.cuda.empty_cache()
                original_cleanup_memory()

            setattr(ModelLedger, "text_encoder", patched_text_encoder)

            for module_name in (
                "ltx_pipelines.utils.helpers",
                "ltx_pipelines.distilled",
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
                "ltx_pipelines.retake_pipeline",
            ):
                try:
                    module = __import__(module_name, fromlist=["cleanup_memory"])
                    if hasattr(module, "cleanup_memory"):
                        setattr(module, "cleanup_memory", patched_cleanup_memory)
                except Exception:
                    logger.warning("Failed to patch cleanup_memory for module %s", module_name, exc_info=True)

            self._model_ledger_patched = True
            logger.info("Installed ModelLedger text encoder patch")
        except Exception as exc:
            logger.warning("Failed to patch ModelLedger: %s", exc, exc_info=True)

    def _install_encode_text_patch(self, state_getter: Callable[[], AppState]) -> None:
        if self._encode_text_patched:
            return

        try:
            from ltx_core.text_encoders import gemma as text_enc_module
            from ltx_pipelines import distilled as distilled_module

            original_encode_text = text_enc_module.encode_text

            def patched_encode_text(
                text_encoder: object,
                prompts: PromptInput,
                *args: object,
                **kwargs: object,
            ) -> list[tuple[torch.Tensor, TensorOrNone]]:
                state = state_getter()
                te_state = state.text_encoder

                if te_state is not None and te_state.api_embeddings is not None:
                    # API embeddings already on self.device (video GPU).
                    video_context = te_state.api_embeddings.video_context
                    audio_context = te_state.api_embeddings.audio_context
                    num_prompts = len(prompts) if not isinstance(prompts, str) else 1
                    out: list[tuple[torch.Tensor, TensorOrNone]] = []
                    for i in range(num_prompts):
                        if i == 0:
                            out.append((video_context, audio_context))
                        else:
                            zero_video = torch.zeros_like(video_context)
                            zero_audio = torch.zeros_like(audio_context) if audio_context is not None else None
                            out.append((zero_video, zero_audio))
                    return out

                prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
                # Encoding runs on transformer_device (text GPU).
                # Transfer results to device (video GPU) so the transformer
                # never sees cross-device tensors.
                result = cast(
                    list[tuple[torch.Tensor, TensorOrNone]],
                    original_encode_text(cast(Any, text_encoder), prompt_list, *args, **kwargs),
                )
                if self.transformer_device != self.device:
                    result = [
                        (
                            v.to(self.device),
                            a.to(self.device) if a is not None else None,
                        )
                        for v, a in result
                    ]
                return result

            setattr(text_enc_module, "encode_text", patched_encode_text)
            setattr(distilled_module, "encode_text", patched_encode_text)

            for module_name in (
                "ltx_pipelines.ti2vid_one_stage",
                "ltx_pipelines.ti2vid_two_stages",
                "ltx_pipelines.ic_lora",
                "ltx_pipelines.a2vid_two_stage",
                "ltx_pipelines.retake",
                "ltx_pipelines.retake_pipeline",
            ):
                try:
                    module = __import__(module_name, fromlist=["encode_text"])
                    setattr(module, "encode_text", patched_encode_text)
                except Exception:
                    logger.warning("Failed to patch encode_text for module %s", module_name, exc_info=True)

            self._encode_text_patched = True
            logger.info("Installed encode_text patch")
        except Exception as exc:
            logger.warning("Failed to patch encode_text: %s", exc, exc_info=True)

    def get_model_id_from_checkpoint(self, checkpoint_path: str) -> str | None:
        try:
            from safetensors import safe_open

            with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata and "encrypted_wandb_properties" in metadata:
                    return metadata["encrypted_wandb_properties"]
        except Exception as exc:
            logger.warning("Could not extract model_id from checkpoint: %s", exc, exc_info=True)
        return None

    def encode_via_api(self, prompt: str, api_key: str, checkpoint_path: str, enhance_prompt: bool) -> TextEncodingResult | None:
        model_id = self.get_model_id_from_checkpoint(checkpoint_path)
        if not model_id:
            return None

        try:
            start = time.time()
            response = self.http.post(
                f"{self.ltx_api_base_url}/v1/prompt-embedding",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json_payload={
                    "prompt": prompt,
                    "model_id": model_id,
                    "enhance_prompt": enhance_prompt,
                },
                timeout=60,
            )

            if response.status_code != 200:
                logger.warning("LTX API error %s: %s", response.status_code, response.text)
                return None

            conditioning = pickle.loads(response.content)  # noqa: S301
            if not conditioning or len(conditioning) == 0:
                logger.warning("LTX API returned unexpected conditioning format")
                return None

            embeddings = conditioning[0][0]
            video_dim = 4096
            if embeddings.shape[-1] > video_dim:
                video_context = embeddings[..., :video_dim].contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = embeddings[..., video_dim:].contiguous().to(dtype=torch.bfloat16, device=self.device)
            else:
                video_context = embeddings.contiguous().to(dtype=torch.bfloat16, device=self.device)
                audio_context = None

            logger.info("Text encoded via API in %.1fs", time.time() - start)
            return TextEncodingResult(video_context=video_context, audio_context=audio_context)

        except Exception as exc:
            logger.warning("LTX API encoding failed: %s", exc, exc_info=True)
            return None


class DummyTextEncoder:
    pass
