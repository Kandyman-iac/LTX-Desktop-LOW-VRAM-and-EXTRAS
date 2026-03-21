"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.lora_service import LoraEntry
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        pre_quantized_transformer_path: str = "",
        loras: list[LoraEntry] | None = None,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            transformer_device=transformer_device,
            block_swap_blocks_on_gpu=block_swap_blocks_on_gpu,
            attention_tile_size=attention_tile_size,
            use_fp8_transformer=use_fp8_transformer,
            gguf_transformer_path=gguf_transformer_path,
            vae_spatial_tile_size=vae_spatial_tile_size,
            vae_temporal_tile_size=vae_temporal_tile_size,
            pre_quantized_transformer_path=pre_quantized_transformer_path,
            loras=loras,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        transformer_device: torch.device | None = None,
        block_swap_blocks_on_gpu: int = 0,
        attention_tile_size: int = 0,
        use_fp8_transformer: bool = False,
        gguf_transformer_path: str = "",
        vae_spatial_tile_size: int = 0,
        vae_temporal_tile_size: int = 0,
        pre_quantized_transformer_path: str = "",
        loras: list[LoraEntry] | None = None,
    ) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        # Transformer device defaults to primary device if not set.
        self._transformer_device = transformer_device or device
        self._block_swap_blocks_on_gpu = block_swap_blocks_on_gpu
        self._attention_tile_size = attention_tile_size
        self._gguf_transformer_path = gguf_transformer_path
        self._vae_spatial_tile_size = vae_spatial_tile_size
        self._vae_temporal_tile_size = vae_temporal_tile_size

        # FP8: use setting OR auto-detect CUDA support.
        # The pipeline (transformer/VAE) always runs on device (video GPU, cuda:0).
        use_fp8 = use_fp8_transformer or device_supports_fp8(device)

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=QuantizationPolicy.fp8_cast() if use_fp8 else None,
        )

        # ── Swap in pre-quantized FP8 transformer (faster load, no on-the-fly downcast) ──
        # Skip if GGUF is configured — GGUF provides its own transformer weights.
        if use_fp8 and pre_quantized_transformer_path and os.path.exists(pre_quantized_transformer_path) and not gguf_transformer_path:
            self._install_pre_quantized_transformer(pre_quantized_transformer_path)

        # ── Install GGUF loader (replaces transformer weights source) ──
        if gguf_transformer_path:
            self._install_gguf(gguf_transformer_path)

        # ── Install block swapping ──
        if block_swap_blocks_on_gpu > 0:
            self._install_block_swap(block_swap_blocks_on_gpu)

        # ── Install attention tiling ──
        if attention_tile_size > 0:
            self._install_attention_tiling(attention_tile_size)

        # ── Apply LoRAs ──
        if loras:
            self._install_loras(loras)

    def _install_pre_quantized_transformer(self, fp8_path: str) -> None:
        """Replace the transformer builder with a pre-quantized FP8 file loader.

        The pre-quantized file contains LTXModel state dict (velocity_model keys,
        already renamed, already fp8).  No ComfyUI renaming or fp8 downcast needed —
        only UPCAST_DURING_INFERENCE to patch nn.Linear.forward at inference time.
        """
        try:
            from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
            from ltx_core.model.transformer import LTXModelConfigurator
            from ltx_core.quantization import QuantizationPolicy, UPCAST_DURING_INFERENCE
            import logging
            _log = logging.getLogger(__name__)

            ledger = self.pipeline.model_ledger
            ledger.transformer_builder = SingleGPUModelBuilder(
                model_class_configurator=LTXModelConfigurator,
                model_path=fp8_path,
                model_sd_ops=None,  # keys already in LTX format, no renaming needed
                registry=ledger.registry,
            )
            ledger.quantization = QuantizationPolicy(
                sd_ops=None,           # already fp8, no downcast needed
                module_ops=(UPCAST_DURING_INFERENCE,),
            )
            _log.info("Pre-quantized FP8 transformer installed from %s", fp8_path)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Pre-quantized FP8 install failed (%s) — falling back to on-the-fly fp8_cast", exc
            )

    def _install_gguf(self, gguf_path: str) -> None:
        try:
            from services.gguf_loader_service import GGUFLoaderService
            service = GGUFLoaderService(gguf_path=gguf_path)
            service.install(self.pipeline.model_ledger)
            self._gguf_service = service
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "GGUF install failed (%s) — falling back to safetensors", exc
            )

    def _install_block_swap(self, blocks_on_gpu: int) -> None:
        try:
            from services.block_swap_service import BlockSwapService
            service = BlockSwapService(
                blocks_on_gpu=blocks_on_gpu,
                device=self._transformer_device,
            )
            # Wrap model_ledger.transformer() persistently so block swap is
            # re-installed on every build (model_ledger never caches the model).
            original_transformer = self.pipeline.model_ledger.transformer

            def patched_transformer() -> torch.nn.Module:
                t = original_transformer()
                service.install(t)
                return t

            self.pipeline.model_ledger.transformer = patched_transformer
            self._block_swap_service = service
            import logging
            logging.getLogger(__name__).info(
                "BlockSwap configured: %d blocks on GPU", blocks_on_gpu
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "BlockSwap install failed (%s)", exc
            )

    def _install_attention_tiling(self, tile_size: int) -> None:
        try:
            from services.attention_tile_service import AttentionTileService
            service = AttentionTileService(tile_size=tile_size)
            service.install()
            self._attention_tile_service = service
            import logging
            logging.getLogger(__name__).info(
                "AttentionTiling installed: tile_size=%d", tile_size
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "AttentionTiling install failed (%s)", exc
            )

    def _install_loras(self, entries: list[LoraEntry]) -> None:
        import logging
        _log = logging.getLogger(__name__)
        try:
            from services.lora_service import LoraService
            service = LoraService(device=self._transformer_device)
            loaded = service.load_loras(entries)
            if not loaded:
                _log.warning("No LoRAs were successfully loaded")
                return

            # Wrap model_ledger.transformer() persistently so LoRA hooks are
            # re-applied on every build (model_ledger never caches the model).
            # At this point model_ledger.transformer may already be wrapped by
            # _install_block_swap, so we chain on top of that.
            _original_transformer_fn = self.pipeline.model_ledger.transformer

            def _transformer_with_loras() -> torch.nn.Module:
                t = _original_transformer_fn()
                service.apply_hooks_to_transformer(t, loaded)
                return t

            self.pipeline.model_ledger.transformer = _transformer_with_loras
            _log.info("LoRAs applied: %d loaded", len(loaded))
        except Exception as exc:
            _log.warning("LoRA install failed (%s)", exc)

    @staticmethod
    def _make_sigma_subset(num_steps: int) -> list[float]:
        """Return a subset of the distilled sigma schedule for fewer denoising steps.

        The full schedule has 8 steps (9 sigma values).  For fewer steps we pick
        evenly spaced values from the full list, always including the first (1.0)
        and last (0.0) so the denoising range stays correct.
        """
        full = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
        n = len(full) - 1  # 8
        if num_steps >= n:
            return full
        if num_steps <= 1:
            return [full[0], full[-1]]
        return [full[round(i * n / num_steps)] for i in range(num_steps + 1)]

    @staticmethod
    def _make_stg_denoising_func(stg_scale: float, stg_block_index: int):
        """Return a drop-in replacement for simple_denoising_func that applies STG.

        STG (Spatio-Temporal Guidance) improves prompt adherence by running a
        second forward pass with a single transformer block's self-attention
        replaced by identity, then steering away from that degraded prediction:
            output = cond + stg_scale * (cond - perturbed)

        No negative prompt is needed — cost is ~2× per step (one extra pass).
        """
        import logging as _logging
        _stg_log = _logging.getLogger(__name__)

        def _stg_func(video_context, audio_context, transformer):
            _stg_log.info("STG denoising func invoked: scale=%.2f block=%d", stg_scale, stg_block_index)
            from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
            from ltx_pipelines.utils.helpers import multi_modal_guider_factory_denoising_func

            video_params = MultiModalGuiderParams(
                cfg_scale=1.0,
                stg_scale=stg_scale,
                stg_blocks=[stg_block_index],
            )
            # No STG on audio — keep it simple / unperturbed.
            audio_params = MultiModalGuiderParams(
                cfg_scale=1.0,
                stg_scale=0.0,
                stg_blocks=[],
            )
            return multi_modal_guider_factory_denoising_func(
                video_guider_factory=MultiModalGuiderFactory.constant(video_params),
                audio_guider_factory=MultiModalGuiderFactory.constant(audio_params),
                v_context=video_context,
                a_context=audio_context,
                transformer=transformer,
            )

        return _stg_func

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType,
        num_steps: int = 8,
        stg_scale: float = 0.0,
        stg_block_index: int = 19,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        import ltx_pipelines.distilled as _distilled_mod

        # Release any fragmented reserved-but-unused CUDA memory before the
        # pipeline runs.  The transformer denoising loop leaves scattered
        # allocations across cuda:0; without this the VAE decoder (which needs
        # a single ~316 MiB contiguous block) can hit OOM even when there is
        # nominally enough free VRAM.
        torch.cuda.empty_cache()

        # Temporarily patch module-level names that DistilledPipeline.__call__
        # reads at call time.  Always restore in the finally block.
        _orig_sigmas = _distilled_mod.DISTILLED_SIGMA_VALUES
        _orig_simple = _distilled_mod.simple_denoising_func

        if num_steps < 8:
            _distilled_mod.DISTILLED_SIGMA_VALUES = self._make_sigma_subset(num_steps)  # type: ignore[attr-defined]
        if stg_scale > 0.0:
            _distilled_mod.simple_denoising_func = self._make_stg_denoising_func(stg_scale, stg_block_index)  # type: ignore[attr-defined]

        try:
            return self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
                tiling_config=tiling_config,
            )
        finally:
            # Synchronize before restoring module state so any async CUDA ops
            # queued inside DistilledPipeline.__call__ (e.g. vae_decode_audio)
            # are fully complete before Python GC frees the pipeline's locals.
            # Without this, dual-conditioning (2× CUDA tensors freed at __call__
            # return) can race with in-flight CUDA work → 0xC0000005 on Windows.
            torch.cuda.synchronize()
            _distilled_mod.DISTILLED_SIGMA_VALUES = _orig_sigmas
            _distilled_mod.simple_denoising_func = _orig_simple

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
        num_steps: int = 8,
        stg_scale: float = 0.0,
        stg_block_index: int = 19,
    ) -> None:
        tiling_config = default_tiling_config(
            spatial_tile_size=self._vae_spatial_tile_size,
            temporal_tile_size=self._vae_temporal_tile_size,
        )
        video, audio = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            num_steps=num_steps,
            stg_scale=stg_scale,
            stg_block_index=stg_block_index,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
        # Synchronize after VAE decode + audio write so all GPU→CPU transfers
        # finish before inference_mode context exits.  Explicit delete + cache
        # clear keeps VRAM tidy for the next generation.
        del video, audio
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        transformer = self.pipeline.model_ledger.transformer()

        compiled = cast(
            torch.nn.Module,
            torch.compile(transformer, mode="reduce-overhead", fullgraph=False),
        )
        setattr(self.pipeline.model_ledger, "transformer", lambda: compiled)