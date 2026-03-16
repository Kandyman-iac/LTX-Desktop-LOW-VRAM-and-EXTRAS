"""CivitAI and standard LoRA loading service for LTX-2.

Handles loading .safetensors LoRA files, detecting their key format
(official Lightricks vs CivitAI/diffusers-style), remapping keys as
needed, and applying them to the transformer via the ModelLedger.

Supports multiple LoRAs simultaneously with per-LoRA strength control.

Usage:
    service = LoraService(device=torch.device("cuda:0"))
    loras = service.load_loras([
        LoraEntry(path="/path/to/lora.safetensors", strength=0.8),
    ])
    service.apply_to_model_ledger(model_ledger, loras)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Data models                                                          #
# ------------------------------------------------------------------ #

@dataclass
class LoraEntry:
    """A single LoRA to load and apply."""
    path: str
    strength: float = 1.0
    enabled: bool = True


@dataclass
class LoadedLora:
    """A loaded and key-remapped LoRA ready to apply."""
    path: str
    strength: float
    state_dict: dict[str, torch.Tensor]
    format: str  # "lightricks" | "civitai" | "diffusers" | "unknown"


# ------------------------------------------------------------------ #
# Key format detection and remapping                                   #
# ------------------------------------------------------------------ #

# CivitAI / diffusers LoRAs use these key prefixes for the transformer.
_CIVITAI_TRANSFORMER_PREFIXES = (
    "lora_unet_",
    "transformer.",
    "lora_transformer_",
)

# Official Lightricks LoRAs use these prefixes.
_LIGHTRICKS_PREFIXES = (
    "transformer_blocks.",
    "single_transformer_blocks.",
)

# Diffusers lora suffix patterns.
_DIFFUSERS_LORA_SUFFIXES = (".lora_A.weight", ".lora_B.weight")
_CIVITAI_LORA_SUFFIXES = (".lora_down.weight", ".lora_up.weight", ".alpha")


def _detect_format(state_dict: dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if not keys:
        return "unknown"

    sample = keys[0]

    if any(sample.startswith(p) for p in _LIGHTRICKS_PREFIXES):
        return "lightricks"

    if any(s in sample for s in _CIVITAI_LORA_SUFFIXES):
        if any(sample.startswith(p) for p in _CIVITAI_TRANSFORMER_PREFIXES):
            return "civitai"

    if any(s in sample for s in _DIFFUSERS_LORA_SUFFIXES):
        return "diffusers"

    # Check a broader sample.
    for key in keys[:20]:
        if ".lora_down." in key or ".lora_up." in key:
            return "civitai"
        if ".lora_A." in key or ".lora_B." in key:
            return "diffusers"

    return "unknown"


def _remap_civitai_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap CivitAI-style keys to Lightricks transformer key format.

    CivitAI pattern:  lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight
    Lightricks pattern: transformer_blocks.0.attn.to_q.lora_down.weight
    """
    remapped: dict[str, torch.Tensor] = {}

    for key, tensor in state_dict.items():
        new_key = key

        # Strip common CivitAI prefixes.
        for prefix in ("lora_unet_", "lora_transformer_"):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break

        # Replace underscores-as-dots back to dots for known block patterns.
        # e.g. transformer_blocks_0_attn -> transformer_blocks.0.attn
        new_key = _underscores_to_dots(new_key)

        remapped[new_key] = tensor

    return remapped


def _remap_diffusers_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remap diffusers lora_A/lora_B keys to lora_down/lora_up convention."""
    remapped: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        new_key = key.replace(".lora_A.weight", ".lora_down.weight")
        new_key = new_key.replace(".lora_B.weight", ".lora_up.weight")
        remapped[new_key] = tensor
    return remapped


def _underscores_to_dots(key: str) -> str:
    """Convert CivitAI underscore-separated key segments to dot notation.

    Handles patterns like:
      transformer_blocks_0_attn_to_q  ->  transformer_blocks.0.attn.to_q
    """
    # Known structural segments that use underscores legitimately.
    # We convert numeric segments and known sub-module names.
    import re
    # Insert dots before numeric segments (block indices).
    key = re.sub(r'_(\d+)_', r'.\1.', key)
    key = re.sub(r'_(\d+)$', r'.\1', key)
    # Convert remaining structural underscores for known patterns.
    for segment in (
        "transformer_blocks",
        "single_transformer_blocks",
        "attn",
        "ff",
        "norm",
        "proj",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "net",
    ):
        # Only replace if it appears as a standalone underscore-joined segment.
        key = key.replace(f"_{segment}_", f".{segment}.")
        key = key.replace(f"_{segment}.", f".{segment}.")
        if key.endswith(f"_{segment}"):
            key = key[: -len(f"_{segment}")] + f".{segment}"

    return key


# ------------------------------------------------------------------ #
# LoRA service                                                         #
# ------------------------------------------------------------------ #

class LoraService:
    """Loads and applies LoRA weights to the LTX-2 transformer."""

    def __init__(self, device: torch.device) -> None:
        self.device = device

    def load_loras(self, entries: list[LoraEntry]) -> list[LoadedLora]:
        """Load and remap a list of LoRA entries from disk."""
        loaded: list[LoadedLora] = []
        for entry in entries:
            if not entry.enabled:
                continue
            lora = self._load_single(entry)
            if lora is not None:
                loaded.append(lora)
        return loaded

    def apply_to_model_ledger(
        self,
        model_ledger: Any,
        loras: list[LoadedLora],
    ) -> None:
        """Apply loaded LoRAs to a ModelLedger instance.

        Attempts to use the ModelLedger's native loras= parameter first.
        Falls back to direct weight patching if unavailable.
        """
        if not loras:
            return

        try:
            from ltx_core.loader.lora import apply_lora_to_model
            transformer = model_ledger.transformer()
            for lora in loras:
                logger.info(
                    "Applying LoRA %s (format=%s, strength=%.2f)",
                    Path(lora.path).name,
                    lora.format,
                    lora.strength,
                )
                apply_lora_to_model(
                    model=transformer,
                    lora_state_dict=lora.state_dict,
                    scale=lora.strength,
                )
            logger.info("Applied %d LoRA(s) via ltx_core loader", len(loras))
        except Exception as exc:
            logger.warning(
                "ltx_core LoRA loader unavailable (%s), falling back to direct merge",
                exc,
            )
            self._apply_direct_merge(model_ledger, loras)

    def _apply_direct_merge(
        self,
        model_ledger: Any,
        loras: list[LoadedLora],
    ) -> None:
        """Fallback: directly merge LoRA weights into transformer parameters."""
        try:
            transformer = model_ledger.transformer()
            model_sd = dict(transformer.named_parameters())

            for lora in loras:
                merged = 0
                sd = lora.state_dict
                keys = list(sd.keys())

                # Find lora_down/lora_up pairs.
                down_keys = [k for k in keys if k.endswith(".lora_down.weight")]
                for down_key in down_keys:
                    up_key = down_key.replace(".lora_down.weight", ".lora_up.weight")
                    base_key = down_key.replace(".lora_down.weight", ".weight")

                    if up_key not in sd:
                        continue
                    if base_key not in model_sd:
                        # Try without .weight suffix.
                        base_key_alt = down_key.replace(".lora_down.weight", "")
                        if base_key_alt not in model_sd:
                            continue
                        base_key = base_key_alt

                    down = sd[down_key].to(self.device, dtype=torch.float32)
                    up = sd[up_key].to(self.device, dtype=torch.float32)

                    # Check for alpha scaling.
                    alpha_key = down_key.replace(".lora_down.weight", ".alpha")
                    alpha = sd[alpha_key].item() if alpha_key in sd else down.shape[0]
                    scale = lora.strength * (alpha / down.shape[0])

                    # Compute delta = up @ down (rank decomposition).
                    if down.dim() == 2 and up.dim() == 2:
                        delta = (up @ down) * scale
                    else:
                        # Conv weights — flatten, multiply, reshape.
                        delta = (up.flatten(1) @ down.flatten(1)) * scale
                        delta = delta.reshape(up.shape[0], down.shape[1], *up.shape[2:])

                    param = model_sd[base_key]
                    param.data.add_(delta.to(param.dtype))
                    merged += 1

                logger.info(
                    "Direct merge: applied %d weight pairs from %s",
                    merged,
                    Path(lora.path).name,
                )
        except Exception as exc:
            logger.error("Direct LoRA merge failed: %s", exc, exc_info=True)

    def _load_single(self, entry: LoraEntry) -> LoadedLora | None:
        path = Path(entry.path)
        if not path.exists():
            logger.warning("LoRA file not found: %s", path)
            return None
        if path.suffix.lower() != ".safetensors":
            logger.warning("LoRA file must be .safetensors format: %s", path)
            return None

        try:
            from safetensors.torch import load_file
            state_dict = load_file(str(path), device="cpu")
        except Exception as exc:
            logger.error("Failed to load LoRA %s: %s", path, exc, exc_info=True)
            return None

        fmt = _detect_format(state_dict)
        logger.info("Loaded LoRA %s: %d keys, format=%s", path.name, len(state_dict), fmt)

        if fmt == "civitai":
            state_dict = _remap_civitai_keys(state_dict)
        elif fmt == "diffusers":
            state_dict = _remap_diffusers_keys(state_dict)

        return LoadedLora(
            path=str(path),
            strength=entry.strength,
            state_dict=state_dict,
            format=fmt,
        )


def build_lora_service(device: torch.device) -> LoraService:
    """Factory for LoraService."""
    return LoraService(device=device)
