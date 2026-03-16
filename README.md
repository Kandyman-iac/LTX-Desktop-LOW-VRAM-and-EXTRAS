# LTX Desktop — Low VRAM & Extras

A custom fork of [Lightricks/LTX-Desktop](https://github.com/Lightricks/LTX-Desktop) enabling low-VRAM video generation with block swapping, multi-GPU support, LoRA, GGUF, and attention tiling.

> **Status: Beta.** Expect breaking changes.
> Frontend architecture is under active refactor; large UI PRs may be declined for now (see [`CONTRIBUTING.md`](docs/CONTRIBUTING.md)).

<p align="center">
  <img src="images/gen-space.png" alt="Gen Space" width="70%">
</p>

<p align="center">
  <img src="images/video-editor.png" alt="Video Editor" width="70%">
</p>

<p align="center">
  <img src="images/timeline-gap-fill.png" alt="Timeline gap fill" width="70%">
</p>

## What's different in this fork

This fork extends the upstream LTX-Desktop with low-VRAM support and additional capabilities, targeting consumer GPUs from 8GB VRAM upwards. The reference hardware is a dual NVIDIA RTX 3090 (24GB each, 48GB total).

| Feature | Description |
| --- | --- |
| **Block swapping** | Keeps only N transformer blocks resident on GPU at a time, swapping the rest to CPU RAM. Configurable via `blockSwapBlocksOnGpu` (0–48). Lowers the VRAM floor from 31GB to ~8GB. |
| **Attention tiling** | Tiles the query sequence dimension during self-attention to reduce peak VRAM. Configurable via `attentionTileSize`. |
| **Multi-GPU support** | Splits the text encoder and transformer across two GPUs (`cuda:0` for transformer, `cuda:1` for text encoder) when `useMultiGpu` is enabled and two CUDA devices are present. |
| **LoRA support** | Load CivitAI-compatible LoRA weight files and merge them into the transformer at configurable strength. Multiple LoRAs can be active simultaneously with per-LoRA enable/disable and strength controls. |
| **GGUF model loading** | Load quantized GGUF model files as an alternative to full-precision SafeTensors, reducing disk and memory requirements. |
| **Abliterated text encoder** | Optional Gemma-based abliterated encoder for uncensored prompt encoding, toggled via `useAbliteratedEncoder`. |
| **Extended generation lengths** | 15, 20, and 25 second video generation (361+ frames at 24fps) on supported hardware. |
| **Lowered VRAM gate** | VRAM requirement lowered from 31GB to 8GB, enabling local generation on a much wider range of consumer GPUs. |

## Features

- Text-to-video generation
- Image-to-video generation
- Audio-to-video generation
- Video edit generation (Retake)
- Video Editor Interface
- Video Editing Projects
- 15 / 20 / 25 second generation lengths
- Block swapping (low-VRAM inference)
- Attention tiling (reduced peak VRAM)
- Multi-GPU inference (dual GPU support)
- LoRA support (CivitAI-compatible)
- GGUF quantized model loading
- Abliterated text encoder (Gemma)

## Local vs API mode

| Platform / hardware | Generation mode | Notes |
| --- | --- | --- |
| Windows + CUDA GPU with **≥8GB VRAM** | Local generation (with block swapping) | Downloads model weights locally |
| Windows + CUDA GPU with **≥24GB VRAM** | Local generation (full speed) | Downloads model weights locally |
| Windows (no CUDA or unknown VRAM) | API-only | **LTX API key required** |
| Linux + CUDA GPU with **≥8GB VRAM** | Local generation (with block swapping) | Downloads model weights locally |
| Linux + CUDA GPU with **≥24GB VRAM** | Local generation (full speed) | Downloads model weights locally |
| Linux (no CUDA or unknown VRAM) | API-only | **LTX API key required** |
| macOS (Apple Silicon builds) | API-only | **LTX API key required** |

In API-only mode, available resolutions/durations may be limited to what the API supports.

## System requirements

### Windows (local generation)

- Windows 10/11 (x64)
- NVIDIA GPU with CUDA support and **≥8GB VRAM** (block swapping enabled; 24GB+ recommended for full performance)
- 16GB+ RAM (32GB+ recommended when using block swapping — CPU RAM absorbs offloaded blocks)
- **160GB+ free disk space** (for model weights, Python environment, and outputs)

### Linux (local generation)

- Ubuntu 22.04+ or similar distro (x64 or arm64)
- NVIDIA GPU with CUDA support and **≥8GB VRAM** (block swapping enabled; 24GB+ recommended for full performance)
- NVIDIA driver installed (PyTorch bundles the CUDA runtime)
- 16GB+ RAM (32GB+ recommended when using block swapping)
- Plenty of free disk space for model weights and outputs

### macOS (API-only)

- Apple Silicon (arm64)
- macOS 13+ (Ventura)
- Stable internet connection

## Low-VRAM configuration

Block swapping and attention tiling are the two main VRAM-saving knobs. Both are configured in **Settings**.

| Setting | Type | Description |
| --- | --- | --- |
| `blockSwapBlocksOnGpu` | int (0–48) | Number of transformer blocks kept on GPU simultaneously. Lower = less VRAM, slower generation. 0 disables block swapping. |
| `attentionTileSize` | int | Query chunk size for tiled attention. Smaller = less peak VRAM during attention. |
| `useMultiGpu` | bool | Split transformer (cuda:0) and text encoder (cuda:1) across two GPUs when available. |
| `useFp8Transformer` | bool | Enable FP8 precision for the transformer (reduces VRAM, minor quality trade-off). |
| `ggufTransformerPath` | string | Path to a GGUF quantized model file (alternative to SafeTensors). |

**Recommended starting config for a single 24GB GPU (e.g. RTX 3090/4090):**
- `blockSwapBlocksOnGpu`: 20–32
- `attentionTileSize`: 256–512

**Recommended config for dual 24GB GPUs:**
- `useMultiGpu`: true
- `blockSwapBlocksOnGpu`: 0–20 (full 24GB available to transformer alone)
- `attentionTileSize`: 512

## LoRA support

Place CivitAI-compatible `.safetensors` LoRA files in any accessible folder and add them via **Settings > LoRAs**. Each LoRA can be independently enabled/disabled with a strength slider. Multiple LoRAs can be active simultaneously.

The `civitaiLoras` setting holds the list of active LoRA configurations (path, strength, enabled).

## GGUF model loading

Set `ggufTransformerPath` in settings to the path of a quantized `.gguf` transformer file. This replaces the default SafeTensors loader and can significantly reduce disk footprint and VRAM usage. Leave blank to use the default full-precision SafeTensors model.

## Abliterated text encoder

Set `useAbliteratedEncoder: true` in settings to swap the default text encoder for a Gemma-based abliterated encoder. This removes content filtering from prompt encoding. The encoder is swapped before embeddings are generated; no other part of the pipeline is affected.

## Install

1. Download the latest installer from GitHub Releases: [Releases](../../releases)
2. Install and launch **LTX Desktop**
3. Complete first-run setup

## First run & data locations

LTX Desktop stores app data (settings, models, logs) in:

- **Windows:** `%LOCALAPPDATA%\LTXDesktop\`
- **macOS:** `~/Library/Application Support/LTXDesktop/`
- **Linux:** `$XDG_DATA_HOME/LTXDesktop/` (default: `~/.local/share/LTXDesktop/`)

Model weights are downloaded into the `models/` subfolder (this can be large and may take time).

On first launch you may be prompted to review/accept model license terms (license text is fetched from Hugging Face; requires internet).

Text encoding: to generate videos you must configure text encoding:

- **LTX API key** (cloud text encoding) — **text encoding via the API is completely FREE** and highly recommended to speed up inference and save memory. Generate a free API key at the [LTX Console](https://console.ltx.video/). [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).
- **Local Text Encoder** (extra download; enables fully-local operation on supported Windows hardware) — if you don't wish to generate an API key, you can encode text locally via the settings menu.

## API keys, cost, and privacy

### LTX API key

The LTX API is used for:

- **Cloud text encoding and prompt enhancement** — **FREE**; text encoding is highly recommended to speed up inference and save memory
- API-based video generations (required on macOS and on unsupported Windows hardware) — paid
- Retake — paid

An LTX API key is required in API-only mode, but optional on Windows/Linux local mode if you enable the Local Text Encoder.

Generate a FREE API key at the [LTX Console](https://console.ltx.video/). Text encoding is free; video generation API usage is paid. [Read more](https://ltx.io/model/model-blog/ltx-2-better-control-for-real-workflows).

When you use API-backed features, prompts and media inputs are sent to the API service. Your API key is stored locally in your app data folder — treat it like a secret.

### fal API key (optional)

Used for Z Image Turbo text-to-image generation in API mode. When enabled, image generation requests are sent to fal.ai.

Create an API key in the [fal dashboard](https://fal.ai/dashboard/keys).

### Gemini API key (optional)

Used for AI prompt suggestions. When enabled, prompt context and frames may be sent to Google Gemini.

## Architecture

LTX Desktop is split into three main layers:

- **Renderer (`frontend/`)**: TypeScript + React UI.
  - Calls the local backend over HTTP at `http://localhost:8000`.
  - Talks to Electron via the preload bridge (`window.electronAPI`).
- **Electron (`electron/`)**: TypeScript main process + preload.
  - Owns app lifecycle and OS integration (file dialogs, native export via ffmpeg, starting/managing the Python backend).
  - Security: renderer is sandboxed (`contextIsolation: true`, `nodeIntegration: false`).
- **Backend (`backend/`)**: Python + FastAPI local server.
  - Orchestrates generation, model downloads, and GPU execution.
  - Calls external APIs only when API-backed features are used.

### Fork-specific backend services

| Service | File | Description |
| --- | --- | --- |
| `BlockSwapService` | `backend/services/block_swap_service.py` | Patches transformer block `forward()` methods to swap blocks on/off GPU in a sliding window during inference. |
| `AttentionTileService` | `backend/services/attention_tile_service.py` | Globally patches `F.scaled_dot_product_attention` to process queries in tiles. |
| `LoraService` | `backend/services/lora_service.py` | Loads and merges CivitAI LoRA weights into the transformer. |
| `GGUFLoaderService` | `backend/services/gguf_loader_service.py` | Replaces the default SafeTensors loader with a GGUF-compatible loader. |
| `AbliterationService` | `backend/services/abliteration_service.py` | Swaps in a Gemma-based abliterated text encoder before embedding generation. |

```mermaid
graph TD
  UI["Renderer (React + TS)"] -->|HTTP: localhost:8000| BE["Backend (FastAPI + Python)"]
  UI -->|IPC via preload: window.electronAPI| EL["Electron main (TS)"]
  EL --> OS["OS integration (files, dialogs, ffmpeg, process mgmt)"]
  BE --> GPU["Local models + GPU (when supported)"]
  BE --> EXT["External APIs (only for API-backed features)"]
  EL --> DATA["App data folder (settings/models/logs)"]
  BE --> DATA
```

## Development (quickstart)

Prereqs:

- Node.js
- `uv` (Python package manager)
- Python 3.12+
- Git

Setup:

```bash
pnpm setup:dev
```

Run:

```bash
pnpm dev
```

Debug:

```bash
pnpm dev:debug
```

`dev:debug` starts Electron with inspector enabled and starts the Python backend with `debugpy`.

Typecheck:

```bash
pnpm typecheck
```

Backend tests:

```bash
pnpm backend:test
```

Building installers:
- See [`INSTALLER.md`](docs/INSTALLER.md)

## Telemetry

LTX Desktop collects minimal, anonymous usage analytics (app version, platform, and a random installation ID) to help prioritize development. No personal information or generated content is collected. Analytics is enabled by default and can be disabled in **Settings > General > Anonymous Analytics**. See [`TELEMETRY.md`](docs/TELEMETRY.md) for details.

## Docs

- [`INSTALLER.md`](docs/INSTALLER.md) — building installers
- [`TELEMETRY.md`](docs/TELEMETRY.md) — telemetry and privacy
- [`backend/architecture.md`](backend/architecture.md) — backend architecture

## Contributing

See [`CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## License

Apache-2.0 — see [`LICENSE.txt`](LICENSE.txt).

Third-party notices (including model licenses/terms): [`NOTICES.md`](NOTICES.md).

Model weights are downloaded separately and may be governed by additional licenses/terms.
