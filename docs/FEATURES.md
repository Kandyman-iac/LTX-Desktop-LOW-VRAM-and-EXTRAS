# Feature Guide

Detailed usage instructions for every feature in this fork.

---

## Contents

- [VRAM optimisation](#vram-optimisation)
  - [Block swapping](#block-swapping)
  - [Attention tiling](#attention-tiling)
  - [VAE tiling](#vae-tiling)
  - [FP8 transformer](#fp8-transformer)
  - [Multi-GPU](#multi-gpu)
- [Model loading](#model-loading)
  - [GGUF quantized models](#gguf-quantized-models)
  - [LoRA support](#lora-support)
  - [Abliterated text encoder](#abliterated-text-encoder)
- [Generation controls](#generation-controls)
  - [Inference steps](#inference-steps)
  - [STG (Spatio-Temporal Guidance)](#stg-spatio-temporal-guidance)
  - [Negative prompt](#negative-prompt)
  - [Seed locking](#seed-locking)
  - [Pipeline reload interval](#pipeline-reload-interval)
- [Prompt tooling](#prompt-tooling)
  - [Prompt enhancement](#prompt-enhancement)
  - [Prompt history](#prompt-history)
  - [Sidecar metadata](#sidecar-metadata)
- [Multi-frame conditioning](#multi-frame-conditioning)
- [Generation queue](#generation-queue)
- [Retake](#retake)
- [Gap fill](#gap-fill)
- [Clip Viewer (Source Monitor)](#clip-viewer-source-monitor)

---

## VRAM optimisation

All VRAM settings are in **Settings > VRAM & Inference**.

### Block swapping

Block swapping keeps only a sliding window of transformer blocks resident on the GPU at any one time. The remaining blocks are held in CPU RAM and swapped in just before they're needed during the forward pass.

**Setting:** `blockSwapBlocksOnGpu` (integer, 0–48)

- `0` — block swapping disabled; all 28 blocks stay on GPU (requires ~24GB+ VRAM)
- `4–8` — minimum mode; works on 8GB GPUs but is slow
- `20–28` — recommended for single 24GB GPUs
- `48` — no blocks on GPU (maximum offload; very slow, minimum VRAM ~4GB)

**Trade-off:** each block swap is a CPU↔GPU memory transfer. Fewer blocks on GPU = more swaps per step = slower generation.

**RAM requirements:** blocks offloaded to CPU sit in RAM. A full 28-block offload uses roughly 12–16GB of RAM depending on precision. Ensure you have 32GB+ RAM if running deep block swapping.

### Attention tiling

During self-attention, the full query sequence can be very large (especially for long or high-resolution videos). Attention tiling processes queries in chunks of size `attentionTileSize`, capping peak VRAM during the attention operation.

**Setting:** `attentionTileSize` (integer, e.g. 64, 128, 256, 512, 1024)

- Smaller values = less peak VRAM, slightly slower
- `256–512` is a good starting point for 24GB GPUs
- `64–128` for 8–12GB GPUs
- `0` or very large values effectively disable tiling

Attention tiling is applied globally to all `scaled_dot_product_attention` calls in the model.

### VAE tiling

The VAE decoder (which converts the latent video back to pixel space) processes the full spatial resolution in one pass by default. VAE tiling splits this into spatial tiles to reduce the peak VRAM spike at the end of generation.

**Setting:** `useVaeTiling` (boolean, default off)

Enable this if you're running out of VRAM at the very end of generation (after diffusion completes). It has no visible effect on output quality.

### FP8 transformer

Runs the transformer weights in 8-bit floating point instead of 16-bit, roughly halving the transformer's VRAM footprint at a minor quality cost.

**Setting:** `useFp8Transformer` (boolean, default off)

- Reduces transformer VRAM by ~50%
- Works best combined with block swapping
- Not compatible with all GPUs (requires Ampere/Ada or newer for hardware FP8; older GPUs use software emulation which is slower)
- Slight reduction in output quality / consistency

### Multi-GPU

Distributes the model across two CUDA GPUs: the transformer runs on `cuda:0` and the text encoder runs on `cuda:1`. This allows each GPU to hold its dedicated component without competing for VRAM.

**Setting:** `useMultiGpu` (boolean, default off)

- Requires exactly two CUDA-capable GPUs
- The app detects GPU count at startup; the toggle is hidden if only one GPU is found
- Particularly effective when one GPU has a different VRAM capacity — put the transformer on the larger GPU (`cuda:0`)
- Can be combined with block swapping on `cuda:0`

---

## Model loading

### GGUF quantized models

Load a quantized transformer from a `.gguf` file instead of the default SafeTensors checkpoint. GGUF models are significantly smaller on disk and use less VRAM.

**Setting:** `ggufTransformerPath` — full path to the `.gguf` file

Leave blank to use the default full-precision SafeTensors model.

**Notes:**
- Only the transformer is quantized; the VAE and text encoder remain in their standard precision
- Quantization levels vary by file (Q4, Q5, Q8, etc.) — higher Q = closer to full quality, more VRAM
- Not all GGUF files are compatible; they must be exported from the correct LTX model architecture

### LoRA support

Load CivitAI-compatible LoRA weight files (`.safetensors`) and merge them into the transformer at inference time.

**Settings > LoRAs:**

1. Click **Add LoRA** and browse to a `.safetensors` file
2. Adjust **strength** (default 1.0; typical range 0.5–1.5)
3. Toggle enable/disable per-LoRA without removing it

Multiple LoRAs can be active simultaneously; their weights are merged additively. High combined strength (especially >1.5 total) may produce artifacts.

**Tips:**
- Lower strength (0.5–0.8) for subtle stylistic influence
- Higher strength (1.0–1.5) for strong character or style lock-in
- If outputs look oversaturated or distorted, reduce total LoRA strength

### Abliterated text encoder

Swaps the standard Gemma text encoder for an abliterated variant that has had its content-filtering directions removed. This allows prompt content that the standard encoder would refuse to process.

**Setting:** `useAbliteratedEncoder` (boolean, default off)

The encoder swap happens before text embeddings are generated; the rest of the pipeline (transformer, VAE, scheduler) is unaffected. Only enable this if you specifically need to bypass encoder-level filtering.

---

## Generation controls

### Inference steps

Controls how many distillation steps the model runs per generation. The LTX-Video distilled model is trained to produce good results in very few steps.

**Setting:** `distilledNumSteps` (integer, 1–8, default 4)

| Steps | Quality | Speed |
| --- | --- | --- |
| 1 | Draft / rough | Very fast |
| 2–3 | Usable | Fast |
| 4 | Good (default) | Normal |
| 6–8 | Best | Slow |

More steps beyond 8 are not meaningful — the distilled model is not designed for high step counts. For full-quality generation, 4 steps is generally the sweet spot.

### STG (Spatio-Temporal Guidance)

Spatio-Temporal Guidance is a classifier-free guidance variant applied at a specific transformer block. It influences the balance between motion coherence and prompt adherence.

**Settings:**
- `stgScale` — strength of the STG signal (default 1.0; range 0–3)
- `stgBlockIndex` — which transformer block to apply STG at (default 0; range 0–27)

**When to adjust:**
- Increase `stgScale` if generated motion is too subtle or the video lacks dynamics
- Decrease if the video looks distorted or overly stylised
- `stgBlockIndex` affects which structural level of the model the guidance targets; lower indices = coarser structure, higher = finer detail

### Negative prompt

Describe what you want the model to avoid. The negative prompt is encoded separately and used to steer generation away from those concepts.

**Usage:**
- In Playground: the negative prompt field is below the main prompt
- In Gap fill: expand the collapsible **Negative prompt** section below the main prompt textarea

**Examples:**
- `blurry, low quality, distorted`
- `text, watermark, logo`
- `fast motion, camera shake` (if you want stable shots)

Negative prompting is less powerful in distilled models than in full models — very short negative prompts work best.

### Seed locking

After each generation, the seed used is displayed in the Playground's settings bar. You can lock it to reproduce the same (or very similar) output.

**Usage in Playground:**
1. Generate a video — the seed appears as a pill in the bottom settings row
2. Click the pill to **lock** the seed (Lock icon)
3. Subsequent generations will use the same seed
4. Click again to **unlock** (Shuffle icon) and return to random seeds

**Notes:**
- Seed alone doesn't guarantee identical output — prompt, settings, and model state all affect the result
- The locked seed is session-only; it resets when you close the app
- For persistent seed control, use the seed field in **Settings > Generation**

### Pipeline reload interval

The generation pipeline accumulates minor state over repeated runs. Periodically reloading it reclaims VRAM and prevents drift in output consistency.

**Setting:** `reloadPipelineEveryNGenerations` (integer, 0 = never reload, default 0)

Set to `5–10` if you run many generations in sequence and notice VRAM creeping up or output quality degrading over time.

---

## Prompt tooling

### Prompt enhancement

A local Gemma model rewrites your short, natural-language prompt into a detailed, structured prompt optimised for LTX-Video. It adds motion cues, preserves any dialogue, and removes filler words while adding descriptive detail.

**Usage:**
1. Type a short prompt in the Playground
2. Click **Enhance** (sparkle icon) before or instead of **Generate**
3. The enhanced prompt replaces your input (the original is saved in sidecar metadata)
4. Edit the enhanced prompt if needed, then generate

**Requires:** local Gemma model (downloaded on first use) and enough RAM to run it alongside the video model. If VRAM is tight, enhancement runs on CPU.

### Prompt history

The last 50 prompts are stored in your browser's localStorage and persist across sessions.

**Usage:**
1. Click the **History** icon next to the prompt field in the Playground
2. Browse previous prompts — click one to restore it
3. History is per-browser-profile and is not synced across machines

### Sidecar metadata

Every generated video is accompanied by a `.json` file with the same name in the same folder. It records:

```json
{
  "prompt": "the original prompt you typed",
  "enhanced_prompt": "the rewritten prompt (if enhancement was used)",
  "settings": { "model": "fast", "duration": 5, "resolution": "540p", ... },
  "seed_used": 123456789,
  "timestamp": "2026-03-21T12:00:00Z"
}
```

This lets you reproduce or reference any generation without having to remember what settings you used.

---

## Multi-frame conditioning

Condition the video on images at specific temporal positions: start frame, one or more middle frames, and/or end frame.

**Usage (Playground):**

The image conditioning panel replaces the single image uploader. It shows:

- **First frame** (always present) — drag/drop or click to upload an image that the video will start from
- **Middle frame(s)** — add up to 3 intermediate conditioning images at configurable timeline positions (e.g. 25%, 50%, 75% through the video)
- **Last frame** — upload an image the video should end on

Each slot has:
- A **strength slider** (0–1) controlling how strongly the model anchors to that image
- A **position slider** (middle frames only) controlling when in the video the frame appears
- An **Extract from video** button to pull a frame from an existing video file

**How it works:**

Each image is resized to match the generation resolution, then passed to the model as a `VideoConditionByLatentIndex` input — the image is injected at the corresponding latent frame index (not the raw frame index). The 8:1 temporal compression ratio of LTX-Video means a 24fps, 5-second video (121 frames) has 16 latent frames; conditioning positions are mapped accordingly.

**Tips:**
- First + last frame conditioning (the "bookend" approach) is the most effective way to control motion arc
- Middle frames work best for constraining a specific pose or composition at a known point in the video
- Reduce strength (0.5–0.7) for looser interpretation; full strength (1.0) enforces the frame closely but may reduce motion naturalness
- If only one conditioning image is set, it behaves identically to standard image-to-video

---

## Generation queue

Queue multiple generation jobs from the Playground and let them run sequentially without blocking the UI.

**Usage:**
1. Set your prompt and settings
2. Click **+Q** (Add to Queue) instead of **Generate**
3. The job appears in the **Queue panel** (bottom of the Playground)
4. Jobs run one at a time in submission order
5. Completed results appear in the gallery as normal
6. Cancel a pending job by clicking **×** next to it in the queue

**Notes:**
- Only one job runs at a time (the GPU can't run two simultaneously)
- The queue persists while the app is open; it resets on restart
- Running jobs can be cancelled, which stops generation and removes the job

---

## Retake

Replace a specific time range within an existing video clip while preserving the surrounding content.

**Usage (Video Editor):**
1. Switch to **Retake mode** in the mode selector or right-click a clip and choose **Retake**
2. Select a clip — a trim panel appears showing the clip's timeline
3. Drag the **in** and **out** handles to mark the region to replace
4. Type a prompt describing what should appear in that region
5. Click **Generate** — the new content is generated and inserted at the marked region

**How it works:**

Unlike image-to-video (which uses one frame as a starting anchor), Retake encodes the **entire source video to latent space** and then uses a `TemporalRegionMask` to mark the region for regeneration. The latents of the frames *outside* the mask remain unchanged and act as boundary conditions on both sides of the cut, guiding the model toward continuity at the edit points.

This means:
- No manual frame stitching required — the model blends naturally
- Motion direction and visual style are implicitly inherited from the surrounding frames
- The output video is the same length and resolution as the input

**Requirements:**
- Source video frame count must satisfy `8k + 1` (e.g. 97, 193, 289 frames). Videos generated by this app always meet this requirement.
- Width and height must be multiples of 32
- Minimum retake region: 2 seconds
- The source file must be accessible locally

**Limitations:**
- Only one contiguous region per retake
- Cannot retake a clip that was itself generated by a retake (this limitation will be lifted in a future version)
- Always runs the distilled model at 40 steps (not configurable from the UI currently)

---

## Gap fill

Click any empty space between clips on the timeline to generate content that fills the exact gap duration.

**Usage (Video Editor):**
1. Click a gap between two clips — a small context menu appears
2. Choose **Fill with Video** or **Fill with Image**
3. The gap generation modal opens, pre-filled with an AI-suggested prompt based on the neighbouring clips (requires a Gemini API key)
4. Edit the prompt if needed, optionally set a negative prompt, and adjust generation settings
5. Click **Generate** — the modal closes and generation runs in the background
6. The result is automatically placed in the gap when complete

**Image-to-video conditioning:**
- In the timeline visualisation at the top of the modal, click **Start frame** or **End frame** to use the adjacent clip frame as a conditioning image
- You can also replace either frame with your own image using the upload button

**AI prompt suggestions:**
- The app automatically extracts the last frame of the clip before the gap and the first frame of the clip after the gap
- These are sent to Google Gemini along with the neighbouring clips' prompts to generate a contextually appropriate suggestion
- Click **Re-analyze** to generate a fresh suggestion
- Requires a Gemini API key (configure in Settings > API Keys)

**Negative prompt:**
- Click **Negative prompt** below the main prompt to expand the field
- A blue dot indicates an active negative prompt

---

## Clip Viewer (Source Monitor)

Inspect and trim any asset before adding it to the timeline. Useful for selecting the exact portion of a clip you want without cluttering the timeline.

**Opening the Clip Viewer:**
- Double-click any asset in the asset gallery to load it
- Right-click an asset → **Open in Clip Viewer** — loads it without placing it on the timeline
- The viewer appears as a split pane to the left of the program monitor; drag the divider to resize

**Playback controls (bottom transport bar):**
- Play / Pause / Stop
- Step one frame forward or back (frame-accurate)
- Play in reverse
- Go to In point / Go to Out point

**Setting In and Out points (3-point editing):**
1. Scrub or play to the frame where you want the clip to start
2. Click **Set In** (bracket icon) or press `I`
3. Scrub to the end of the desired portion
4. Click **Set Out** (bracket icon) or press `O`
5. The selected range is highlighted in blue on the scrub bar; the In/Out timecodes and total duration are shown below it
6. Click either marker again (or drag it) to adjust; click when the playhead is on top of the marker to clear it

**Adding the trimmed clip to the timeline:**

There are three ways once In/Out are set:

| Method | Effect |
| --- | --- |
| **Insert Edit** (`,` key or + button in transport bar) | Ripples clips on the target track forward to make room, then inserts the trimmed clip at the current playhead position |
| **Overwrite Edit** (`.` key or ■ button in transport bar) | Places the trimmed clip at the playhead position, overwriting whatever is underneath |
| **Drag to timeline** | Drag directly from the viewer's video area and drop onto any track — the clip lands at the drop position with the In/Out trim already applied |

Right-clicking the video area also gives **Insert at playhead** and **Overwrite at playhead** options.

**Target track selection:**
- Insert and Overwrite target the first unlocked, source-patched video track (and a matching audio track for video assets)
- Drag-to-timeline drops onto whichever track you release on

**Notes:**
- If no In point is set, the clip starts from frame 0
- If no Out point is set, the clip runs to the end of the asset
- Images loaded in the viewer are supported but In/Out points only apply to video and audio assets (no scrub bar is shown for images)
- The playhead in the Clip Viewer is independent of the timeline playhead
