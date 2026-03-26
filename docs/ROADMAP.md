# Roadmap

Planned features, known gaps, and contribution opportunities.

Items are roughly ordered by priority within each section. If you want to work on something, open an issue first so we can coordinate and avoid duplicate effort.

---

## Status key

| Symbol | Meaning |
| --- | --- |
| ✅ | Shipped |
| 🔨 | In progress |
| 📋 | Planned — design clear, ready to build |
| 💡 | Exploratory — needs investigation before scoping |
| ❓ | Uncertain — depends on upstream model support |

---

## Currently shipped (this fork)

| Feature | Notes |
| --- | --- |
| ✅ Block swapping | Configurable via Settings |
| ✅ Attention tiling | Configurable via Settings |
| ✅ VAE tiling | Configurable via Settings |
| ✅ FP8 transformer | Configurable via Settings |
| ✅ Multi-GPU split | cuda:0 transformer / cuda:1 text encoder |
| ✅ GGUF model loading | Alternative to SafeTensors |
| ✅ LoRA support | CivitAI-compatible, multi-LoRA |
| ✅ Abliterated text encoder | Gemma-based, encoder-level bypass |
| ✅ Configurable inference steps | 1–8 distillation steps |
| ✅ STG (Spatio-Temporal Guidance) | Scale + block index controls |
| ✅ Negative prompt | Playground + gap fill modal |
| ✅ Seed locking (Playground) | Display + toggle per session |
| ✅ Pipeline reload interval | Every N generations |
| ✅ Prompt enhancement | Local Gemma model |
| ✅ Prompt history | localStorage, last 50 |
| ✅ Sidecar metadata | .json alongside each video |
| ✅ Multi-frame conditioning | First / middle / last frame with per-slot strength |
| ✅ Generation queue | Backend queue + Playground queue panel |
| ✅ Retake | Full latent-space temporal conditioning |
| ✅ Gap fill | Text-to-video / image-to-video / text-to-image |
| ✅ Gap fill negative prompt | Collapsible field in gap modal |
| ✅ Clip Viewer (Source Monitor) | 3-point editing with In/Out, Insert/Overwrite, drag-to-timeline, right-click send |
| ✅ Gallery clip management | Star ratings (1–5), free-text notes, colour labels (10 colours), bins (folders), filter bar |
| ✅ 0xC0000005 crash fix | `torch.cuda.synchronize()` around dual-frame conditioning prevents Windows STATUS_ACCESS_VIOLATION |

---

## Near-term (high priority)

### Distribution

**📋 Portable installer**

Build a self-contained Windows installer (`.exe`) or portable zip that bundles Node, the Python embed environment, and all ML dependencies. Users should be able to install and run without pre-installing Node, uv, or Python.

- `scripts/prepare-python.ps1` already builds the embedded Python environment
- `scripts/local-build.ps1` already produces an unpacked Electron build
- Work needed: wire them together into a single distributable, test auto-update flow, document model-weight placement for end users
- Relevant files: `scripts/prepare-python.ps1`, `scripts/local-build.ps1`, `electron-builder.yml`

---

### GenSpace parity with Playground

The GenSpace (asset gallery) view is missing several controls that Playground has. These need to be wired in for a consistent experience.

**📋 GenSpace queue integration**
- Add a `+Q` button to the GenSpace prompt bar
- Show the QueuePanel below the prompt bar (same as Playground)
- The backend queue API is already in place; this is frontend wiring only
- Relevant files: `frontend/views/GenSpace.tsx`, `frontend/hooks/use-queue.ts`

**📋 GenSpace inline generation controls**
- Expose steps, STG scale, STG block index, and pipeline reload interval directly in the GenSpace prompt bar (or a collapsible panel below it)
- Currently these settings only take effect if configured globally in the Settings Modal
- Relevant files: `frontend/views/GenSpace.tsx`

**📋 GenSpace seed display**
- The seed capture via `onGenerationSuccess` is wired; the PromptBar props are plumbed — but the GenSpace PromptBar settings type needs updating to surface the seed pill in video mode
- Relevant files: `frontend/views/GenSpace.tsx` (PromptBar component, `DEFAULT_VIDEO_SETTINGS`)

---

## Medium-term

### Video editor — timeline controls

**📋 In/out point markers with keyboard shortcuts**

Mark in (`I`) and out (`O`) points on the timeline to define an export region or generation region, and clear them (`X`). Standard NLE keyboard shorthand.

- In/out state already exists in `VideoEditor.tsx` as `inPoint` / `outPoint`
- Need: visual markers on the ruler, keyboard handler wiring, clear action
- Relevant files: `frontend/views/VideoEditor.tsx`, `frontend/views/editor/useTimelineDrag.ts`

**📋 Export selected region**

Export only the in→out region to a file (via ffmpeg trim), rather than the full timeline.

**💡 Clip speed controls in the inspector**

Expose the `speed` and `reversed` clip properties in the clip inspector panel. Currently these fields exist on the clip model and are respected by the renderer but cannot be set from the UI.

### Multi-frame conditioning improvements

**📋 Better middle-frame UX**

The current middle-frame slot uses a linear position slider. Consider a visual timeline strip where the user drags a marker to the desired time position.

**📋 Retake retake support**

Retaking a clip that was itself generated by a retake is currently blocked. Remove the restriction once the pipeline handles this correctly (the source video is always a valid input regardless of how it was generated).

---

## Experimental / investigatory

### Video-to-Audio (V2A)

**💡 Generate audio from video**

The backend already has an `LTXa2vPipeline` skeleton. Investigate generating matching audio from an existing video using the LTX audio model. Would appear as a new mode in the video editor (right-click clip → Generate audio).

Dependencies:
- Full LTX audio model support in the backend pipeline
- Audio track wiring in the frontend editor
- Likely a separate settings panel for audio generation parameters

### First-frame-to-last-frame

**📋 Bookend generation**

Provide both a start image (frame index 0) and an end image (frame index -1) as simultaneous conditioning inputs. The model generates the motion between them. This is already supported by the multi-frame conditioning backend (`conditioningImages` list with `frameIdx: 0` and `frameIdx: -1`). The frontend multi-frame panel already supports this — it just needs UX confirmation that both first and last slots work correctly end-to-end.

**Testing needed:** verify temporal blending quality with full bookend conditioning at various durations.

### IC-LoRA

**💡 Instruction-conditioned LoRA**

The GenSpace view already has an `ic-lora` mode with conditioning type and strength controls. The backend pipeline support needs investigation. If the ltx_core supports IC-LoRA natively, this may be straightforward to wire in.

### Prompt enhancement in GenSpace

**📋 Expose prompt enhancement button in GenSpace**

Currently prompt enhancement is only available in Playground. GenSpace should have the same sparkle/enhance button in its prompt bar.

---

## Tech debt

| Item | File(s) | Notes |
| --- | --- | --- |
| Remove STG debug log line | `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py` | Leftover from STG development — `_stg_log.info(...)` inside `_stg_func` |
| Remove copy files | `backend/handlers/__pipelines_handler - Copy.py`, `backend/services/___block_swap_service - Copy.py` | Stale scratch copies, safe to delete |
| Fix `retake_pipeline` import warning | Startup logs | Module absent in this ltx version; suppress or conditionally import |
| Pre-existing TS errors in GenSpace / editor hooks | `frontend/views/GenSpace.tsx`, `frontend/views/editor/useRegeneration.ts`, `frontend/views/editor/useGapGeneration.ts` | Missing `seed` field and 6-arg `generate()` call mismatches |
| GenSpace `DEFAULT_VIDEO_SETTINGS` type completeness | `frontend/views/GenSpace.tsx` | Missing `seed`, full type annotation vs inline object |

---

## How to contribute

1. Check the list above — items marked **📋** are the most straightforward to pick up
2. Open an issue describing what you intend to build and roughly how
3. Wait for a response before investing significant time (to avoid conflicting work)
4. Follow the code style of the surrounding files — no large refactors without prior agreement
5. Run `pnpm typecheck` and `pnpm backend:test` before opening a PR

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for full details.
