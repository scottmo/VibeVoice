# ComfyUI VibeVoice vs. Standalone VibeVoice: Feature Handoff

## Purpose

Implementation update (2026-10-07): the selected follow-up adds ASR upload/transcription controls, a per-request seed in both TTS paths, and `[N]` speaker labels while retaining the long form. Generated length remains automatic. Native ASR and TTS now share the project virtual environment with Transformers 5.3.0; ASR still runs in a disposable worker process for memory cleanup. The local ASR processor has been loaded successfully without downloading weights or running full inference. The comparison below remains the historical pre-implementation audit. See the README's shared TTS and ASR environment section for setup.

This document records the feature differences between the installed ComfyUI
custom node and the standalone VibeVoice app in this checkout. It is intended
to give a future agent enough context to investigate or implement a specific
follow-up without repeating the initial comparison. The list is an inventory,
not a decision that every ComfyUI feature should be copied into the app.

“ComfyUI” below means the installed source at
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice`. “Standalone” means
this repository, `C:\Users\max61\Home\workspace-code\VibeVoice`.

## Audit context

- Reviewed: 2026-10-06.
- Standalone Git revision: `bb8242734d4744cc8e37100ca2f6f1be687e91b3`
  (`main`, clean at review).
- ComfyUI custom-node Git revision: `774c1aeeb374bdb6312f68be1be3ad9e781ed557`
  (`main`, clean at review).
- Scope: static comparison of the checked-out source and the reported local
  asset layout. No speech inference was run, and runtime support, output
  quality, speed, and memory use were not benchmarked. No tests were run for
  this documentation-only audit.
- Software snapshot queried read-only from the standalone `venv` on
  2026-10-06: Python 3.11.9, Gradio 5.50.0, Transformers 4.51.3, and Torch
  2.7.1+cu128. This is a point-in-time environment snapshot; a future agent
  should recheck installed versions before integration work.
- The configured model root for this machine is reported as
  `F:\Apps\vibevoice_models`, with `tts\`, `tokenizers\`, and
  `vocal_isolation\` directories. The TTS assets include `VibeVoice-1.5B`,
  `VibeVoice-7B`, `VibeVoice-Large-Q8`, and `VibeVoice-ASR-HF`. This describes
  model assets, not source-code locations. The `.env` file was not opened for
  this comparison.

The main standalone entrypoints and model-loading implementation are
[`main.py`](../main.py) and
[`vibevoice/model_loading.py`](../vibevoice/model_loading.py). The ComfyUI
entrypoints are `nodes/tts_node.py`, `nodes/asr_node.py`, and
`nodes/external_loader_node.py` inside the installed custom-node directory;
their shared logic is in `modules/` there. Links to the standalone files below
are relative to this document. ComfyUI paths are absolute because that source
is outside this repository.

## What already overlaps

The standalone app already does standard VibeVoice text-to-speech. Its model
resolver supports complete local checkpoint directories and Hugging Face model
repositories; the model list includes the 1.5B and 7B families and configured
quantized selections such as Large-Q8 and the legacy 7B 4-bit option. It uses
the standard `VibeVoiceForConditionalGenerationInference` model and
`VibeVoiceProcessor` pair. See `DEFAULT_MODEL_REPOSITORIES`,
`validate_tts_model()`, `resolve_model()`, and
`load_model_and_processor()` in [`vibevoice/model_loading.py`](../vibevoice/model_loading.py)
(roughly lines 19–58, 268–282, 390–420, and 495–608).

Both products have standard multi-speaker generation, reference-voice cloning,
CFG guidance, diffusion-step count, sampling, temperature, top-p, and top-k
controls. The standalone UI allows up to four selected speakers and can stream
audio chunks while standard generation is running. Its worker-based `--lod`
mode returns the completed audio after generation instead. These shared
features are not gaps to recreate; the gaps below concern controls, input
surfaces, model families, or workflows that differ.

## Comparison matrix

| Capability | Standalone status | Practical difference |
|---|---|---|
| Standard 1.5B/7B multi-speaker TTS and voice cloning | Present | Core generation path exists; voice references are chosen from discovered local audio presets. |
| Per-speaker ComfyUI `AUDIO` graph inputs | Missing | Standalone has no per-generation graph/audio socket; adding a file to its voice-preset folders is the current reference workflow. |
| ASR transcription and timestamped/speaker-labelled segments | Missing | ComfyUI has a separate ASR node and model family; standalone validation explicitly rejects ASR checkpoints as TTS. |
| VibeVoice-Realtime-0.5B architecture and cached prompt presets | Missing | This is a distinct model and generation path, not ordinary TTS streamed to the browser. |
| External single-weight-file loader | Missing | Standalone expects a validated checkpoint directory; ComfyUI has a separate node for `.safetensors`, `.bin`, and `.gguf` files with config/tokenizer resolution. |
| Quantization | Partial | Standalone consumes supported checkpoint quantization configuration and has named model selections; it has no Comfy-style per-load LLM 4-bit toggle or Comfy external-file loader. |
| Per-generation seed | Missing | Standalone calls `set_seed(42)` at app startup but exposes no seed input and does not pass a request seed through its normal and worker paths. |
| Explicit maximum generated-token budget | Missing in UI/plumbing | Comfy exposes `max_new_tokens`; standalone generation paths pass `None`. The model API has a token-budget concept, so a future change can investigate wiring it through. |
| `[N]` speaker marker syntax | Partial | Comfy parses `[1]` and `Speaker 1:` through a normalizer; standalone preprocessing preserves only lines beginning with case-sensitive `Speaker ` and containing `:`, and auto-assigns other lines. |
| Explicit dtype/attention/device choices | Partial | Comfy exposes selectors and availability checks. Standalone exposes device choice, automatically chooses attention, and uses its model-loading defaults/fallback. |
| ComfyUI shared VRAM manager and patcher lifecycle | Platform-specific | ComfyUI uses its `ModelPatcher` integration. Standalone keeps a model resident or unloads the worker process in `--lod` mode. |
| Wider CFG/steps/top-k ranges | Partial | ComfyUI widgets permit wider values than the standalone sliders; a wider numeric range does not itself establish better output. |
| Browser audio controls and output handling | Platform-specific | Standalone has Gradio streaming/completed outputs, stop, save, trim/playback, and gain UI. The Comfy node returns an `AUDIO` value to the graph. |

## Detailed differences and follow-up notes

### 1. ASR is a separate ComfyUI capability

ComfyUI defines `VibeVoiceASRNode` in
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice\nodes\asr_node.py`.
Its `define_schema()` (about lines 49–161) accepts an audio input and context
text/hotwords, and exposes token budget, temperature, top-p, sampling, beams,
device, dtype, attention, and force-offload controls. `execute()` (about lines
213–300) returns a transcription string and a segments JSON string. The node
uses the dedicated ASR model registry and patcher/loader path. The schema
description says “50+ languages” and “up to 60 minutes”; those are UI claims,
not performance or accuracy results from this audit.

Standalone `validate_tts_model()` in
[`vibevoice/model_loading.py`](../vibevoice/model_loading.py#L268) rejects an
ASR-named or ASR-architecture directory, as well as a config whose model type
is not `vibevoice`. The shared loader constructs only
`VibeVoiceForConditionalGenerationInference` with `VibeVoiceProcessor`; there
is no ASR model class, processor, transcript UI, or segment output in this
app. The local `tts\VibeVoice-ASR-HF` asset is therefore not a feature already
wired into the standalone app.

**Possible integration seam:** make ASR a distinct app capability with its
own compatible model validation/loading and input/output UI. Keep it out of
the TTS model selector: ComfyUI also separates `get_asr_models()` from
`get_tts_family_models()` in
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice\modules\model_info.py`
(about lines 141–174). Do not assume TTS checkpoint validation or processor
setup can load ASR. If this is selected for implementation, a useful acceptance
check is a transcript plus parseable segment JSON for a supplied sample,
including timestamps/speaker IDs where the model emits them; any runtime
accuracy or duration claim needs its own explicit validation.

### 2. Realtime TTS is a different model family

ComfyUI registers `VibeVoice-Realtime-0.5B` as `streaming_tts`, separate from
standard TTS and ASR, in
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice\modules\model_info.py`
(about lines 20–40). `VibeVoiceTTSNode` dispatches to `_generate_realtime()`
for that family (`nodes/tts_node.py`, about lines 311–426 and 519–570), which
calls `generate_realtime_audio()` in `modules/realtime_generation.py`. That
path expects the realtime model/processor pair and a cached `.pt` voice prompt
from `modules/voice_presets.py`. It is single-speaker; speaker reference audio
is ignored. The node also warns that the standard sampling widgets are not
used by the realtime generation loop.

Standalone loading validates standard TTS directories and instantiates only
the standard inference model and processor. It has no realtime architecture
loader, cached prompt adapter, or `.pt` prompt selector. Streaming audio chunks
from the standalone standard model through Gradio does not add the distinct
realtime architecture. ComfyUI's node output is a completed `AUDIO` graph
value (`nodes/tts_node.py`, about lines 245–247), even though its internals
have a realtime generation path; it is not an equivalent browser live-PCM
interface.

**Possible integration seam:** this is a larger, separate model-family feature.
First determine which user-visible outcome is desired: using the realtime
checkpoint and its cached voice prompts, or delivering live playback in the
browser, or both. Treat its model loader, prompt format, single-speaker
semantics, and output plumbing as a separate design; do not route it through
the standard TTS class and call it realtime. Acceptance should establish the
selected model loads from the intended local asset, the chosen prompt is used,
and the requested output behavior works in the target UI.

### 3. ComfyUI can load a standalone weight file

`VibeVoiceExternalLoaderNode` in
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice\nodes\external_loader_node.py`
(about lines 46–160 and 220–306) discovers files under ComfyUI's model folders
and exposes a config selector with auto-detection. Its companion
`modules/external_loader.py` supports standalone `.safetensors`, `.bin`, and
`.gguf` weight files, config/tokenizer/preprocessor sidecars or packaged
defaults, and quantized/FP8/GGUF loading paths. Consult that loader's format
and config-resolution helpers before using it as a behavioral reference.

Standalone `validate_tts_model()` requires a directory with `config.json` and
supported direct weight filenames or a Hugging Face shard index. This is a
directory-checkpoint contract, not a generic raw-file loader. A `.safetensors`
extension by itself does not make an arbitrary file loadable in the standalone
app.

**Possible integration seam:** a future file loader would need a specific
format contract, model architecture/config resolution, tokenizer and
preprocessor lookup, error handling, and local-only behavior. Keep this path
separate from directory discovery unless the chosen user workflow calls for
unification. Acceptance should cover each explicitly selected format and
missing/ambiguous sidecars; do not infer support for a format from its
extension alone.

### 4. Quantization exists in both places, with different controls

The standalone resolver includes a named legacy 7B 4-bit selection and a
Large-Q8 selection in `DEFAULT_MODEL_REPOSITORIES` in
[`vibevoice/model_loading.py`](../vibevoice/model_loading.py#L19). Its
`_quantization_config()` and `load_model_and_processor()` (about lines
495–608) apply serialized bitsandbytes 4-bit or 8-bit configuration when a
checkpoint provides it. The code defaults to BF16, and uses FP16 for the
4-bit path. This is source-level support; no inference was run here to verify
the local Q8 or bitsandbytes runtime.

ComfyUI additionally exposes `quantize_llm_4bit` on the TTS node (about lines
120–128 of `nodes/tts_node.py`), described as NF4 on the language model while
the diffusion head remains higher precision. Its external loader also has
specialized quantized file paths. This is not evidence that standalone lacks
all quantization; the gap is the control surface and the other loader formats.

**Possible integration seam:** decide whether the intended request is an
interactive LLM-only 4-bit toggle, support for more checkpoint-serialized
quantization, or file-format loading. These are separate capabilities with
different dependencies. Preserve a clear error when a required quantization
backend is unavailable and verify the exact target configuration before
claiming runtime support.

### 5. Per-generation seed and length controls are absent in standalone UI

ComfyUI's standard TTS schema exposes `seed` (default 42, with zero meaning a
random seed) and `max_new_tokens` (zero means automatic) in
`nodes/tts_node.py` (about lines 157–201). The node passes both through its
standard generation call (about lines 468–509); the generation API accepts
them in `modules/generation.py` (about lines 359–389).

Standalone `main()` calls `set_seed(42)` once at startup (`main.py`, about
line 2658), but the Gradio controls do not expose a per-request seed. The
standard streamer passes `max_new_tokens=None` in `_generate_with_streamer()`
(`main.py`, about lines 1549–1594); `_generate_with_worker()` also sends no
token budget in its worker request (about lines 1596–1610), and the worker's
generation call uses `None` (about lines 452–500). A once-at-startup seed is
not the same interface as setting a seed for each generation, and does not by
itself promise identical outputs on repeated requests.

**Possible integration seam:** thread a per-request seed and optional token
budget through the UI callback, standard generation function, and worker
message/API. Keep the no-cap/automatic default compatible with existing
behavior. Acceptance should show that the requested values reach both normal
and `--lod` worker paths and that the cap is honored; document any limits on
repeatability instead of promising cross-device bit-for-bit identity.

### 6. Speaker-script parsing differs

ComfyUI's `parse_script_1_based()` in
`F:\Apps\ComfyUI\ComfyUI\custom_nodes\ComfyUI-VibeVoice\modules\audio_utils.py`
(about lines 80–139) parses `Speaker N:` and `[N]` forms case-insensitively,
converts these 1-based labels to model speaker IDs, and normalizes parsed
turns before generation. The Comfy TTS schema advertises both marker forms
(`nodes/tts_node.py`, about lines 107–120).

Standalone `generate_podcast_streaming()` in [`main.py`](../main.py#L1079)
checks each input line with a case-sensitive `line.startswith('Speaker ')`
and a colon test (about lines 1200–1225). Lines matching that simple check
pass through; other non-empty lines are auto-assigned in rotation. It does
not use the Comfy normalizer, so `[1] ...` is not converted by this step and
may be passed onward as literal text. The processor's own speaker parsing is
a separate layer, so inspect it before changing label semantics.

**Possible integration seam:** consider a shared standalone normalization
function that accepts the desired marker forms and produces the exact format
expected by the current processor. Preserve plain-text auto-assignment and
existing labels/continuation text. Explicitly test case variants, `[N]` with
or without a colon, malformed labels, speaker numbering, blank lines, and
more speakers than selected. Do not assume Comfy's parser can be copied
without checking its different graph-input mapping and fallback behavior.

### 7. Reference voices use different input workflows

ComfyUI accepts optional `speaker_1_voice` through `speaker_4_voice` audio
inputs on the TTS node (`nodes/tts_node.py`, about lines 229–236). The standard
path maps parsed speaker IDs to those inputs in `_generate_standard()` (about
lines 468–509). The same node also accepts an `external_model` bundle. This
lets an audio graph supply reference samples directly for a generation.

Standalone discovers audio files under `demo/voices` and `custom_voices` in
`setup_voice_presets()` and `_scan_voice_directory()` (`main.py`, about lines
848–909), then uses the existing selectors to choose one per active speaker.
Voice cloning is already present, but there is no per-generation audio upload,
recording control, or graph `AUDIO` input.

There is a documentation/implementation caveat in the installed Comfy node:
its schema tooltip says speakers without their own reference are cloned from
provided references (`nodes/tts_node.py`, about lines 107–120), while the
standard execution path filters out missing references before passing the
remaining samples to the processor (`modules/generation.py`, about lines
424–469). Do not promise automatic fallback or reuse mapping for missing
references until the generation implementation and a real graph case confirm
it.

**Possible integration seam:** add an optional per-speaker upload or recording
path only if that direct-input workflow is desired. Reuse the current voice
selectors and local preset folders where they meet the user's needs. Define
how missing references are handled for multi-speaker scripts before coding;
acceptance should cover a reference for every active speaker and missing or
extra references.

### 8. Runtime controls and memory lifecycle differ

ComfyUI's TTS node exposes `device`, `dtype`, `attention_mode`, and
`force_offload` controls in `nodes/tts_node.py` (about lines 204–222). The
available attention list is derived from installed backends and compatibility
checks. Its `quantize_llm_4bit` switch is separate from these controls.

Standalone provides `--device` and `--lod` in `parse_args()` (`main.py`, about
lines 2580–2635). The device selects an automatic attention implementation;
`load_model_and_processor()` defaults to BF16, and retries with SDPA when a
non-SDPA attention implementation fails (`vibevoice/model_loading.py`, about
lines 525–608). There is no user-facing CLI/UI override for arbitrary dtype or
attention mode and no generic Comfy-style 4-bit toggle. Named/serialized
quantization still exists as described above.

ComfyUI's `modules/patcher.py` defines `VibeVoicePatcher` as a ComfyUI
`ModelPatcher` subclass and integrates model loading, offload, caching, and
device placement with ComfyUI's manager (about lines 22–210). Standalone
either retains the model in the app or uses `--lod` multiprocessing to load
and later terminate a worker process. Both can free VRAM, but the worker
process is not ComfyUI's shared VRAM manager or partial offload lifecycle.

**Possible integration seam:** expose only controls with a clear standalone
meaning and validate availability on the actual device. Do not transplant
ComfyUI patcher classes directly: imports and lifecycle calls are
ComfyUI-bound. Adapt the user-visible intent to the standalone loader and
report unsupported combinations clearly.

### 9. ComfyUI numeric ranges are wider

Current widget ranges in Comfy `nodes/tts_node.py` and standalone `main.py`
are:

| Parameter | ComfyUI | Standalone |
|---|---:|---:|
| CFG scale | 0.1–50 | 1–2 |
| Diffusion steps | 1–500 | 5–30 |
| Temperature | 0–2 | 0.1–1.5 |
| Top-p | 0–1 | 0–1 |
| Top-k | 0–500 | 0–100 |

The standalone controls are in `main.py` around lines 2140–2182; the Comfy
schema is around lines 139–193 of `nodes/tts_node.py`. Defaults also differ
for CFG: 1.3 in ComfyUI and 1.6 in standalone. If expanding ranges, preserve
the current defaults and add model-specific guardrails or explanatory
behavior where evidence supports them. A larger range is parity, not a
quality improvement guarantee.

## Differences that are not automatically gaps

- **ComfyUI graph-only concerns:** custom node sockets, graph bundle typing,
  Comfy model-folder registration, and `ModelPatcher` lifecycle exist to fit
  ComfyUI's host. Port only the user-facing outcome that is relevant to the
  standalone app.
- **Standalone app conveniences:** source includes optional vocal isolation
  and voice normalization, a negative-prompt field, save/stop handling, and
  playback/trim/gain controls. These are app features rather than Comfy parity
  requirements. Static inspection does not validate each runtime path.
- **AI Chat:** its removal was intentional per the user's prior direction. Do
  not list it as a feature gap or restore it as part of unrelated work.
- **Model files:** `F:\Apps\vibevoice_models` is an asset root; the source
  checkouts above are code repositories. Do not move or duplicate large model
  weights just to implement a UI or loader change.

## Suggested order if the user asks for parity work

This is a suggested sequence only; a later user's selected feature scope takes
priority.

1. **Small control/input changes:** per-generation seed and token budget;
   parser normalization. They have relatively narrow seams and can be
   validated without changing the model architecture.
2. **Direct voice reference input:** add upload/recording only if the desired
   workflow is clear; define missing-speaker reference behavior first.
3. **Runtime controls:** expose dtype/attention/quantization choices only
   after defining supported values for this Windows setup and preserving safe
   defaults.
4. **Separate model workflows:** ASR, realtime TTS, and standalone external
   single-file loading deserve independent designs and acceptance criteria.
   They have distinct model classes, processors, formats, dependencies, and
   user interfaces.

## Notes for the next agent

- Follow the feature scope in the later user request; this inventory itself
  does not select or authorize an implementation. A clear implementation
  request is authorization. Ask for clarification only if the requested
  feature or its expected behavior is ambiguous.
- The user prefers a planner/orchestrator workflow for coding work, with
  implementation delegated to an agent using Luna at xhigh or max reasoning
  when available. Follow that preference when the task setup permits it.
- Preserve the current local model configuration and model root, existing
  native-select model and voice selector interactions, and the intentional AI
  Chat removal.
- Use the standalone project's isolated virtual environment and pinned
  dependencies. Avoid importing ComfyUI packages or changing the ComfyUI
  environment to satisfy standalone dependencies.
- Keep ASR model discovery and controls separate from the TTS selector.
- Do not download, install, or copy model weights unless a later user request
  clearly includes that work. Avoid changing `.env` or exposing its contents.
- Speech generation was left to the user for this comparison. A later
  explicit request to run an audio test overrides that current-session
  preference; otherwise use static checks or non-inference validation and
  state the limits clearly.
- Do not commit changes unless the user asks for a commit.
