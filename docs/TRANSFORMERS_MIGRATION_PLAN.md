# Shared Transformers migration plan

Requested: 2026-10-07. Implementation: Luna at xhigh reasoning, with parent-agent review.

## Outcome

Run the existing standard TTS models and the local native `VibeVoice-ASR-HF` checkpoint from the same project virtual environment. Preserve streaming TTS and `--lod`, local model settings, voice/model native-select interactions, automatic generation length, and the intentional removal of AI Chat. Finish the selected features: ASR upload plus Transcribe button, a request seed, and short `[N]` speaker labels.

## Steps

1. **Establish the shared dependency contract.** Use a pinned Transformers 5.x release that includes native VibeVoice ASR (5.3.0 is the initial target). Align `requirements.txt` and `pyproject.toml`, install into the existing project `venv`, and check related dependency compatibility. Preserve PyTorch/CUDA and avoid touching ComfyUI's environment or model weights.
2. **Adapt standard TTS to the shared release.** Update tokenizer imports, configuration/dtype handling, checkpoint initialization and loading, generation helper calls, and KV cache access where the selected version requires it. Use the installed ComfyUI node as a source reference; keep all runtime code standalone. Preserve the existing generation algorithm and serialized checkpoint behavior.
3. **Simplify ASR to the project environment.** Remove the separate ASR setup script, requirements overlay and environment requirement. Keep the disposable ASR worker for memory cleanup, launched with the current project Python. Maintain local-only ASR model loading, separate discovery/UI, upload/context input, readable transcript and valid segment JSON, error reporting, and serialization of model operations.
4. **Verify the shared paths.** Run the full project tests plus focused compatibility tests using tiny synthetic models/checkpoints to exercise actual construction, save/load, processor setup and generation/cache behavior without loading large weights. Check ASR class imports and processor loading from the configured local checkpoint without inference. Verify seed propagation in streaming and worker paths, speaker/voice mapping, and UI callback wiring. Do not run full speech inference unless separately requested; report that limit explicitly.
5. **Review and document.** Parent reviews the patch and test evidence. Update README and the gaps doc with current shared-runtime setup and any remaining limits. Check the diff and leave changes uncommitted for user review.

## Acceptance

- One project environment and one pinned Transformers version support TTS and native ASR imports and tested loading/generation seams.
- ASR has an upload field and Transcribe button, and the worker provides a transcript plus parseable timestamp/speaker segment JSON when the model emits structured output.
- Seed defaults to 42; zero resolves to a random positive seed displayed in the log. The same resolved seed reaches both standard streaming and `--lod` generation.
- `[1] Text` and `[1]: Text` map to the first selected voice. Long labels, legacy `Speaker 0:` scripts, plain-text rotation and labelled continuation text remain supported. Invalid labels and unselected speakers produce errors.
- Length remains automatic. No `.env` changes, weight downloads/copies, ComfyUI dependency changes, AI Chat restoration, or commits.

## Starting state

The first feature implementation already added ASR controls/worker, seeds, speaker parsing and tests; 40 tests passed under Transformers 4.51.3. Its separate `venv-asr` setup was attempted but dependency installation was declined. The user subsequently explicitly authorized migration to a shared Transformers environment, superseding the separate-environment approach. A default-sandbox install of Transformers 5.3.0 failed due to network restrictions; migration installation will require the normal escalation flow. The ignored partial `venv-asr` directory can remain unused; do not recursively delete environments as incidental cleanup.

The paragraph above records the starting state. The shared installation was later completed in the project `venv` using the normal escalation flow. The unused ignored `venv-asr` directory was left in place.

## Implementation status and validation

- `requirements.txt` and `pyproject.toml` both pin `transformers==5.3.0`; the separate ASR setup script and dependency overlay were removed. `vibevoice/asr.py` launches the disposable worker using the current project's Python executable.
- TTS construction and `save_pretrained`/`from_pretrained` round-trips pass with tiny synthetic checkpoints for both tied and untied Qwen embeddings. The test compares every state-dict tensor and checks that tied pointers are reconstructed while distinct untied weights remain distinct.
- Tiny autoregressive generation reaches speech-diffusion and EOS paths in both negative-cache modes. The tests observe actual Transformers 5 `DynamicCache` mutations, including a second speech turn that exercises cache refresh after prior negative-model forwards.
- The full test suite passes: `python -m unittest discover -s tests -q` (45 tests). `python -m pip check` reports no broken requirements, and `git diff --check` reports no whitespace errors.
- The configured local native ASR processor loads through `AutoProcessor` with `local_files_only=True`; processor setup was checked without loading ASR model weights or running inference. The local TTS config/tokenizer/processor path was also checked without loading TTS weights.
- No full speech inference, quality benchmark, or GPU-memory benchmark was run. These remain outside the authorized migration validation; the UI callbacks are covered by tests and a mock-backed browser check is handled separately.

## Q8 loading follow-up (2026-10-07)

The first migration missed a change in bitsandbytes exclusion matching: Transformers 5 matches full module paths, while the existing Q8 checkpoint lists short speech-component names such as `acoustic_tokenizer`. Nested speech linear layers were consequently replaced by INT8 layers even though their saved weights are BF16, producing `'Parameter' object has no attribute 'CB'` during voice encoding.

The shared loader now preserves those exclusions and adds their `model.*` paths in the in-memory quantization config. It updates the loaded model config because Transformers prioritizes serialized quantization settings over loading kwargs. Checkpoint files remain unchanged. A tiny mixed BF16/INT8 checkpoint reproduces the exact original failure with the old exclusions; the corrected loader passes CUDA voice encoding and language-model/output-head forward execution, preserves every speech-component tensor, and retains genuine INT8 language-model weights. The regression requires CUDA and skips on machines without it; no full-size Q8 speech generation was run.

Follow-up validation: all 46 tests pass, including the CUDA regression; `pip check` and `git diff --check` pass.
