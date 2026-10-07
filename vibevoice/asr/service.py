"""Local ASR discovery and a disposable worker in the shared runtime."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

from vibevoice.runtime.model_loading import _checkpoint_files, _model_config

ASR_WORKER = Path(__file__).resolve().with_name("worker.py")


def validate_asr_model(path):
    path = Path(path).resolve()
    config = _model_config(path)
    if config.get("model_type") != "vibevoice_asr":
        raise ValueError(f"{path} is not a native VibeVoice-ASR-HF checkpoint.")
    _checkpoint_files(path, config)
    for name in ("processor_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
        if not (path / name).is_file():
            raise ValueError(f"Missing ASR processor asset: {path / name}")
    return config


def discover_asr_models(settings):
    models = {}
    # Existing installations store ASR alongside TTS; also allow a dedicated folder.
    for root in (settings.models_dir / "asr", settings.tts_dir):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            try:
                validate_asr_model(path)
            except ValueError:
                continue
            models[path.name] = str(path.resolve())
    return models


def asr_python(check=True):
    python = Path(sys.executable).resolve()
    if check:
        completed = subprocess.run([str(python), str(ASR_WORKER), "--check"],
                                   capture_output=True, text=True)
        if completed.returncode:
            raise RuntimeError(
                "The shared ASR runtime is not ready. Reinstall this project's requirements in its venv. "
                f"{completed.stderr.strip()}"
            )
    return python


def run_transcription(audio_path, model_path, device, context=""):
    if not audio_path or not Path(audio_path).is_file():
        raise ValueError("Upload an audio file before transcribing.")
    validate_asr_model(model_path)
    python = asr_python(check=False)
    with tempfile.TemporaryDirectory(prefix="vibevoice-asr-") as directory:
        request = Path(directory) / "request.json"
        result = Path(directory) / "result.json"
        request.write_text(json.dumps({
            "audio_path": str(Path(audio_path).resolve()), "model_path": str(model_path),
            "device": device, "context": context or "",
        }), encoding="utf-8")
        # Logs go to the terminal. The result has its own JSON file, so logs can
        # never corrupt the protocol. Process exit releases the complete ASR model.
        completed = subprocess.run([str(python), str(ASR_WORKER),
                                    str(request), str(result)])
        if not result.is_file():
            raise RuntimeError(f"ASR worker exited without a result (code {completed.returncode}). See the terminal log.")
        payload = json.loads(result.read_text(encoding="utf-8"))
        if completed.returncode or "error" in payload:
            raise RuntimeError(payload.get("error", "ASR worker failed."))
        return payload
