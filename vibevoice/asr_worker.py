"""Native ASR entrypoint. Run as a file to avoid importing the TTS package."""

import json
import math
from pathlib import Path
import sys


def normalize_result(parsed, raw):
    if not isinstance(parsed, list) or not parsed:
        return {"transcript": raw.strip(), "segments": [],
                "warning": "The model did not emit parseable segments; showing its raw output."}
    segments = []
    for item in parsed:
        try:
            start, end = float(item["Start"]), float(item["End"])
            speaker = int(item["Speaker"])
            text = str(item["Content"]).strip()
            if not all(map(math.isfinite, (start, end))) or start < 0 or end < start or speaker < 0:
                raise ValueError
        except (KeyError, TypeError, ValueError, OverflowError):
            return {"transcript": raw.strip(), "segments": [],
                    "warning": "The model emitted invalid segments; showing its raw output."}
        segments.append({"start": start, "end": end, "speaker": speaker, "text": text})
    return {"transcript": "\n".join(f"[{s['speaker'] + 1}] {s['text']}" for s in segments),
            "segments": segments, "warning": ""}


def transcribe(request):
    import librosa
    import numpy as np
    import torch
    from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration

    device = request["device"]
    dtype = torch.float32 if device == "cpu" else torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(request["model_path"], local_files_only=True)
    options = {"local_files_only": True, "dtype": dtype, "attn_implementation": {
        "": "sdpa", "acoustic_tokenizer_encoder_config": "eager", "semantic_tokenizer_encoder_config": "eager",
    }}
    if device.startswith("cuda"):
        import psutil
        # Reserve space for audio embeddings and KV cache on small GPUs.
        free_bytes, _ = torch.cuda.mem_get_info(torch.device(device))
        gpu_index = torch.device(device).index or 0
        options.update(device_map="auto", max_memory={gpu_index: max(1, int(free_bytes * 0.75)),
                                                     "cpu": int(psutil.virtual_memory().available * 0.8)})
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(request["model_path"], **options).eval()
    if not device.startswith("cuda"):
        model.to(device)
    target_sr = processor.feature_extractor.sampling_rate
    audio, _ = librosa.load(request["audio_path"], sr=target_sr, mono=True)
    if not audio.size or not np.isfinite(audio).all():
        raise ValueError("The uploaded audio is empty or contains invalid samples.")
    inputs = processor.apply_transcription_request(audio=np.ascontiguousarray(audio), prompt=request["context"] or None)
    inputs = inputs.to(device, dtype)
    with torch.inference_mode():
        outputs = model.generate(**inputs, do_sample=False)
    generated = outputs[:, inputs["input_ids"].shape[1]:]
    raw = processor.decode(generated, skip_special_tokens=True)[0]
    parsed = processor.decode(generated, return_format="parsed")[0]
    return normalize_result(parsed, raw)


def main():
    if sys.argv[1:] == ["--check"]:
        from transformers import AutoProcessor, VibeVoiceAsrForConditionalGeneration
        import torch
        print(f"ASR runtime ready: {VibeVoiceAsrForConditionalGeneration.__name__}; Torch {torch.__version__}")
        return
    result = Path(sys.argv[2])
    try:
        payload = transcribe(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")))
    except Exception as exc:
        result.write_text(json.dumps({"error": str(exc)}, ensure_ascii=False), encoding="utf-8")
        raise
    result.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
