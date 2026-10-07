"""Realtime model loading, cached voice prompts, and cancellable audio streaming."""

from __future__ import annotations

import copy
import json
import queue
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import set_seed
from transformers.cache_utils import DynamicCache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from vibevoice.realtime.voices import resolve_voice_preset
from vibevoice.runtime.model_loading import (
    TOKENIZER_PATTERNS,
    ModelLoadingSettings,
    _checkpoint_files,
    download_support_asset,
)
from vibevoice.runtime.script import resolve_seed

SAMPLE_RATE = 24000
REALTIME_MODEL_ID = "microsoft/VibeVoice-Realtime-0.5B"
REALTIME_MODEL_FOLDER = "VibeVoice-Realtime-0.5B"
PROMPT_BRANCHES = ("lm", "tts_lm", "neg_lm", "neg_tts_lm")
_PROMPT_LOAD_LOCK = threading.RLock()


def validate_realtime_model(path):
    path = Path(path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "vibevoice_streaming":
        raise ValueError(f"Not a realtime VibeVoice checkpoint: {path}")
    _checkpoint_files(path, config)
    if not (path / "preprocessor_config.json").is_file():
        raise ValueError(f"Missing realtime preprocessor_config.json: {path}")
    return config


def discover_realtime_models(settings):
    models = {REALTIME_MODEL_ID: REALTIME_MODEL_ID}
    if settings.tts_dir.is_dir():
        for path in sorted(settings.tts_dir.iterdir()):
            if not path.is_dir():
                continue
            try:
                validate_realtime_model(path)
            except (ValueError, OSError):
                continue
            label = (
                REALTIME_MODEL_ID if path.name == REALTIME_MODEL_FOLDER else path.name
            )
            models[label] = str(path.resolve())
    return models


def resolve_realtime_model(selection, settings):
    if str(selection) in (REALTIME_MODEL_ID, REALTIME_MODEL_FOLDER):
        path = settings.tts_dir / REALTIME_MODEL_FOLDER
        try:
            validate_realtime_model(path)
        except (ValueError, OSError):
            path = download_support_asset(
                REALTIME_MODEL_ID,
                path,
                settings,
                allow_patterns=["*.json", "*.safetensors", "*.bin"],
            )
    else:
        path = Path(selection).expanduser().resolve()
    validate_realtime_model(path)
    return path


@contextmanager
def _restricted_prompt_mappings():
    # PyTorch's restricted loader rejects HF's approved OrderedDict subclass.
    import torch._weights_only_unpickler as restricted

    with _PROMPT_LOAD_LOCK:
        original = getattr(restricted.Unpickler, "_check_set_item_target", None)
        if original is not None:

            def check_target(self, opcode):
                if type(self.stack[-1]) is BaseModelOutputWithPast:
                    return
                original(self, opcode)

            restricted.Unpickler._check_set_item_target = check_target
        try:
            with torch.serialization.safe_globals(
                [BaseModelOutputWithPast, DynamicCache, DynamicLayer]
            ):
                yield
        finally:
            if original is not None:
                restricted.Unpickler._check_set_item_target = original


def _cache_tensors(cache):
    if isinstance(cache, DynamicCache) and hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    if isinstance(cache, DynamicCache) and hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    raise ValueError("Realtime prompt has an unsupported KV cache")


def load_voice_prompt(path, config, device, dtype):
    with _restricted_prompt_mappings():
        preset = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(preset, dict):
        raise TypeError("Realtime prompt must contain four cached branches")
    result = {}
    for name in PROMPT_BRANCHES:
        branch = preset.get(name)
        if branch is None:
            raise ValueError(f"Realtime prompt is missing cached branch {name}")
        if not isinstance(branch, (dict, BaseModelOutputWithPast)):
            raise TypeError(f"Invalid realtime prompt branch: {name}")
        hidden = branch.get("last_hidden_state")
        if (
            not torch.is_tensor(hidden)
            or hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
        ):
            raise ValueError(f"Invalid realtime prompt hidden state: {name}")
        if hidden.shape[-1] != config.decoder_config.hidden_size:
            raise ValueError(
                f"Realtime prompt hidden size does not match the model: {name}"
            )
        backbone = config.backbone_config(tts="tts_lm" in name)
        pairs = _cache_tensors(branch.get("past_key_values"))
        if len(pairs) != backbone.num_hidden_layers:
            raise ValueError(
                f"Realtime prompt layer count does not match the model: {name}"
            )
        cache = DynamicCache(config=backbone)
        head_dim = (
            getattr(backbone, "head_dim", None)
            or backbone.hidden_size // backbone.num_attention_heads
        )
        shape = (1, backbone.num_key_value_heads, hidden.shape[1], head_dim)
        for index, (keys, values) in enumerate(pairs):
            if (
                not torch.is_tensor(keys)
                or not torch.is_tensor(values)
                or tuple(keys.shape) != shape
                or tuple(values.shape) != shape
            ):
                raise ValueError(
                    f"Realtime prompt KV shape does not match the model: {name}"
                )
            cache.update(
                keys.to(device=device, dtype=dtype),
                values.to(device=device, dtype=dtype),
                index,
            )
        result[name] = BaseModelOutputWithPast(
            last_hidden_state=hidden.to(device=device, dtype=dtype),
            past_key_values=cache,
        )
    return result


def resolve_realtime_tokenizer(path, settings):
    preprocessor = json.loads(
        (path / "preprocessor_config.json").read_text(encoding="utf-8")
    )
    tokenizer_repo = preprocessor.get(
        "language_model_pretrained_name", "Qwen/Qwen2.5-0.5B"
    )
    tokenizer_name = tokenizer_repo.rsplit("/", 1)[-1]
    tokenizer_path = (
        path
        if (path / "tokenizer.json").is_file()
        else settings.tokenizers_dir / tokenizer_name
    )
    if not (tokenizer_path / "tokenizer.json").is_file():
        tokenizer_path = download_support_asset(
            tokenizer_repo,
            tokenizer_path,
            settings,
            allow_patterns=TOKENIZER_PATTERNS,
        )
        if not (tokenizer_path / "tokenizer.json").is_file():
            raise ValueError(
                f"Realtime tokenizer download is incomplete: {tokenizer_path}"
            )
    return tokenizer_path


def load_realtime_model(path, settings, device):
    from vibevoice.modular.configuration_vibevoice_streaming import (
        VibeVoiceStreamingConfig,
    )
    from vibevoice.modular.modeling_vibevoice_streaming_inference import (
        VibeVoiceStreamingForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_streaming_processor import (
        VibeVoiceStreamingProcessor,
    )

    path = resolve_realtime_model(path, settings)
    tokenizer_path = resolve_realtime_tokenizer(path, settings)
    processor = VibeVoiceStreamingProcessor.from_pretrained(
        path,
        tokenizer_path=tokenizer_path,
        local_files_only=True,
    )
    config = VibeVoiceStreamingConfig.from_pretrained(path, local_files_only=True)
    dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    model, loading_info = (
        VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
            path,
            config=config,
            local_files_only=True,
            dtype=dtype,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
    )
    if loading_info["missing_keys"] or loading_info["unexpected_keys"]:
        raise ValueError(
            "Realtime checkpoint weights do not match the streaming architecture"
        )
    model = model.to(device).eval()
    if (
        not torch.isfinite(model.speech_scaling_factor).all()
        or not torch.isfinite(model.speech_bias_factor).all()
        or (model.speech_scaling_factor == 0).any()
    ):
        raise ValueError("Realtime checkpoint has invalid speech scaling factors")
    model.model.noise_scheduler = model.model.noise_scheduler.from_config(
        model.model.noise_scheduler.config,
        algorithm_type="sde-dpmsolver++",
        beta_schedule="squaredcos_cap_v2",
    )
    return processor, model


def generation_budget(text_tokens, prompt_length, context_length):
    # Each window charges five text tokens and six speech latents.
    windows = (text_tokens + 4) // 5
    requested = max(12, (3 * windows + 1) * 11)
    available = context_length - prompt_length
    if available < requested:
        raise ValueError(
            "Realtime text exceeds the remaining model context; shorten the text"
        )
    return requested


class RealtimeStreamer:
    """Bounded single-request audio queue with cancellable producer waits."""

    def __init__(self, stop_event, max_chunks=32):
        self.stop_event = stop_event
        self.audio_queue = queue.Queue(maxsize=max_chunks)
        self.finished = threading.Event()

    def put(self, audio_chunks, sample_indices):
        for chunk, index in zip(audio_chunks, sample_indices):
            if int(index) != 0:
                raise ValueError("Realtime streamer supports one sample")
            chunk = chunk.detach().cpu().float().numpy().reshape(-1)
            if chunk.size == 0:
                continue
            if not np.isfinite(chunk).all():
                raise ValueError("Realtime model produced nonfinite audio samples")
            while not self.stop_event.is_set() and not self.finished.is_set():
                try:
                    self.audio_queue.put(chunk, timeout=0.1)
                    break
                except queue.Full:
                    continue

    def end(self, sample_indices=None):
        self.finished.set()

    def get_stream(self, sample_index):
        if sample_index != 0:
            raise ValueError("Realtime streamer supports one sample")
        while not self.stop_event.is_set():
            try:
                yield self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                if self.finished.is_set():
                    return


class RealtimeService:
    def __init__(self, settings: ModelLoadingSettings, device):
        self.settings, self.device = settings, device
        self.model = self.processor = self.prompt = None
        self.selection = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

    def load(self, model_path, voice_path):
        selection = (str(model_path), str(voice_path))
        if self.selection == selection and self.model is not None:
            return
        self.unload()
        processor, model = load_realtime_model(model_path, self.settings, self.device)
        voice_path = resolve_voice_preset(voice_path, self.settings)
        prompt = load_voice_prompt(voice_path, model.config, self.device, model.dtype)
        self.processor, self.model, self.prompt = processor, model, prompt
        self.selection = selection

    def stop(self):
        self.stop_event.set()

    def unload(self):
        if self.lock.locked():
            raise RuntimeError("Stop realtime generation before unloading the model")
        self.model = self.processor = self.prompt = None
        self.selection = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def stream(
        self, text, *, seed=42, cfg_scale=1.5, diffusion_steps=5, cancel_event=None
    ):
        if self.model is None or self.prompt is None:
            raise ValueError("Load a realtime model and voice preset first")
        if not text.strip():
            raise ValueError("Provide text for realtime speech")
        if not 1.5 <= cfg_scale <= 3.0 or not 1 <= int(diffusion_steps) <= 50:
            raise ValueError("Invalid realtime guidance or diffusion steps")
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Realtime generation is already active")
        self.stop_event = (
            cancel_event if cancel_event is not None else threading.Event()
        )
        thread = None
        try:
            inputs = self.processor.process_input_with_cached_prompt(
                text, cached_prompt=self.prompt
            )
            budget = generation_budget(
                inputs["tts_text_ids"].shape[1],
                inputs["tts_lm_input_ids"].shape[1],
                self.model.config.decoder_config.max_position_embeddings,
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            streamer = RealtimeStreamer(self.stop_event)
            errors, results = [], []

            def produce():
                try:
                    set_seed(resolve_seed(seed))
                    self.model.set_ddpm_inference_steps(int(diffusion_steps))
                    results.append(
                        self.model.generate(
                            **inputs,
                            tokenizer=self.processor.tokenizer,
                            all_prefilled_outputs=copy.deepcopy(self.prompt),
                            max_new_tokens=budget,
                            cfg_scale=cfg_scale,
                            audio_streamer=streamer,
                            stop_check_fn=self.stop_event.is_set,
                            return_speech=False,
                            show_progress_bar=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - propagate producer errors to the consumer
                    errors.append(exc)
                finally:
                    streamer.end()

            thread = threading.Thread(target=produce, daemon=True)
            thread.start()
            pending, size = [], 0
            for chunk in streamer.get_stream(0):
                pending.append(chunk)
                size += chunk.size
                if size >= SAMPLE_RATE // 2:
                    yield np.concatenate(pending)
                    pending, size = [], 0
            if pending and not self.stop_event.is_set():
                yield np.concatenate(pending)
            thread.join()
            if errors:
                raise errors[0]
            if (
                results
                and results[0].reach_max_step_sample.any()
                and not self.stop_event.is_set()
            ):
                raise RuntimeError(
                    "Realtime generation reached its length limit before finishing"
                )
        finally:
            if thread is not None:
                if thread.is_alive():
                    self.stop_event.set()
                thread.join()
            self.lock.release()
