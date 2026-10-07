"""Shared configuration, discovery, and loading for VibeVoice model assets.

The module deliberately separates TTS checkpoints from supporting assets. In
local mode a missing TTS checkpoint is always an error; the support-download
switch only permits tokenizers and optional vocal-isolation weights.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REPOSITORIES = {
    "microsoft/VibeVoice-1.5B": {
        "folder": "VibeVoice-1.5B",
        "repo_id": "microsoft/VibeVoice-1.5B",
    },
    "vibevoice/VibeVoice-7B": {
        "folder": "VibeVoice-7B",
        "repo_id": "vibevoice/VibeVoice-7B",
    },
    "WestZhang/VibeVoice-Large-pt": {
        # Preserve the historical selection while avoiding its old fallback
        # path, which downloaded a second full 7B checkpoint unnecessarily.
        "folder": "VibeVoice-7B",
        "repo_id": "vibevoice/VibeVoice-7B",
    },
    "DevParker/VibeVoice7b-low-vram (4-bit)": {
        "folder": "VibeVoice-7B-4bit",
        "repo_id": "DevParker/VibeVoice7b-low-vram",
        "subfolder": "4bit",
        "config_repo_id": "vibevoice/VibeVoice-7B",
        "tokenizer_size": "7B",
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
        },
    },
    "FabioSarracino/VibeVoice-Large-Q8": {
        "folder": "VibeVoice-Large-Q8",
        "repo_id": "FabioSarracino/VibeVoice-Large-Q8",
        "tokenizer_size": "7B",
    },
}
MODEL_ALIASES = {
    "VibeVoice-1.5B": "microsoft/VibeVoice-1.5B",
    "VibeVoice-7B": "vibevoice/VibeVoice-7B",
    "VibeVoice-Large-Q8": "FabioSarracino/VibeVoice-Large-Q8",
    "VibeVoice-Large-pt": "WestZhang/VibeVoice-Large-pt",
}
TOKENIZER_REPOSITORIES = {
    "1.5B": ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    "7B": ("Qwen/Qwen2.5-7B", "Qwen2.5-7B"),
}
TOKENIZER_PATTERNS = [
    "tokenizer*",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "*.model",
    "*.tiktoken",
]


def _parse_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def load_project_env() -> None:
    """Load the checkout's .env before any model settings are resolved."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Keep command-line use possible in a minimal environment. The package
        # metadata installs python-dotenv for the supported setup.
        return
    load_dotenv(env_path, override=False)


load_project_env()


def add_model_cli_arguments(parser: Any) -> None:
    """Add the shared source/download options to an entrypoint parser."""
    parser.add_argument("--model-source", choices=("local", "huggingface"), default=None)
    parser.add_argument("--models-dir", type=str, default=None)
    support = parser.add_mutually_exclusive_group()
    support.add_argument("--allow-support-downloads", dest="allow_support_downloads", action="store_true")
    support.add_argument("--no-support-downloads", dest="allow_support_downloads", action="store_false")
    parser.set_defaults(allow_support_downloads=None)
    parser.add_argument("--hf-offline", action="store_true", default=None)
    parser.add_argument("--hf-cache-dir", type=str, default=None)


def launch_compatibly(interface: Any, **kwargs: Any) -> Any:
    """Call Gradio launch with only parameters supported by this release."""
    try:
        supported = inspect.signature(interface.launch).parameters
    except (TypeError, ValueError):
        filtered = dict(kwargs)
        filtered.pop("show_api", None)
        return interface.launch(**filtered)
    filtered = {key: value for key, value in kwargs.items() if key in supported}
    return interface.launch(**filtered)


@dataclass(frozen=True)
class ModelLoadingSettings:
    source: str
    models_dir: Path
    allow_support_downloads: bool
    hf_offline: bool
    hf_cache_dir: Optional[Path] = None

    @property
    def tts_dir(self) -> Path:
        return self.models_dir / "tts"

    @property
    def tokenizers_dir(self) -> Path:
        return self.models_dir / "tokenizers"


def _setting(cli_value: Any, env_name: str, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    return os.environ.get(env_name, default)


def settings_from_args(args: Any = None) -> ModelLoadingSettings:
    """Resolve model settings using CLI > environment > compatible defaults."""
    source = str(_setting(getattr(args, "model_source", None), "VIBEVOICE_MODEL_SOURCE", "huggingface")).strip().lower()
    if source not in {"local", "huggingface"}:
        raise ValueError("VIBEVOICE_MODEL_SOURCE must be 'local' or 'huggingface'")

    models_value = _setting(getattr(args, "models_dir", None), "VIBEVOICE_MODELS_DIR", "models")
    models_dir = Path(models_value).expanduser()
    if not models_dir.is_absolute():
        models_dir = PROJECT_ROOT / models_dir
    models_dir = models_dir.resolve()

    support_value = _setting(
        getattr(args, "allow_support_downloads", None),
        "VIBEVOICE_ALLOW_SUPPORT_DOWNLOADS",
        "true",
    )
    allow_support_downloads = _parse_bool(support_value, name="VIBEVOICE_ALLOW_SUPPORT_DOWNLOADS")

    # Offline environment flags are intentionally sticky. In particular,
    # setting support downloads true cannot override HF_HUB_OFFLINE=1.
    offline_env = any(
        _parse_bool(os.environ.get(name, "false"), name=name)
        for name in ("VIBEVOICE_HF_OFFLINE", "HF_HUB_OFFLINE")
    )
    hf_offline = bool(getattr(args, "hf_offline", False)) or offline_env

    cache_value = (
        getattr(args, "hf_cache_dir", None)
        or os.environ.get("HF_HOME")
        or os.environ.get("TRANSFORMERS_CACHE")
    )
    cache_dir = Path(cache_value).expanduser() if cache_value else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    return ModelLoadingSettings(
        source=source,
        models_dir=models_dir,
        allow_support_downloads=allow_support_downloads,
        hf_offline=hf_offline,
        hf_cache_dir=cache_dir,
    )


def default_model_name(settings: ModelLoadingSettings, legacy_default: Optional[str] = None) -> str:
    configured = os.environ.get("VIBEVOICE_MODEL")
    if configured:
        return configured.strip()
    if settings.source == "local":
        return "VibeVoice-1.5B"
    return legacy_default or "microsoft/VibeVoice-1.5B"


def normalize_model_selection(
    model_name: str,
    available_models: Mapping[str, Any],
    source: str,
) -> str:
    """Match a configured catalog alias to an actual UI choice when possible."""
    if model_name in available_models:
        return model_name

    canonical = MODEL_ALIASES.get(model_name, model_name)
    repository = DEFAULT_MODEL_REPOSITORIES.get(canonical)
    if repository is None:
        return model_name

    if source == "local":
        target_folder = repository["folder"].casefold()
        for label, value in available_models.items():
            label_name = Path(str(label)).name.casefold()
            value_name = Path(str(value)).name.casefold()
            if label_name == target_folder or value_name == target_folder:
                return label
        return model_name

    for label in available_models:
        if MODEL_ALIASES.get(label, label) == canonical:
            return label
    return model_name


def _model_config(path: Path) -> dict[str, Any]:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing required model config: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read model config {config_path}: {exc}") from exc


def _checkpoint_files(path: Path, config: dict[str, Any]) -> list[str]:
    index_names = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
    for index_name in index_names:
        index_path = path / index_name
        if index_path.is_file():
            try:
                with index_path.open("r", encoding="utf-8") as index_file:
                    weight_map = json.load(index_file).get("weight_map", {})
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Cannot read checkpoint index {index_path}: {exc}") from exc
            shard_names = sorted(set(weight_map.values()))
            if not shard_names:
                raise ValueError(f"Checkpoint index has no weight_map entries: {index_path}")
            missing = [name for name in shard_names if not (path / name).is_file()]
            if missing:
                raise ValueError(f"Checkpoint is incomplete at {path}; missing shard(s): {', '.join(missing)}")
            return shard_names

    direct_names = ("model.safetensors", "pytorch_model.bin", "pytorch_model.safetensors")
    existing = [name for name in direct_names if (path / name).is_file()]
    if existing:
        return existing
    raise ValueError(f"No supported model weights or shard index found in {path}")


def validate_tts_model(path: str | Path) -> dict[str, Any]:
    """Validate a complete VibeVoice TTS checkpoint and return its config."""
    model_dir = Path(path).expanduser().resolve()
    if not model_dir.is_dir():
        raise ValueError(f"Model directory does not exist: {model_dir}")
    config = _model_config(model_dir)
    lower_name = model_dir.name.lower()
    architectures = config.get("architectures") or []
    architecture_text = " ".join(str(item) for item in architectures).lower()
    if "asr" in lower_name or "asr" in architecture_text or config.get("model_type") != "vibevoice":
        raise ValueError(f"{model_dir} is not a supported VibeVoice TTS model")
    if architecture_text and not any(token in architecture_text for token in ("conditionalgeneration", "inference")):
        raise ValueError(f"{model_dir} does not declare a supported TTS architecture")
    _checkpoint_files(model_dir, config)
    return config


def discover_local_models(settings: ModelLoadingSettings) -> dict[str, str]:
    """Return valid local TTS checkpoints keyed by their visible folder name."""
    models: dict[str, str] = {}
    if not settings.tts_dir.is_dir():
        return models
    for candidate in sorted(settings.tts_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not candidate.is_dir() or "asr" in candidate.name.casefold():
            continue
        try:
            validate_tts_model(candidate)
        except ValueError:
            continue
        models[candidate.name] = str(candidate.resolve())
    return models


def _repository_for_model(model_name: str) -> dict[str, Any]:
    model_name = MODEL_ALIASES.get(model_name, model_name)
    if model_name in DEFAULT_MODEL_REPOSITORIES:
        return DEFAULT_MODEL_REPOSITORIES[model_name]
    # A full repository ID may be selected in the model path field. Avoid
    # turning a local folder name into an implicit Hub repository ID.
    model_path = Path(model_name).expanduser()
    anchored_path = model_path if model_path.is_absolute() else PROJECT_ROOT / model_path
    if "/" in model_name and not anchored_path.exists():
        final_name = model_name.rsplit("/", 1)[-1]
        return {"folder": final_name, "repo_id": model_name}
    raise ValueError(f"Unknown Hugging Face model selection: {model_name}")


def _snapshot_download(repo_id: str, destination: Path, settings: ModelLoadingSettings, **kwargs: Any) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required to download model assets; install project requirements") from exc
    destination.mkdir(parents=True, exist_ok=True)
    return Path(snapshot_download(
        repo_id=repo_id,
        local_dir=str(destination),
        cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
        local_files_only=settings.hf_offline,
        **kwargs,
    )).resolve()


def _base_config_directory(repo_id: str, settings: ModelLoadingSettings) -> Path:
    """Resolve a config-only dependency without fetching its large weights."""
    config_dir = settings.tts_dir / "VibeVoice-7B-config"
    config_file = config_dir / "config.json"
    if not config_file.is_file():
        if settings.hf_offline or not settings.allow_support_downloads:
            reason = "HF offline mode is enabled" if settings.hf_offline else "supporting downloads are disabled"
            raise RuntimeError(
                f"The selected quantized checkpoint needs the 7B base config, missing from {config_dir}; {reason}. "
                "Allow supporting downloads or place config.json there."
            )
        _snapshot_download(
            repo_id,
            config_dir,
            settings,
            allow_patterns=["config.json", "generation_config.json"],
        )
    return config_dir


def _resolved_from_repository(
    model_dir: Path,
    load_dir: Path,
    repository: dict[str, Any],
    settings: ModelLoadingSettings,
) -> "ResolvedModel":
    """Validate ordinary or config-less quantized repository weights."""
    config_path = load_dir / "config.json"
    if config_path.is_file():
        config = validate_tts_model(load_dir)
    elif repository.get("config_repo_id"):
        if not any((load_dir / name).is_file() for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")):
            raise ValueError(f"No checkpoint index found in quantized model folder: {load_dir}")
        config_dir = _base_config_directory(repository["config_repo_id"], settings)
        with (config_dir / "config.json").open("r", encoding="utf-8") as source:
            config = json.load(source)
        if config.get("model_type") != "vibevoice":
            raise ValueError(f"Base config is not a VibeVoice TTS config: {config.get('model_type')!r}")
        _checkpoint_files(load_dir, config)
    else:
        config = validate_tts_model(load_dir)
    return ResolvedModel(
        model_dir.resolve(), config, settings, repository["repo_id"],
        repository.get("subfolder"), repository.get("config_repo_id"), repository.get("tokenizer_size"),
        repository.get("quantization"),
    )


@dataclass(frozen=True)
class ResolvedModel:
    model_dir: Path
    model_config: dict[str, Any]
    settings: ModelLoadingSettings
    repo_id: Optional[str] = None
    subfolder: Optional[str] = None
    config_repo_id: Optional[str] = None
    tokenizer_size: Optional[str] = None
    quantization: Optional[dict[str, Any]] = None


def resolve_model(model_name: str, settings: ModelLoadingSettings) -> ResolvedModel:
    """Resolve a user selection to validated local files or an allowed download."""
    candidate = Path(model_name).expanduser()
    anchored_candidate = candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)
    if candidate.is_absolute() or anchored_candidate.exists():
        model_dir = anchored_candidate
        config = validate_tts_model(model_dir)
        return ResolvedModel(model_dir.resolve(), config, settings)

    if settings.source == "local":
        local_name = model_name.rsplit("/", 1)[-1] if model_name in DEFAULT_MODEL_REPOSITORIES else model_name
        if model_name in MODEL_ALIASES:
            local_name = MODEL_ALIASES[model_name].rsplit("/", 1)[-1]
        # Repository IDs often have names which differ from the imported
        # directory; use the canonical catalog destination when known.
        canonical_repo = MODEL_ALIASES.get(model_name, model_name)
        if canonical_repo in DEFAULT_MODEL_REPOSITORIES:
            local_name = DEFAULT_MODEL_REPOSITORIES[canonical_repo]["folder"]
        local_candidate = settings.tts_dir / local_name
        try:
            config = validate_tts_model(local_candidate)
        except ValueError as exc:
            raise ValueError(
                f"Local VibeVoice model '{model_name}' is unavailable or incomplete at {local_candidate}: {exc}. "
                "Place a complete TTS model folder under models/tts/<model-name>. "
                "Local mode never downloads replacement TTS weights."
            ) from exc
        return ResolvedModel(local_candidate.resolve(), config, settings)

    repository = _repository_for_model(model_name)
    destination = settings.tts_dir / repository["folder"]
    if destination.is_dir():
        try:
            load_dir = destination / repository["subfolder"] if repository.get("subfolder") else destination
            return _resolved_from_repository(destination, load_dir, repository, settings)
        except ValueError:
            if settings.hf_offline:
                raise
    download_options = {}
    if repository.get("subfolder"):
        download_options["allow_patterns"] = [f"{repository['subfolder']}/**"]
    # local_files_only is set by _snapshot_download when offline; this can
    # materialize an already cached repo without making network requests.
    model_dir = _snapshot_download(repository["repo_id"], destination, settings, **download_options)
    load_dir = model_dir / repository["subfolder"] if repository.get("subfolder") else model_dir
    return _resolved_from_repository(model_dir, load_dir, repository, settings)


def tokenizer_size_for_model(config: dict[str, Any], model_dir: Optional[Path] = None) -> str:
    if model_dir is not None:
        preprocessor_path = model_dir / "preprocessor_config.json"
        if preprocessor_path.is_file():
            try:
                with preprocessor_path.open("r", encoding="utf-8") as source:
                    language_model = json.load(source).get("language_model_pretrained_name")
                if language_model:
                    lowered = str(language_model).lower()
                    if "7b" in lowered:
                        return "7B"
                    if "1.5b" in lowered:
                        return "1.5B"
            except (OSError, json.JSONDecodeError):
                pass
    decoder = config.get("decoder_config") or {}
    hidden_size = decoder.get("hidden_size", config.get("hidden_size"))
    if hidden_size == 1536:
        return "1.5B"
    if hidden_size == 3584:
        return "7B"
    raise ValueError(f"Cannot determine VibeVoice tokenizer from decoder hidden_size={hidden_size!r}")


def _has_tokenizer_files(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in ("tokenizer.json", "vocab.json", "tokenizer.model"))


def resolve_tokenizer_path(resolved: ResolvedModel) -> Path:
    """Use bundled tokenizer files or fetch only the selected Qwen tokenizer."""
    model_dir = resolved.model_dir / resolved.subfolder if resolved.subfolder else resolved.model_dir
    if _has_tokenizer_files(model_dir):
        return model_dir.resolve()

    model_dir = resolved.model_dir / resolved.subfolder if resolved.subfolder else resolved.model_dir
    size = resolved.tokenizer_size or tokenizer_size_for_model(resolved.model_config, model_dir)
    repo_id, folder = TOKENIZER_REPOSITORIES[size]
    destination = resolved.settings.tokenizers_dir / folder
    if _has_tokenizer_files(destination):
        return destination.resolve()
    if resolved.settings.hf_offline or not resolved.settings.allow_support_downloads:
        reason = "HF offline mode is enabled" if resolved.settings.hf_offline else "supporting downloads are disabled"
        raise RuntimeError(
            f"Tokenizer {repo_id} is missing from {model_dir} and {destination}; {reason}. "
            "Add the tokenizer files locally or allow supporting downloads."
        )
    downloaded = _snapshot_download(
        repo_id,
        destination,
        resolved.settings,
        allow_patterns=TOKENIZER_PATTERNS,
    )
    if not _has_tokenizer_files(downloaded):
        raise RuntimeError(f"Tokenizer download from {repo_id} did not contain tokenizer files: {downloaded}")
    return downloaded


def _quantization_config(model_config: dict[str, Any], forced_config: Optional[dict[str, Any]] = None) -> Any:
    config = model_config.get("quantization_config") or forced_config
    if not config:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except Exception as exc:
        raise RuntimeError(
            "This model uses serialized bitsandbytes quantization settings, but the required "
            "Transformers/bitsandbytes support is unavailable. Install bitsandbytes in the project venv."
        ) from exc
    is_8bit = _parse_bool(config.get("load_in_8bit", config.get("_load_in_8bit", False)), name="load_in_8bit")
    is_4bit = _parse_bool(config.get("load_in_4bit", config.get("_load_in_4bit", False)), name="load_in_4bit")
    if not is_8bit and not is_4bit:
        raise RuntimeError(f"Unsupported serialized quantization configuration: {config}")
    try:
        import torch
        normalized = dict(config)
        dtype_value = normalized.get("bnb_4bit_compute_dtype")
        if isinstance(dtype_value, str):
            normalized["bnb_4bit_compute_dtype"] = getattr(torch, dtype_value.rsplit(".", 1)[-1], None)
            if normalized["bnb_4bit_compute_dtype"] is None:
                raise ValueError(f"Unsupported bnb_4bit_compute_dtype: {dtype_value}")
        # Older checkpoints serialize private transformers flags as well as
        # public options; from_dict handles both across the pinned 4.51 release.
        return BitsAndBytesConfig.from_dict(normalized)
    except Exception as exc:
        raise RuntimeError(f"Could not apply the model's bitsandbytes quantization config: {exc}") from exc


def load_model_and_processor(
    model_name: str,
    settings: ModelLoadingSettings,
    *,
    device: str,
    attn_implementation: str,
    torch_dtype: Any = None,
) -> tuple[Any, Any, ResolvedModel]:
    """Load shared processor and model objects for every app entrypoint."""
    import torch
    from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
    from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    resolved = resolve_model(model_name, settings)
    tokenizer_path = resolve_tokenizer_path(resolved)
    source_dir = resolved.model_dir / resolved.subfolder if resolved.subfolder else resolved.model_dir
    config_path = source_dir
    if resolved.config_repo_id:
        config_path = _base_config_directory(resolved.config_repo_id, settings)

    processor = VibeVoiceProcessor.from_pretrained(
        str(source_dir),
        tokenizer_path=str(tokenizer_path),
        local_files_only=settings.hf_offline,
        cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
    )
    model_config = VibeVoiceConfig.from_pretrained(
        str(config_path),
        local_files_only=settings.hf_offline,
        cache_dir=str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
    )
    kwargs: dict[str, Any] = {
        "config": model_config,
        "device_map": device,
        "attn_implementation": attn_implementation,
        "local_files_only": settings.hf_offline,
        "cache_dir": str(settings.hf_cache_dir) if settings.hf_cache_dir else None,
    }
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype
    else:
        kwargs["torch_dtype"] = torch.bfloat16
    quantization_config = _quantization_config(resolved.model_config, resolved.quantization)
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if resolved.subfolder:
        kwargs["subfolder"] = resolved.subfolder
    quantization_options = resolved.model_config.get("quantization_config") or resolved.quantization or {}
    is_4bit = _parse_bool(
        quantization_options.get("load_in_4bit", quantization_options.get("_load_in_4bit", False)),
        name="load_in_4bit",
    )
    if quantization_config is not None and is_4bit:
        kwargs["torch_dtype"] = torch.float16
    model_path = str(resolved.model_dir)
    try:
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, **kwargs)
        except Exception as first_error:
            if attn_implementation == "sdpa":
                raise
            print(f"⚠️ {attn_implementation} failed, retrying with SDPA: {first_error}")
            kwargs["attn_implementation"] = "sdpa"
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(model_path, **kwargs)
    except ImportError as exc:
        if quantization_config is not None:
            raise RuntimeError(
                "The selected checkpoint requires bitsandbytes quantization, but the Windows bitsandbytes "
                "runtime could not load. Install bitsandbytes in the project venv and verify its CUDA runtime."
            ) from exc
        raise
    model.eval()
    return processor, model, resolved


def download_support_asset(repo_id: str, destination: str | Path, settings: ModelLoadingSettings, **kwargs: Any) -> Path:
    """Fetch an optional support asset only when policy permits it."""
    if settings.hf_offline:
        raise RuntimeError("HF offline mode prohibits all downloads")
    if not settings.allow_support_downloads:
        raise RuntimeError("Supporting downloads are disabled by VIBEVOICE_ALLOW_SUPPORT_DOWNLOADS")
    return _snapshot_download(repo_id, Path(destination), settings, **kwargs)
