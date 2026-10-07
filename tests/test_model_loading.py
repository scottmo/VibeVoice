import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import vibevoice.model_loading as loading
from vibevoice.model_loading import ModelLoadingSettings


def write_model(path: Path, hidden_size: int = 1536, *, complete: bool = True,
               quantization: dict | None = None, preprocessor: dict | None = None) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "vibevoice",
        "architectures": ["VibeVoiceForConditionalGeneration"],
        "decoder_config": {"model_type": "qwen2", "hidden_size": hidden_size},
    }
    if quantization:
        config["quantization_config"] = quantization
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    shard = "model-00001-of-00001.safetensors"
    index = {"weight_map": {"model.embed_tokens.weight": shard}}
    (path / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    if complete:
        (path / shard).write_bytes(b"fixture shard")
    if preprocessor:
        (path / "preprocessor_config.json").write_text(json.dumps(preprocessor), encoding="utf-8")
    return path


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def settings(self, source="local", *, support=True, offline=False, models_dir=None):
        return ModelLoadingSettings(
            source=source,
            models_dir=Path(models_dir or self.models),
            allow_support_downloads=support,
            hf_offline=offline,
        )

    def test_cli_settings_override_environment_and_windows_paths_anchor(self):
        env = {
            "VIBEVOICE_MODEL_SOURCE": "local",
            "VIBEVOICE_MODELS_DIR": "models",
            "VIBEVOICE_ALLOW_SUPPORT_DOWNLOADS": "true",
            "HF_HUB_OFFLINE": "0",
        }
        args = SimpleNamespace(
            model_source="huggingface",
            models_dir="F:\\VibeVoice Models",
            allow_support_downloads=False,
            hf_offline=True,
            hf_cache_dir=None,
        )
        with patch.dict(os.environ, env, clear=True):
            resolved = loading.settings_from_args(args)
        self.assertEqual(resolved.source, "huggingface")
        self.assertEqual(str(resolved.models_dir), "F:\\VibeVoice Models")
        self.assertFalse(resolved.allow_support_downloads)
        self.assertTrue(resolved.hf_offline)

    def test_huggingface_remains_default_without_source_setting(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(loading.settings_from_args().source, "huggingface")

    def test_relative_model_paths_resolve_from_repository_root(self):
        project = self.root / "checkout"
        model = write_model(project / "assets" / "VibeVoice-1.5B")
        settings = self.settings(models_dir=project / "models")
        with patch.object(loading, "PROJECT_ROOT", project):
            resolved = loading.resolve_model("assets/VibeVoice-1.5B", settings)
        self.assertEqual(resolved.model_dir, model.resolve())

    def test_discovery_finds_only_complete_tts_models_and_excludes_asr(self):
        tts = self.models / "tts"
        write_model(tts / "VibeVoice-1.5B", 1536)
        write_model(tts / "VibeVoice-7B", 3584)
        write_model(tts / "VibeVoice-Large-Q8", 3584, quantization={"load_in_8bit": True})
        asr = write_model(tts / "VibeVoice-ASR-HF", 1536)
        config = json.loads((asr / "config.json").read_text(encoding="utf-8"))
        config["model_type"] = "qwen2"
        (asr / "config.json").write_text(json.dumps(config), encoding="utf-8")
        wrong_model_type = write_model(tts / "ImportedCheckpoint", 1536)
        config = json.loads((wrong_model_type / "config.json").read_text(encoding="utf-8"))
        config["model_type"] = "vibevoice_asr"
        config["architectures"] = []
        (wrong_model_type / "config.json").write_text(json.dumps(config), encoding="utf-8")
        write_model(tts / "Incomplete", 1536, complete=False)

        discovered = loading.discover_local_models(self.settings())
        self.assertEqual(set(discovered), {"VibeVoice-1.5B", "VibeVoice-7B", "VibeVoice-Large-Q8"})

    def test_missing_shard_is_actionable_and_local_mode_never_downloads_replacement(self):
        model = write_model(self.models / "tts" / "VibeVoice-1.5B", complete=False)
        with self.assertRaisesRegex(ValueError, "missing shard"):
            loading.validate_tts_model(model)
        with patch.object(loading, "_snapshot_download") as download:
            with self.assertRaisesRegex(ValueError, "model-00001-of-00001.safetensors"):
                loading.resolve_model("VibeVoice-1.5B", self.settings())
            with self.assertRaisesRegex(ValueError, "never downloads replacement TTS weights"):
                loading.resolve_model("VibeVoice-7B", self.settings())
        download.assert_not_called()

    def test_catalog_aliases_resolve_in_local_mode(self):
        one_point_five = write_model(self.models / "tts" / "VibeVoice-1.5B")
        seven_b = write_model(self.models / "tts" / "VibeVoice-7B", 3584)
        q8 = write_model(
            self.models / "tts" / "VibeVoice-Large-Q8",
            3584,
            quantization={"load_in_8bit": True},
        )
        selections = {
            "VibeVoice-1.5B": one_point_five,
            "microsoft/VibeVoice-1.5B": one_point_five,
            "VibeVoice-7B": seven_b,
            "vibevoice/VibeVoice-7B": seven_b,
            "WestZhang/VibeVoice-Large-pt": seven_b,
            "VibeVoice-Large-Q8": q8,
            "FabioSarracino/VibeVoice-Large-Q8": q8,
        }
        for selection, expected in selections.items():
            with self.subTest(selection=selection):
                self.assertEqual(loading.resolve_model(selection, self.settings()).model_dir, expected.resolve())

    def test_model_selection_alias_matches_existing_dropdown_entry(self):
        choices = {
            "VibeVoice-1.5B": "VibeVoice-1.5B",
            "VibeVoice-7B": "VibeVoice-7B",
            "VibeVoice-Large-Q8": "VibeVoice-Large-Q8",
        }
        self.assertEqual(
            loading.normalize_model_selection("microsoft/VibeVoice-1.5B", choices, "local"),
            "VibeVoice-1.5B",
        )
        self.assertEqual(
            loading.normalize_model_selection("WestZhang/VibeVoice-Large-pt", choices, "local"),
            "VibeVoice-7B",
        )
        self.assertEqual(
            loading.normalize_model_selection("FabioSarracino/VibeVoice-Large-Q8", choices, "local"),
            "VibeVoice-Large-Q8",
        )
        online_choices = {"microsoft/VibeVoice-1.5B": "microsoft/VibeVoice-1.5B"}
        self.assertEqual(
            loading.normalize_model_selection("VibeVoice-1.5B", online_choices, "huggingface"),
            "microsoft/VibeVoice-1.5B",
        )

    def test_q8_serialized_skip_modules_and_default_bfloat16_are_preserved(self):
        from transformers import BitsAndBytesConfig
        from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
        from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        import torch

        model_dir = write_model(
            self.models / "tts" / "VibeVoice-Large-Q8",
            3584,
            quantization={
                "load_in_8bit": True,
                "llm_int8_skip_modules": ["audio_decoder"],
            },
        )
        (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
        settings = self.settings()
        received = {}
        fake_model = MagicMock()

        def fake_load(_path, **kwargs):
            received.update(kwargs)
            return fake_model

        config = VibeVoiceConfig.from_dict(json.loads((model_dir / "config.json").read_text()))
        with (
            patch.object(VibeVoiceProcessor, "from_pretrained", return_value=object()),
            patch.object(VibeVoiceConfig, "from_pretrained", return_value=config),
            patch.object(VibeVoiceForConditionalGenerationInference, "from_pretrained", side_effect=fake_load),
        ):
            _processor, model, _resolved = loading.load_model_and_processor(
                "VibeVoice-Large-Q8",
                settings,
                device="cuda",
                attn_implementation="sdpa",
            )

        self.assertIs(model, fake_model)
        self.assertEqual(received["dtype"], torch.bfloat16)
        self.assertNotIn("quantization_config", received)
        self.assertEqual(received["config"].quantization_config["llm_int8_skip_modules"], ["audio_decoder"])
        self.assertTrue(BitsAndBytesConfig.from_dict(received["config"].quantization_config).load_in_8bit)

    def test_huggingface_download_uses_canonical_project_destination_and_reuses_assets(self):
        settings = self.settings("huggingface")
        calls = []

        def download_model(repo_id, destination, _settings, **kwargs):
            calls.append((repo_id, destination, kwargs))
            write_model(destination, 1536)
            return destination

        with patch.object(loading, "_snapshot_download", side_effect=download_model):
            resolved = loading.resolve_model("VibeVoice-1.5B", settings)
        self.assertEqual(resolved.repo_id, "microsoft/VibeVoice-1.5B")
        self.assertEqual(resolved.model_dir, (self.models / "tts" / "VibeVoice-1.5B").resolve())
        self.assertEqual(calls[0][0], "microsoft/VibeVoice-1.5B")
        self.assertEqual(calls[0][1], self.models / "tts" / "VibeVoice-1.5B")
        with patch.object(loading, "_snapshot_download", side_effect=AssertionError("unexpected download")):
            reused = loading.resolve_model("microsoft/VibeVoice-1.5B", settings)
        self.assertEqual(reused.model_dir, resolved.model_dir)

    def test_tokenizer_uses_preprocessor_declaration_then_hidden_size_fallback(self):
        one_point_five = write_model(
            self.models / "tts" / "VibeVoice-1.5B",
            1536,
            preprocessor={"language_model_pretrained_name": "Qwen/Qwen2.5-7B"},
        )
        config = json.loads((one_point_five / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(loading.tokenizer_size_for_model(config, one_point_five), "7B")
        seven_b = write_model(self.models / "tts" / "VibeVoice-7B", 3584)
        seven_config = json.loads((seven_b / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(loading.tokenizer_size_for_model(seven_config, seven_b), "7B")

    def test_explicit_arbitrary_tokenizer_path_works_without_preprocessor_config(self):
        from vibevoice.modular.modular_vibevoice_text_tokenizer import VibeVoiceTextTokenizerFast
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

        model_dir = self.root / "checkpoint-without-processor-config"
        model_dir.mkdir()
        tokenizer = object()
        with patch.object(VibeVoiceTextTokenizerFast, "from_pretrained", return_value=tokenizer) as load_tokenizer:
            processor = VibeVoiceProcessor.from_pretrained(
                str(model_dir),
                tokenizer_path="C:/assets/text_vocab",
                language_model_pretrained_name="organization/custom-model",
            )

        self.assertIs(processor.tokenizer, tokenizer)
        load_tokenizer.assert_called_once_with("C:/assets/text_vocab")

    def test_tokenizer_download_is_scoped_to_tokenizer_folder_and_reused(self):
        settings = self.settings(support=True)
        model_path = write_model(self.models / "tts" / "VibeVoice-7B", 3584)
        resolved_model = loading.resolve_model("VibeVoice-7B", settings)
        calls = []

        def fake_download(repo_id, destination, _settings, **kwargs):
            calls.append((repo_id, destination, kwargs))
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "tokenizer.json").write_text("{}", encoding="utf-8")
            return destination

        with patch.object(loading, "_snapshot_download", side_effect=fake_download):
            tokenizer_path = loading.resolve_tokenizer_path(resolved_model)
        self.assertEqual(tokenizer_path, (self.models / "tokenizers" / "Qwen2.5-7B").resolve())
        self.assertEqual(calls[0][0], "Qwen/Qwen2.5-7B")
        self.assertEqual(calls[0][1], self.models / "tokenizers" / "Qwen2.5-7B")
        self.assertIn("allow_patterns", calls[0][2])
        with patch.object(loading, "_snapshot_download", side_effect=AssertionError("cached tokenizer should be reused")):
            cached_path = loading.resolve_tokenizer_path(resolved_model)
        self.assertEqual(cached_path, tokenizer_path)
        self.assertFalse((model_path / "tokenizer.json").exists())

    def test_offline_environment_blocks_vocal_and_support_asset_downloads(self):
        from vibevoice.utils.vocal_isolation import get_model_path

        settings = self.settings(support=True, offline=True)
        with self.assertRaisesRegex(RuntimeError, "offline mode prohibits downloads"):
            get_model_path(settings)
        with patch.object(loading, "_snapshot_download") as download:
            with self.assertRaisesRegex(RuntimeError, "offline mode prohibits"):
                loading.download_support_asset("org/model", self.models / "support", settings)
        download.assert_not_called()

    def test_vocal_isolation_honors_cli_models_dir_and_offline_override(self):
        from vibevoice.utils.vocal_isolation import get_model_path

        custom = self.root / "custom-model-root"
        args = SimpleNamespace(
            model_source=None,
            models_dir=str(custom),
            allow_support_downloads=True,
            hf_offline=True,
            hf_cache_dir=None,
        )
        with patch.dict(os.environ, {"VIBEVOICE_ALLOW_SUPPORT_DOWNLOADS": "true", "HF_HUB_OFFLINE": "0"}, clear=True):
            settings = loading.settings_from_args(args)
        with self.assertRaisesRegex(RuntimeError, str(custom).replace("\\", "\\\\")):
            get_model_path(settings)

    def test_cached_config_less_q4_weights_do_not_redownload_model_repo(self):
        settings = self.settings("huggingface", support=False, offline=True)
        repo = loading.DEFAULT_MODEL_REPOSITORIES["DevParker/VibeVoice7b-low-vram (4-bit)"]
        model_root = settings.tts_dir / repo["folder"]
        q4_weights = model_root / repo["subfolder"]
        write_model(settings.tts_dir / "VibeVoice-7B-config", 3584)
        q4_weights.mkdir(parents=True)
        shard = "model-00001-of-00001.safetensors"
        (q4_weights / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": shard}}), encoding="utf-8"
        )
        (q4_weights / shard).write_bytes(b"fixture q4 shard")

        with patch.object(loading, "_snapshot_download", side_effect=AssertionError("must reuse local Q4 weights")):
            resolved = loading.resolve_model("DevParker/VibeVoice7b-low-vram (4-bit)", settings)
        self.assertTrue(resolved.quantization["load_in_4bit"])
        self.assertEqual(resolved.model_dir, model_root.resolve())


if __name__ == "__main__":
    unittest.main()
