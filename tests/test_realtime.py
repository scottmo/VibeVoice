import copy
import json
import multiprocessing
import os
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from transformers import Qwen2Config
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
)
from vibevoice.modular.configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.realtime.service import (
    REALTIME_MODEL_ID,
    RealtimeService,
    RealtimeStreamer,
    discover_realtime_models,
    generation_budget,
    load_realtime_model,
    load_voice_prompt,
    resolve_realtime_model,
    resolve_realtime_tokenizer,
    validate_realtime_model,
)
from vibevoice.realtime.voices import (
    OFFICIAL_VOICES,
    discover_realtime_voices,
    resolve_voice_preset,
)
from vibevoice.realtime.worker import RealtimeWorkerService, _realtime_worker
from vibevoice.runtime.model_loading import ModelLoadingSettings


def synthetic_worker(settings, device, selection, text, parameters, output, cancel):
    if text == "crash":
        os._exit(3)
    output.put(("chunk", np.full(24000, 0.1, dtype=np.float32)))
    if text == "fail":
        output.put(("error", "worker failed"))
    elif text == "wait":
        cancel.wait(30)
    else:
        output.put(("chunk", np.full(24000, 0.2, dtype=np.float32)))
        output.put(("done", None))


def tiny_config():
    return VibeVoiceStreamingConfig(
        acoustic_tokenizer_config=VibeVoiceAcousticTokenizerConfig(
            channels=1,
            vae_dim=2,
            fix_std=0.0,
            std_dist_type="none",
            encoder_n_filters=4,
            encoder_ratios=[2],
            encoder_depths="1-1",
            decoder_n_filters=4,
            decoder_ratios=[2],
            decoder_depths="1-1",
        ),
        decoder_config=Qwen2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=16,
            max_position_embeddings=256,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            tie_word_embeddings=False,
        ),
        diffusion_head_config=VibeVoiceDiffusionHeadConfig(
            hidden_size=16,
            head_layers=1,
            head_ffn_ratio=2,
            latent_size=2,
            speech_vae_dim=2,
            ddpm_num_steps=10,
            ddpm_num_inference_steps=1,
            ddpm_batch_mul=1,
        ),
        tts_backbone_num_hidden_layers=1,
        dtype="float32",
    )


def prompt(config, legacy=False, negative_length=3):
    result = {}
    for branch in ("lm", "tts_lm", "neg_lm", "neg_tts_lm"):
        cache = DynamicCache(config=config.backbone_config(tts="tts_lm" in branch))
        length = negative_length if branch.startswith("neg_") else 3
        keys = torch.randn(1, 2, length, 8)
        if legacy:
            cache.__dict__.clear()
            cache.key_cache = [keys]
            cache.value_cache = [keys.clone()]
        else:
            cache.update(keys, keys.clone(), 0)
        result[branch] = BaseModelOutputWithPast(
            last_hidden_state=torch.zeros(1, length, 16),
            past_key_values=cache,
        )
    return result


class AssetTests(unittest.TestCase):
    def test_family_discovery_and_missing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = ModelLoadingSettings(Path(directory))
            model = settings.tts_dir / "VibeVoice-Realtime-0.5B"
            model.mkdir(parents=True)
            (model / "config.json").write_text(json.dumps(tiny_config().to_dict()))
            (model / "model.safetensors").touch()
            (model / "preprocessor_config.json").write_text("{}")
            self.assertEqual(
                discover_realtime_models(settings),
                {REALTIME_MODEL_ID: str(model.resolve())},
            )
            (model / "model.safetensors").unlink()
            with self.assertRaises(ValueError):
                validate_realtime_model(model)
            self.assertEqual(
                discover_realtime_models(settings),
                {REALTIME_MODEL_ID: REALTIME_MODEL_ID},
            )
            voices = Path(directory) / "voices" / "realtime"
            (voices / "en").mkdir(parents=True)
            (voices / "en" / "Carter.pt").touch()
            available = discover_realtime_voices(settings)
            self.assertEqual(
                available["en/Carter"], str((voices / "en" / "Carter.pt").resolve())
            )
            self.assertTrue(set(OFFICIAL_VOICES).issubset(available))

    def test_voice_download_is_atomic_cached_and_allows_local_overrides(self):
        import io

        with tempfile.TemporaryDirectory() as directory:
            settings = ModelLoadingSettings(Path(directory))
            root = settings.models_dir / "voices" / "realtime"
            destination = root / "en-Carter_man.pt"
            with patch(
                "vibevoice.realtime.voices.urlopen", return_value=io.BytesIO(b"preset")
            ) as fetch:
                self.assertEqual(
                    resolve_voice_preset("en-Carter_man", settings), destination
                )
                self.assertEqual(
                    resolve_voice_preset("en-Carter_man", settings), destination
                )
                self.assertEqual(
                    discover_realtime_voices(settings)["en-Carter_man"],
                    str(destination.resolve()),
                )
                fetch.assert_called_once()
                self.assertIn("microsoft/VibeVoice/1541f590", fetch.call_args.args[0])
                self.assertEqual(
                    resolve_voice_preset(destination, settings), destination.resolve()
                )
                fetch.assert_called_once()
                with self.assertRaises(FileNotFoundError):
                    resolve_voice_preset(root / "missing.pt", settings)
                fetch.assert_called_once()
            destination.unlink()
            with (
                patch(
                    "vibevoice.realtime.voices.urlopen",
                    side_effect=OSError("interrupted"),
                ),
                self.assertRaisesRegex(OSError, "interrupted"),
            ):
                resolve_voice_preset("en-Carter_man", settings)
            self.assertEqual(list(root.iterdir()), [])
            with (
                patch("vibevoice.realtime.voices.urlopen", return_value=io.BytesIO()),
                self.assertRaisesRegex(ValueError, "is empty"),
            ):
                resolve_voice_preset("en-Carter_man", settings)
            self.assertEqual(list(root.iterdir()), [])

    def test_model_download_repairs_incomplete_checkpoint_and_reuses_local_files(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = ModelLoadingSettings(Path(directory))
            model = settings.tts_dir / "VibeVoice-Realtime-0.5B"
            model.mkdir(parents=True)
            (model / "config.json").write_text('{"model_type":"vibevoice_streaming"}')

            def download(repo, destination, actual_settings, **kwargs):
                self.assertEqual(repo, REALTIME_MODEL_ID)
                self.assertEqual(destination, model)
                self.assertEqual(actual_settings, settings)
                (model / "model.safetensors").touch()
                (model / "preprocessor_config.json").write_text("{}")
                return model

            with patch(
                "vibevoice.realtime.service.download_support_asset",
                side_effect=download,
            ) as fetch:
                self.assertEqual(
                    resolve_realtime_model(REALTIME_MODEL_ID, settings), model
                )
                self.assertEqual(
                    resolve_realtime_model(REALTIME_MODEL_ID, settings), model
                )
                fetch.assert_called_once()
                self.assertIn("*.safetensors", fetch.call_args.kwargs["allow_patterns"])
            (model / "model.safetensors").unlink()
            with (
                patch(
                    "vibevoice.realtime.service.download_support_asset",
                    return_value=model,
                ),
                self.assertRaisesRegex(ValueError, "No supported model weights"),
            ):
                resolve_realtime_model(REALTIME_MODEL_ID, settings)
            with patch("vibevoice.realtime.service.download_support_asset") as fetch:
                with self.assertRaises(OSError):
                    resolve_realtime_model(Path(directory) / "missing-custom", settings)
                fetch.assert_not_called()

    def test_tokenizer_download_fetches_only_tokenizer_files_and_reuses_bundled_assets(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            settings = ModelLoadingSettings(Path(directory))
            model = Path(directory) / "model"
            model.mkdir()
            (model / "preprocessor_config.json").write_text(
                '{"language_model_pretrained_name":"Qwen/Qwen2.5-0.5B"}'
            )
            tokenizer = settings.tokenizers_dir / "Qwen2.5-0.5B"

            def download(repo, destination, actual_settings, **kwargs):
                self.assertEqual(repo, "Qwen/Qwen2.5-0.5B")
                self.assertEqual(destination, tokenizer)
                self.assertEqual(actual_settings, settings)
                self.assertNotIn("*.safetensors", kwargs["allow_patterns"])
                destination.mkdir(parents=True)
                (destination / "tokenizer.json").write_text("{}")
                return destination

            with patch(
                "vibevoice.realtime.service.download_support_asset",
                side_effect=download,
            ) as fetch:
                self.assertEqual(resolve_realtime_tokenizer(model, settings), tokenizer)
                self.assertEqual(resolve_realtime_tokenizer(model, settings), tokenizer)
                fetch.assert_called_once()
                (model / "tokenizer.json").write_text("{}")
                self.assertEqual(resolve_realtime_tokenizer(model, settings), model)
                fetch.assert_called_once()
            (model / "tokenizer.json").unlink()
            (tokenizer / "tokenizer.json").unlink()
            with (
                patch(
                    "vibevoice.realtime.service.download_support_asset",
                    return_value=tokenizer,
                ),
                self.assertRaisesRegex(ValueError, "download is incomplete"),
            ):
                resolve_realtime_tokenizer(model, settings)

    def test_restricted_prompt_load_converts_legacy_cache_and_keeps_voice(self):
        config = tiny_config()
        original = prompt(config, legacy=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.pt"
            torch.save(original, path)
            loaded = load_voice_prompt(path, config, "cpu", torch.float32)
            self.assertEqual(loaded["lm"].past_key_values.get_seq_length(), 3)
            self.assertEqual(len(loaded["lm"].past_key_values.layers), 1)
            self.assertEqual(len(loaded["tts_lm"].past_key_values.layers), 1)
            keys = loaded["lm"].past_key_values.layers[0].keys
            torch.testing.assert_close(
                keys, original["lm"].past_key_values.key_cache[0]
            )
            copied = copy.deepcopy(loaded)
            copied["lm"].past_key_values.update(
                torch.zeros(1, 2, 1, 8), torch.zeros(1, 2, 1, 8), 0
            )
            self.assertEqual(loaded["lm"].past_key_values.get_seq_length(), 3)
            torch.save({"lm": original["lm"]}, path)
            with self.assertRaisesRegex(ValueError, "tts_lm"):
                load_voice_prompt(path, config, "cpu", torch.float32)
            torch.save(SimpleNamespace(unexpected=True), path)
            with self.assertRaises(pickle.UnpicklingError):
                load_voice_prompt(path, config, "cpu", torch.float32)

    def test_native_cache_loading_and_malformed_prompt_rejection_restore_loader(self):
        import torch._weights_only_unpickler as restricted

        original = getattr(restricted.Unpickler, "_check_set_item_target", None)
        config = tiny_config()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "voice.pt"
            cached = prompt(config)
            torch.save(cached, path)
            loaded = load_voice_prompt(path, config, "cpu", torch.float32)
            torch.testing.assert_close(
                loaded["lm"].past_key_values.layers[0].keys,
                cached["lm"].past_key_values.layers[0].keys,
            )
            self.assertIs(
                getattr(restricted.Unpickler, "_check_set_item_target", None), original
            )
            cached["lm"].past_key_values.layers[0].keys = torch.zeros(1, 1, 3, 8)
            torch.save(cached, path)
            with self.assertRaisesRegex(ValueError, "KV shape"):
                load_voice_prompt(path, config, "cpu", torch.float32)
            self.assertIs(
                getattr(restricted.Unpickler, "_check_set_item_target", None), original
            )

    def test_context_budget_fails_before_truncation(self):
        self.assertEqual(generation_budget(5, 3, 256), 44)
        with self.assertRaisesRegex(ValueError, "shorten the text"):
            generation_budget(10, 3, 32)


class CompatibilityTests(unittest.TestCase):
    def test_decoder_only_checkpoint_loads_and_rejects_missing_decoder_weights(self):
        from safetensors.torch import save_file

        config = tiny_config()
        model = VibeVoiceStreamingForConditionalGenerationInference(config).eval()
        model.model.speech_scaling_factor.fill_(1.0)
        model.model.speech_bias_factor.zero_()
        weights = {
            key: value.contiguous()
            for key, value in model.state_dict().items()
            if not key.startswith("model.acoustic_tokenizer.encoder.")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config.save_pretrained(root)
            save_file(weights, root / "model.safetensors")
            (root / "preprocessor_config.json").write_text(
                '{"speech_tok_compress_ratio":3200}'
            )
            (root / "tokenizer.json").write_text("{}")
            with patch.object(VibeVoiceStreamingProcessor, "from_pretrained"):
                _, loaded = load_realtime_model(root, ModelLoadingSettings(root), "cpu")
                audio = loaded.model.acoustic_tokenizer.decode(torch.zeros(1, 2, 3))
                self.assertTrue(torch.isfinite(audio).all())
                self.assertGreater(audio.numel(), 0)
                weights.pop(
                    next(
                        key
                        for key in weights
                        if key.startswith("model.acoustic_tokenizer.decoder.")
                    )
                )
                save_file(weights, root / "model.safetensors")
                with self.assertRaisesRegex(ValueError, "weights do not match"):
                    load_realtime_model(root, ModelLoadingSettings(root), "cpu")

    def test_checkpoint_round_trip_and_windowed_generation(self):
        config = tiny_config()
        model = VibeVoiceStreamingForConditionalGenerationInference(config).eval()
        model.model.speech_scaling_factor.fill_(1.0)
        model.model.speech_bias_factor.zero_()
        saved_weights = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            root = Path(directory)
            (root / "preprocessor_config.json").write_text(
                '{"speech_tok_compress_ratio":3200}'
            )
            (root / "tokenizer.json").write_text("{}")
            with patch.object(
                VibeVoiceStreamingProcessor, "from_pretrained"
            ) as processor_factory:
                _, model = load_realtime_model(root, ModelLoadingSettings(root), "cpu")
                self.assertTrue(processor_factory.call_args.kwargs["local_files_only"])
        self.assertEqual(saved_weights.keys(), model.state_dict().keys())
        for name, value in model.state_dict().items():
            torch.testing.assert_close(
                value, saved_weights[name], msg=lambda msg, name=name: f"{name}: {msg}"
            )
        self.assertEqual(model.model.language_model.config._attn_implementation, "sdpa")
        tokenizer = SimpleNamespace(
            pad_id=0,
            encode=lambda *args, **kw: list(range(5, 15)),
            convert_tokens_to_ids=lambda _: 3,
        )
        cached = prompt(config, negative_length=2)
        processor = VibeVoiceStreamingProcessor(tokenizer)
        inputs = processor.process_input_with_cached_prompt(
            "synthetic test", cached_prompt=cached
        )
        model.sample_speech_tokens = Mock(return_value=torch.zeros(1, 2))
        model.model.acoustic_tokenizer.decode = Mock(
            return_value=torch.zeros(1, 1, 3200)
        )
        # Keep EOS false and force the combined text/speech cap.
        model.tts_eos_classifier.forward = Mock(return_value=torch.tensor([[-100.0]]))
        forward_lm = model.forward_lm
        forward_tts_lm = model.forward_tts_lm
        seen_lengths = []

        def check_forward(forward, **kwargs):
            prefix = kwargs["past_key_values"].get_seq_length()
            query = kwargs["input_ids"].shape[1]
            seen_lengths.append(query)
            self.assertEqual(kwargs["attention_mask"].shape[1], prefix + query)
            torch.testing.assert_close(
                kwargs["cache_position"], torch.arange(prefix, prefix + query)
            )
            return forward(**kwargs)

        model.forward_lm = lambda **kwargs: check_forward(forward_lm, **kwargs)
        model.forward_tts_lm = lambda **kwargs: check_forward(forward_tts_lm, **kwargs)
        streamer = RealtimeStreamer(threading.Event())
        output = model.generate(
            **inputs,
            tokenizer=tokenizer,
            all_prefilled_outputs=copy.deepcopy(cached),
            max_new_tokens=15,
            audio_streamer=streamer,
            show_progress_bar=False,
        )
        chunks = list(streamer.get_stream(0))
        self.assertTrue(output.reach_max_step_sample.item())
        self.assertTrue(chunks)
        self.assertLessEqual(output.sequences.shape[-1], 18)
        self.assertEqual(
            output.speech_outputs[0].shape[-1], sum(c.size for c in chunks)
        )
        self.assertEqual(cached["tts_lm"].past_key_values.get_seq_length(), 3)
        self.assertIn(5, seen_lengths)
        self.assertIn(1, seen_lengths)

        # Exercise the next text window after interleaved speech updates both caches.
        seen_lengths.clear()
        multiwindow = model.generate(
            **inputs,
            tokenizer=tokenizer,
            all_prefilled_outputs=copy.deepcopy(cached),
            max_new_tokens=30,
            show_progress_bar=False,
        )
        self.assertEqual(seen_lengths.count(5), 4)
        self.assertTrue(multiwindow.reach_max_step_sample.item())

        # A budget smaller than the text window must stop before any forward or audio.
        model.sample_speech_tokens.reset_mock()
        capped = model.generate(
            **inputs,
            tokenizer=tokenizer,
            all_prefilled_outputs=copy.deepcopy(cached),
            max_new_tokens=4,
            show_progress_bar=False,
        )
        self.assertTrue(capped.reach_max_step_sample.item())
        self.assertEqual(capped.sequences.shape[1], 3)
        model.sample_speech_tokens.assert_not_called()

        # EOS ends the speech window immediately; cancellation can prevent its first latent.
        model.tts_eos_classifier.forward.return_value = torch.tensor([[100.0]])
        eos = model.generate(
            **inputs,
            tokenizer=tokenizer,
            all_prefilled_outputs=copy.deepcopy(cached),
            max_new_tokens=30,
            show_progress_bar=False,
        )
        self.assertFalse(eos.reach_max_step_sample.item())
        self.assertEqual(eos.speech_outputs[0].shape[-1], 3200)
        model.sample_speech_tokens.assert_called_once()
        model.sample_speech_tokens.reset_mock()
        model.generate(
            **inputs,
            tokenizer=tokenizer,
            all_prefilled_outputs=copy.deepcopy(cached),
            max_new_tokens=30,
            show_progress_bar=False,
            stop_check_fn=lambda: True,
        )
        model.sample_speech_tokens.assert_not_called()


class StreamingTests(unittest.TestCase):
    def test_streamer_rejects_nonfinite_audio_and_ignores_empty_chunks(self):
        streamer = RealtimeStreamer(threading.Event())
        streamer.put(torch.empty(1, 1, 0), torch.tensor([0]))
        streamer.end()
        self.assertEqual(list(streamer.get_stream(0)), [])
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            RealtimeStreamer(threading.Event()).put(
                torch.full((1, 1, 10), float("nan")), torch.tensor([0])
            )

    def test_first_chunk_precedes_completion_and_close_cancels_producer(self):
        release = threading.Event()
        finished = threading.Event()
        model = Mock()

        def generate(**kwargs):
            kwargs["audio_streamer"].put(
                torch.full((1, 1, 12800), 0.1), torch.tensor([0])
            )
            while not release.wait(0.01) and not kwargs["stop_check_fn"]():
                pass
            finished.set()
            return SimpleNamespace(reach_max_step_sample=torch.tensor([False]))

        model.generate.side_effect = generate
        service = RealtimeService(ModelLoadingSettings(Path("unused")), "cpu")
        service.model = model
        service.processor = Mock()
        service.processor.process_input_with_cached_prompt.return_value = {
            "tts_text_ids": torch.ones(1, 5, dtype=torch.long),
            "tts_lm_input_ids": torch.ones(1, 3, dtype=torch.long),
        }
        service.model.config.decoder_config.max_position_embeddings = 256
        service.prompt = {}
        stream = service.stream("hello world", seed=42)
        first = next(stream)
        self.assertEqual(first.shape, (12800,))
        self.assertFalse(finished.is_set())
        stream.close()
        self.assertTrue(finished.wait(1))

    def test_producer_failure_surfaces_before_or_after_audio(self):
        for emit in (False, True):
            service = RealtimeService(ModelLoadingSettings(Path("unused")), "cpu")
            service.model = Mock()
            service.model.config.decoder_config.max_position_embeddings = 256
            service.processor = Mock()
            service.processor.process_input_with_cached_prompt.return_value = {
                "tts_text_ids": torch.ones(1, 5, dtype=torch.long),
                "tts_lm_input_ids": torch.ones(1, 3, dtype=torch.long),
            }
            service.prompt = {}

            def generate(emit=emit, **kwargs):
                if emit:
                    kwargs["audio_streamer"].put(
                        torch.zeros(1, 1, 24000), torch.tensor([0])
                    )
                raise RuntimeError("producer failed")

            service.model.generate.side_effect = generate
            with self.assertRaisesRegex(RuntimeError, "producer failed"):
                list(service.stream("hello world"))

    def test_normal_completion_keeps_external_cancel_clear(self):
        service = RealtimeService(ModelLoadingSettings(Path("unused")), "cpu")
        service.model = Mock()
        service.model.config.decoder_config.max_position_embeddings = 256
        service.processor = Mock()
        service.processor.process_input_with_cached_prompt.return_value = {
            "tts_text_ids": torch.ones(1, 5, dtype=torch.long),
            "tts_lm_input_ids": torch.ones(1, 3, dtype=torch.long),
        }
        service.prompt = {}

        def generate(**kwargs):
            kwargs["audio_streamer"].put(torch.zeros(1, 1, 24000), torch.tensor([0]))
            return SimpleNamespace(reach_max_step_sample=torch.tensor([False]))

        service.model.generate.side_effect = generate
        cancel = threading.Event()
        self.assertEqual(len(list(service.stream("Hello", cancel_event=cancel))), 1)
        self.assertFalse(cancel.is_set())


class WorkerTests(unittest.TestCase):
    def test_real_process_chunk_order_and_terminal_errors(self):
        for text in ("complete", "fail", "crash", "wait"):
            with self.subTest(text=text):
                service = RealtimeWorkerService(
                    ModelLoadingSettings(Path("unused")), "cpu"
                )
                service.load("model", "voice")
                children = {child.pid for child in multiprocessing.active_children()}
                with patch(
                    "vibevoice.realtime.worker._realtime_worker", synthetic_worker
                ):
                    stream = service.stream(text)
                    if text == "complete":
                        chunks = list(stream)
                        self.assertEqual(
                            [float(chunk[0]) for chunk in chunks],
                            [float(np.float32(0.1)), float(np.float32(0.2))],
                        )
                    elif text == "wait":
                        first = next(stream)
                        self.assertEqual(first.size, 24000)
                        self.assertTrue(service.process.is_alive())
                        stream.close()
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "worker failed" if text == "fail" else "exit 3",
                        ):
                            list(stream)
                self.assertIsNone(service.process)
                self.assertFalse(service.lock.locked())
                self.assertEqual(
                    {child.pid for child in multiprocessing.active_children()}, children
                )

    def test_worker_protocol_uses_streaming_service_and_completion(self):
        import queue

        output, cancel = queue.Queue(), threading.Event()
        with patch("vibevoice.realtime.worker.RealtimeService") as factory:
            factory.return_value.stream.return_value = iter([np.zeros(10), np.ones(10)])
            _realtime_worker(
                "settings",
                "cpu",
                ("model", "voice"),
                "text",
                {"seed": 42},
                output,
                cancel,
            )
        self.assertEqual(
            [output.get()[0] for _ in range(4)], ["loaded", "chunk", "chunk", "done"]
        )
        factory.return_value.load.assert_called_once_with("model", "voice")
        self.assertEqual(factory.return_value.stream.call_args.kwargs["seed"], 42)


if __name__ == "__main__":
    unittest.main()
