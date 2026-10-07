import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import main as app_module


class DemoStub:
    def __init__(self):
        self.available_voices = {
            "en-Alice_woman": "alice.wav",
            "en-Carter_man": "carter.wav",
            "en-Frank_man": "frank.wav",
            "en-Maya_woman": "maya.wav",
        }
        self.available_models = {"Test model": "test-model"}
        self.model_path = "Test model"
        self.model_settings = SimpleNamespace(source="local", tts_dir=Path("models/tts"))
        self.inference_steps = 5
        self.load_on_demand = False
        self.model_loaded = True
        self.saved_audio_calls = []
        self.stop_called = False

    def setup_voice_presets(self):
        pass

    def ensure_model_loaded(self):
        pass

    def generate_podcast_streaming(self, **_kwargs):
        return iter(())

    def _save_generated_audio(self, audio_data, speaker_names):
        self.saved_audio_calls.append((audio_data, speaker_names))
        return "saved.wav"

    def stop_audio_generation(self):
        self.stop_called = True

    def switch_model(self, _model):
        return True


class MainInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = DemoStub()
        cls.interface = app_module.create_demo_interface(cls.demo)
        cls.functions = cls.interface.fns

    def function(self, name):
        return next(fn.fn for fn in self.functions.values() if getattr(fn.fn, "__name__", "") == name)

    def test_manual_audio_ui_and_model_switch_are_wired_without_ai_chat(self):
        components = self.interface.config["components"]
        component_text = [
            str(component.get("props", {}).get(key, ""))
            for component in components
            for key in ("label", "value", "info")
        ]
        labels = [component.get("props", {}).get("label") for component in components]
        button_values = [
            component.get("props", {}).get("value")
            for component in components
            if component.get("type") == "button"
        ]

        self.assertIn("Conversation Script", labels)
        self.assertIn("🚀 Generate Audio", button_values)
        self.assertIn("🛑 Stop Generation", button_values)
        self.assertIn("🔄 Load Selected Model", button_values)
        self.assertIn("Gain (dB)", labels)
        self.assertFalse(any("AI Chat" in text for text in component_text))
        self.assertFalse(any("Feeling Lucky" in text for text in component_text))
        self.assertFalse(any("Submit" == value for value in button_values))

        callback_names = {getattr(fn.fn, "__name__", "") for fn in self.functions.values()}
        for removed_callback in ("generate_ai_script", "feeling_lucky", "update_chat_history"):
            self.assertNotIn(removed_callback, callback_names)
        self.assertFalse(any("chat" in name.lower() for name in callback_names))

        button_ids = {
            component.get("props", {}).get("value"): component["id"]
            for component in components
            if component.get("type") == "button"
        }
        targets = {
            (target_id, event): dependency
            for dependency in self.interface.config["dependencies"]
            for target_id, event in dependency.get("targets", [])
        }
        self.assertIn((button_ids["🚀 Generate Audio"], "click"), targets)
        self.assertIn((button_ids["🛑 Stop Generation"], "click"), targets)
        self.assertIn((button_ids["🔄 Load Selected Model"], "click"), targets)
        for button, callback in (
            ("🚀 Generate Audio", "clear_audio_outputs"),
            ("🛑 Stop Generation", "stop_generation_handler"),
            ("🔄 Load Selected Model", "switch_model"),
        ):
            dependency = targets[(button_ids[button], "click")]
            self.assertEqual(self.functions[dependency["id"]].fn.__name__, callback)

        generate = self.function("generate_podcast_wrapper")
        generate_entry = next(fn for fn in self.functions.values() if fn.fn is generate)
        self.assertEqual(len(generate_entry.outputs), 6)
        switch = self.function("switch_model")
        switch_entry = next(fn for fn in self.functions.values() if fn.fn is switch)
        self.assertEqual(len(switch_entry.outputs), 5)

    def test_generation_callback_yields_six_outputs_for_start_stream_complete_and_save(self):
        stream_audio = np.array([0.1, 0.2], dtype=np.float32)
        complete_audio = (24000, np.array([0.2, -0.2], dtype=np.float32))
        self.demo.saved_audio_calls.clear()
        self.demo.generate_podcast_streaming = lambda **_kwargs: iter([
            (stream_audio, None, "stream chunk", True),
            (None, None, "stream status", True),
            (None, complete_audio, "generation complete", False),
        ])

        with patch.object(app_module, "cache_original_audio") as cache_audio:
            results = list(self.function("generate_podcast_wrapper")(
                2,
                "Speaker 1: Hello.\nSpeaker 2: Hi.",
                "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman",
                1.6, 10, True, 0.95, 0.95, 0, "", True, False, True,
            ))

        self.assertEqual([len(result) for result in results], [6, 6, 6, 6])
        self.assertEqual(results[0][2], "🎙️ Starting generation...")
        self.assertIs(results[1][0], stream_audio)
        self.assertEqual(results[-1][1]["value"], complete_audio)
        self.assertIn("saved.wav", results[-1][2])
        self.assertEqual(self.demo.saved_audio_calls, [(complete_audio, ["en-Alice_woman", "en-Carter_man"])])
        cache_audio.assert_called_once_with(complete_audio)

    def test_generation_callback_resets_all_six_outputs_after_error(self):
        def fail_generation(**_kwargs):
            raise RuntimeError("mock generation failure")

        self.demo.generate_podcast_streaming = fail_generation
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            results = list(self.function("generate_podcast_wrapper")(
                1,
                "Speaker 1: Hello.",
                "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman",
                1.6, 10, True, 0.95, 0.95, 0, "", True, False, False,
            ))
        self.assertEqual([len(result) for result in results], [6, 6])
        self.assertIn("mock generation failure", results[-1][2])

    def test_stop_callback_and_audio_clear_keep_their_output_arity(self):
        stop_result = self.function("stop_generation_handler")()
        clear_result = self.function("clear_audio_outputs")()
        self.assertEqual(len(stop_result), 4)
        self.assertEqual(len(clear_result), 2)
        self.assertTrue(self.demo.stop_called)

    def test_saved_audio_keeps_speaker_based_name_without_ai_topic(self):
        with tempfile.TemporaryDirectory() as directory:
            audio = (24000, np.array([0.1], dtype=np.float32))
            with patch.object(app_module.os.path, "dirname", return_value=directory), patch.object(
                app_module.sf, "write"
            ) as write_audio:
                path = app_module.VibeVoiceDemo._save_generated_audio(
                    self.demo, audio, ["en-Alice_woman", "en-Carter_man"]
                )

        self.assertIn("Alice-woman_Carter-man_audio-generation_001.wav", path)
        write_audio.assert_called_once_with(path, audio[1], audio[0])

    def test_gain_cache_applies_gain_to_original_audio(self):
        samples = np.array([0.2, -0.2], dtype=np.float32)
        with (
            patch.object(app_module, "_original_audio_cache", None),
            patch.object(app_module, "_last_cached_audio_hash", None),
            patch.object(app_module, "_current_gain_db", 0.0),
        ):
            app_module.cache_original_audio((24000, samples))
            output, _info = app_module.apply_gain_to_complete_audio(None, 6.0)

        self.assertEqual(output[0], 24000)
        self.assertEqual(output[1].dtype, np.int16)
        self.assertGreater(abs(int(output[1][0])), int(samples[0] * 32767))

    def test_removed_script_ai_flags_are_rejected_and_model_flags_remain(self):
        with patch.object(sys, "argv", ["main.py", "--model-source", "local", "--model-path", "VibeVoice-1.5B", "--debug"]):
            args = app_module.parse_args()
        self.assertEqual(args.model_source, "local")
        self.assertTrue(args.debug)

        for removed_flag in (
            "--script-ai-url",
            "--script_ai_url",
            "--script-ai-model",
            "--script_ai_model",
            "--script-ai-api-key",
            "--script_ai_api_key",
        ):
            with self.subTest(flag=removed_flag):
                with patch.object(sys, "argv", ["main.py", removed_flag, "unused"]):
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exit_info:
                        app_module.parse_args()
                self.assertEqual(exit_info.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
