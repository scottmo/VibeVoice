import contextlib
from html.parser import HTMLParser
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import main as app_module


class SelectMarkupParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.options = []
        self.selected = None

    def handle_starttag(self, tag, attrs):
        if tag == "option":
            attrs = dict(attrs)
            value = attrs.get("value")
            self.options.append(value)
            if "selected" in attrs:
                self.selected = value


def parse_select_markup(markup):
    parser = SelectMarkupParser()
    parser.feed(markup)
    return parser.options, parser.selected


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
        self.model_settings = SimpleNamespace(source="local", models_dir=Path("models"), tts_dir=Path("models/tts"))
        self.device = "cpu"
        self.inference_steps = 5
        self.load_on_demand = False
        self.model_loaded = True
        self.saved_audio_calls = []
        self.stop_called = False
        self.switch_model_calls = []
        self.switch_model_result = True
        self.switch_model_error = None

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

    def switch_model(self, model):
        self.switch_model_calls.append(model)
        if self.switch_model_error:
            raise self.switch_model_error
        if self.switch_model_result:
            self.model_path = model
        return self.switch_model_result


class MainInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.demo = DemoStub()
        cls.interface = app_module.create_demo_interface(cls.demo)
        cls.functions = cls.interface.fns

    def function(self, name):
        return next(fn.fn for fn in self.functions.values() if getattr(fn.fn, "__name__", "") == name)

    def test_theme_toggle_and_initialization_run_in_browser_without_model_queue(self):
        button = next(component for component in self.interface.config["components"]
                      if component.get("props", {}).get("elem_id") == "theme-toggle")
        dependencies = self.interface.config["dependencies"]
        toggle = next(dependency for dependency in dependencies
                      if (button["id"], "click") in dependency["targets"])
        initialize = next(dependency for dependency in dependencies
                          if dependency["outputs"] == [button["id"]]
                          and any(event == "load" for _, event in dependency["targets"]))
        for dependency in (toggle, initialize):
            self.assertIsNone(self.functions[dependency["id"]].fn)
            self.assertFalse(dependency["queue"])
            self.assertEqual(dependency["inputs"], [])
            self.assertEqual(dependency["outputs"], [button["id"]])
            self.assertTrue(dependency["js"])

    def test_startup_opens_page_without_loading_and_default_selection_loads_manually(self):
        settings = app_module.ModelLoadingSettings("local", Path("models"), False, True)
        for lod in (False, True):
            with (
                self.subTest(lod=lod),
                contextlib.redirect_stdout(io.StringIO()),
                patch.object(app_module, "discover_local_models", return_value={"Test model": "test-model"}),
                patch.object(app_module.VibeVoiceDemo, "load_model") as load,
                patch.object(app_module.VibeVoiceDemo, "_spawn_worker_process") as spawn,
            ):
                demo = app_module.VibeVoiceDemo(
                    "Test model", device="cpu", load_on_demand=lod, model_settings=settings,
                )
                interface = app_module.create_demo_interface(demo)
                load.assert_not_called()
                spawn.assert_not_called()
                self.assertFalse(demo.model_loaded)
                self.assertIsNone(demo.model)
                self.assertIsNone(demo.processor)
                self.assertTrue(demo.available_voices)
                log = next(component for component in interface.config["components"]
                           if component.get("props", {}).get("label") == "Generation Log")
                self.assertIn("No model loaded", log["props"]["value"])
                callback = next(fn.fn for fn in interface.fns.values()
                                if getattr(fn.fn, "__name__", "") == "switch_model")

                def mark_loaded():
                    demo.model_loaded = True

                load.side_effect = mark_loaded
                result = callback("Test model", *list(demo.available_voices)[:4])
                self.assertIn("✅", result[0])
                if lod:
                    load.assert_not_called()
                    demo.ensure_model_loaded()
                    spawn.assert_called_once_with()
                else:
                    load.assert_called_once_with()
                    self.assertTrue(demo.model_loaded)
                    callback("Test model")
                    load.assert_called_once_with()

    def test_asr_ui_callback_and_shared_model_queue(self):
        demo = DemoStub()
        demo.unload_model = Mock()
        with patch.object(app_module, "discover_asr_models", return_value={"ASR": "checkpoint"}):
            interface = app_module.create_demo_interface(demo)
        callback = next(fn.fn for fn in interface.fns.values() if getattr(fn.fn, "__name__", "") == "transcribe_upload")
        components = interface.config["components"]
        labels = [component.get("props", {}).get("label") for component in components]
        for label in ("Upload Audio", "Transcript", "Segments (timestamps and speaker IDs)", "Seed"):
            self.assertIn(label, labels)
        upload = next(component for component in components if component.get("props", {}).get("label") == "Upload Audio")
        self.assertEqual(upload["props"]["type"], "filepath")
        self.assertEqual(upload["props"]["sources"], ["upload"])
        self.assertIn("Upload", list(callback(None, "ASR", ""))[-1][2])
        demo.unload_model.assert_not_called()
        payload = {"transcript": "[1] Hi", "segments": [{"speaker": 0, "start": 0, "end": 1, "text": "Hi"}]}
        with patch.object(app_module, "asr_python"), patch.object(app_module, "run_transcription", return_value=payload) as transcribe:
            results = list(callback("upload.wav", "ASR", "Names"))
        self.assertEqual(results[-1], (payload["transcript"], payload["segments"], "Transcription complete."))
        transcribe.assert_called_once_with("upload.wav", "checkpoint", "cpu", "Names")
        demo.unload_model.assert_called_once()
        model_functions = [fn for fn in interface.fns.values() if getattr(fn.fn, "__name__", "") in
                           {"transcribe_upload", "generate_podcast_wrapper", "switch_model"}]
        self.assertEqual(len(model_functions), 3)
        self.assertTrue(all(fn.concurrency_id == "model_operations" for fn in model_functions))

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
        self.assertEqual(len(switch_entry.outputs), 6)

        selector_components = {
            component.get("props", {}).get("elem_id"): component
            for component in components
            if component.get("props", {}).get("elem_id") in {
                "speaker-select-field-1", "speaker-select-field-2",
                "speaker-select-field-3", "speaker-select-field-4", "model-select-field",
            }
        }
        self.assertEqual(len(selector_components), 5)
        self.assertTrue(all(component.get("type") == "html" for component in selector_components.values()))
        self.assertTrue(all("<select " in component["props"]["value"] for component in selector_components.values()))

        dependencies_by_name = {
            self.functions[dependency["id"]].fn.__name__: dependency
            for dependency in self.interface.config["dependencies"]
            if dependency["id"] in self.functions
            and self.functions[dependency["id"]].fn is not None
        }
        for name in ("update_speaker_visibility", "refresh_voices", "generate_podcast_wrapper", "switch_model"):
            self.assertIn("document.getElementById", dependencies_by_name[name]["js"])
            self.assertIn("DOMParser", dependencies_by_name[name]["js"])

    def test_native_select_markup_escapes_values_and_uses_valid_defaults(self):
        special_voice = 'Guest & <Host> "A" / 音'
        special_model = 'Model & <One> "v2" / 音'
        demo = DemoStub()
        demo.available_voices = {special_voice: "special.wav", "Second voice": "second.wav"}
        demo.available_models = {special_model: "model-path"}
        demo.model_path = special_model
        interface = app_module.create_demo_interface(demo)
        components = {
            component.get("props", {}).get("elem_id"): component
            for component in interface.config["components"]
        }

        speaker_markup = components["speaker-select-field-1"]["props"]["value"]
        model_markup = components["model-select-field"]["props"]["value"]
        speaker_options, selected_speaker = parse_select_markup(speaker_markup)
        model_options, selected_model = parse_select_markup(model_markup)

        self.assertEqual(speaker_options, [special_voice, "Second voice"])
        self.assertEqual(selected_speaker, special_voice)
        self.assertEqual(model_options, [special_model])
        self.assertEqual(selected_model, special_model)
        self.assertIn("Guest &amp; &lt;Host&gt; &quot;A&quot; / 音", speaker_markup)
        self.assertIn("Model &amp; &lt;One&gt; &quot;v2&quot; / 音", model_markup)

    def test_empty_native_selects_are_disabled_and_have_help_text(self):
        demo = DemoStub()
        demo.available_voices = {}
        demo.available_models = {}
        demo.model_path = "missing model"
        interface = app_module.create_demo_interface(demo)
        components = {
            component.get("props", {}).get("elem_id"): component
            for component in interface.config["components"]
        }

        for elem_id in (
            "speaker-select-field-1", "speaker-select-field-2",
            "speaker-select-field-3", "speaker-select-field-4", "model-select-field",
        ):
            markup = components[elem_id]["props"]["value"]
            options, selected = parse_select_markup(markup)
            self.assertEqual(options, [])
            self.assertIsNone(selected)
            self.assertIn(" disabled>", markup)
            self.assertIn("No ", markup)

    def test_generation_callback_yields_six_outputs_for_start_stream_complete_and_save(self):
        stream_audio = np.array([0.1, 0.2], dtype=np.float32)
        complete_audio = (24000, np.array([0.2, -0.2], dtype=np.float32))
        self.demo.saved_audio_calls.clear()
        generated_requests = []

        def mock_generate(**kwargs):
            generated_requests.append(kwargs)
            return iter([
                (stream_audio, None, "stream chunk", True),
                (None, None, "stream status", True),
                (None, complete_audio, "generation complete", False),
            ])

        self.demo.generate_podcast_streaming = mock_generate
        special_speakers = ['A & <One> "Q" / 音', "B / 二", "C", "D"]

        with patch.object(app_module, "cache_original_audio") as cache_audio:
            results = list(self.function("generate_podcast_wrapper")(
                2,
                "Speaker 1: Hello.\nSpeaker 2: Hi.",
                *special_speakers,
                1.6, 10, True, 0.95, 0.95, 0, "", True, False, True, 12345,
            ))

        self.assertEqual([len(result) for result in results], [6, 6, 6, 6])
        self.assertEqual(results[0][2], "🎙️ Starting generation...")
        self.assertIs(results[1][0], stream_audio)
        self.assertEqual(results[-1][1]["value"], complete_audio)
        self.assertIn("saved.wav", results[-1][2])
        self.assertEqual(self.demo.saved_audio_calls, [(complete_audio, special_speakers[:2])])
        self.assertEqual(
            [generated_requests[0][f"speaker_{i + 1}"] for i in range(4)],
            special_speakers,
        )
        cache_audio.assert_called_once_with(complete_audio)
        self.assertEqual(generated_requests[0]["seed"], 12345)

    def test_refresh_preserves_current_voice_and_falls_back_when_removed(self):
        previous_voices = self.demo.available_voices
        refreshed_voices = {"First & <new>": "first.wav", "Kept voice / 二": "kept.wav"}
        self.demo.available_voices = refreshed_voices
        try:
            result = self.function("refresh_voices")(
                "Kept voice / 二", "removed voice", "First & <new>", None
            )
        finally:
            self.demo.available_voices = previous_voices

        selected = [parse_select_markup(markup)[1] for markup in result]
        self.assertEqual(selected, ["Kept voice / 二", "First & <new>", "First & <new>", "First & <new>"])
        self.assertIn("First &amp; &lt;new&gt;", result[0])

    def test_model_switch_refreshes_voice_choices_and_keeps_matching_speakers(self):
        previous_models = self.demo.available_models
        previous_voices = self.demo.available_voices
        previous_model_path = self.demo.model_path
        new_voices = {"New voice & one": "new.wav", "Kept voice": "kept.wav"}
        self.demo.available_models = {"Test model": "test-model", "Second model": "second-model"}
        self.demo.available_voices = {"Kept voice": "kept.wav", "Removed voice": "removed.wav"}
        self.demo.model_path = "Test model"
        try:
            with patch.object(
                self.demo,
                "setup_voice_presets",
                side_effect=lambda: setattr(self.demo, "available_voices", new_voices),
            ):
                result = self.function("switch_model")(
                    "Second model", "Kept voice", "Removed voice", "Kept voice", "Removed voice"
                )
        finally:
            self.demo.available_models = previous_models
            self.demo.available_voices = previous_voices
            self.demo.model_path = previous_model_path

        self.assertEqual(parse_select_markup(result[1])[1], "Second model")
        self.assertEqual([parse_select_markup(markup)[1] for markup in result[2:]], [
            "Kept voice", "New voice & one", "Kept voice", "New voice & one",
        ])
        self.assertIn("New voice &amp; one", result[2])

    def test_speaker_count_hides_and_restores_without_losing_four_values(self):
        callback = self.function("update_speaker_visibility")
        selected = ["en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"]
        four_visible = callback(4, *selected)
        two_visible = callback(2, *[parse_select_markup(item["value"])[1] for item in four_visible])
        four_again = callback(4, *[parse_select_markup(item["value"])[1] for item in two_visible])

        self.assertEqual([parse_select_markup(item["value"])[1] for item in four_again], [
            "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman",
        ])
        self.assertEqual([item["visible"] for item in two_visible], [True, True, "hidden", "hidden"])
        self.assertEqual([item["visible"] for item in four_again], [True, True, True, True])

    def test_model_switch_receives_exact_native_value_and_preserves_state_on_failure(self):
        previous_models = self.demo.available_models
        previous_model_path = self.demo.model_path
        previous_result = self.demo.switch_model_result
        special_model = 'Voice & <Model> "A" / 音'
        self.demo.available_models = {"Test model": "test-model", special_model: "special-model"}
        self.demo.model_path = "Test model"
        self.demo.switch_model_result = True
        try:
            success = self.function("switch_model")(
                special_model, "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"
            )
            self.assertEqual(self.demo.switch_model_calls[-1], special_model)
            self.assertEqual(parse_select_markup(success[1])[1], special_model)

            self.demo.switch_model_result = False
            failed = self.function("switch_model")(
                "Test model", "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"
            )
            self.assertIn("Failed to switch", failed[0])
            self.assertEqual(parse_select_markup(failed[1])[1], special_model)
            self.assertEqual(self.demo.model_path, special_model)
            self.assertTrue(all(update == {"__type__": "update"} for update in failed[2:]))

            self.demo.switch_model_result = True
            with patch.object(self.demo, "setup_voice_presets", side_effect=RuntimeError("voice refresh failed")):
                setup_failed = self.function("switch_model")(
                    "Test model", "en-Alice_woman", "en-Carter_man", "en-Frank_man", "en-Maya_woman"
                )
            self.assertIn("voice refresh failed", setup_failed[0])
            self.assertEqual(parse_select_markup(setup_failed[1])[1], "Test model")
            self.assertEqual(self.demo.model_path, "Test model")
        finally:
            self.demo.available_models = previous_models
            self.demo.model_path = previous_model_path
            self.demo.switch_model_result = previous_result

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
