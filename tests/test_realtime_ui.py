import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from gradio.state_holder import SessionState
from test_main_interface import DemoStub, parse_select_markup

import main
from vibevoice.runtime.model_loading import ModelLoadingSettings


class RealtimeUITests(unittest.TestCase):
    def build(self, lod=False, models=None, voices=None):
        demo = DemoStub()
        demo.load_on_demand = lod
        demo.unload_model = Mock()
        with (
            patch(
                "vibevoice.realtime.ui.discover_realtime_models",
                return_value={"Realtime": "model"} if models is None else models,
            ),
            patch(
                "vibevoice.realtime.ui.discover_realtime_voices",
                return_value={"Carter": "voice"} if voices is None else voices,
            ),
        ):
            interface = main.create_demo_interface(demo)
        functions = {
            fn.fn.__name__: fn
            for fn in interface.fns.values()
            if hasattr(fn.fn, "__name__")
        }
        controller = functions["generate_realtime"].fn.__self__
        controller.service = Mock()
        return demo, interface, functions, controller

    def test_controls_and_shared_model_lifecycle(self):
        _demo, interface, functions, controller = self.build()
        props = {
            item.get("props", {}).get("label"): item.get("props", {})
            for item in interface.config["components"]
        }
        for label in (
            "Realtime Text",
            "Realtime Seed",
        ):
            self.assertIn(label, props)
        selectors = {
            item["props"].get("elem_id"): item
            for item in interface.config["components"]
        }
        for elem_id, choice in (
            ("realtime-model-select-field", "Realtime"),
            ("realtime-voice-select-field", "Carter"),
        ):
            component = selectors[elem_id]
            self.assertEqual(component["type"], "html")
            self.assertEqual(
                parse_select_markup(component["props"]["value"]), ([choice], choice)
            )
        self.assertTrue(props["Realtime Live Audio"]["streaming"])
        self.assertTrue(props["Realtime Live Audio"]["autoplay"])
        self.assertEqual(
            functions["generate_realtime"].concurrency_id, "model_operations"
        )
        self.assertFalse(functions["stop_realtime"].queue)
        functions["switch_model"].fn("Test model")
        controller.service.unload.assert_called_once()

    def test_refresh_preserves_native_selection_and_falls_back_when_removed(self):
        special = 'Voice & <One> "A" / 音'
        _, interface, _, controller = self.build(
            models={"First": "first", "Second": "second"},
            voices={"Carter": "carter", special: "special"},
        )
        with (
            patch(
                "vibevoice.realtime.ui.discover_realtime_models",
                return_value=controller.models,
            ),
            patch(
                "vibevoice.realtime.ui.discover_realtime_voices",
                return_value=controller.voices,
            ),
        ):
            model, voice, generate, _ = controller.refresh("Second", special)
            self.assertEqual(
                parse_select_markup(model), (["First", "Second"], "Second")
            )
            self.assertEqual(parse_select_markup(voice), (["Carter", special], special))
            self.assertIn("&amp; &lt;One&gt; &quot;A&quot;", voice)
            self.assertTrue(generate["interactive"])
            model, voice, _, _ = controller.refresh("Removed", "Removed")
            self.assertEqual(parse_select_markup(model)[1], "First")
            self.assertEqual(parse_select_markup(voice)[1], "Carter")
        dependencies = {
            interface.fns[dep["id"]].fn.__name__: dep
            for dep in interface.config["dependencies"]
            if interface.fns[dep["id"]].fn is not None
        }
        for name in ("refresh", "generate_realtime"):
            self.assertIn("document.getElementById", dependencies[name]["js"])
            self.assertIn("DOMParser", dependencies[name]["js"])
            self.assertIn("realtime-model-select", dependencies[name]["js"])
            self.assertIn("realtime-voice-select", dependencies[name]["js"])
        self.assertIn("...values.slice(2)", dependencies["generate_realtime"]["js"])

    def test_empty_native_selects_show_asset_paths_and_disable_generation(self):
        _, interface, _, controller = self.build(models={}, voices={})
        components = {
            item["props"].get("elem_id"): item
            for item in interface.config["components"]
        }
        for elem_id, path in (
            ("realtime-model-select-field", "models/tts"),
            ("realtime-voice-select-field", "models/voices/realtime"),
        ):
            markup = components[elem_id]["props"]["value"]
            self.assertEqual(parse_select_markup(markup), ([], None))
            self.assertIn(" disabled>", markup)
            self.assertIn(path, markup)
        generate = next(
            item
            for item in interface.config["components"]
            if item["props"].get("value") == "Generate Realtime Speech"
        )
        self.assertFalse(generate["props"]["interactive"])
        self.assertIn(
            "checkpoint and cached .pt voice presets", controller.asset_status()
        )

    def test_fresh_install_lists_downloadable_assets_without_startup_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            demo = DemoStub()
            demo.model_settings = ModelLoadingSettings(Path(directory))
            with (
                patch("vibevoice.realtime.service.download_support_asset") as download,
                patch("vibevoice.realtime.voices.urlopen") as voice_download,
            ):
                interface = main.create_demo_interface(demo)
            download.assert_not_called()
            voice_download.assert_not_called()
            components = {
                item["props"].get("elem_id"): item
                for item in interface.config["components"]
            }
            models, model = parse_select_markup(
                components["realtime-model-select-field"]["props"]["value"]
            )
            voices, voice = parse_select_markup(
                components["realtime-voice-select-field"]["props"]["value"]
            )
            self.assertEqual(models, ["microsoft/VibeVoice-Realtime-0.5B"])
            self.assertEqual(model, models[0])
            self.assertEqual(len(voices), 25)
            self.assertEqual(voice, "en-Carter_man")
            generate = next(
                item
                for item in interface.config["components"]
                if item["props"].get("value") == "Generate Realtime Speech"
            )
            self.assertTrue(generate["props"]["interactive"])

    def test_pcm_chunks_and_complete_download_preserve_order_and_scale(self):
        demo, _, functions, controller = self.build()
        chunks = [
            np.array([0.1, 0.2], dtype=np.float32),
            np.array([-0.1, 2.0], dtype=np.float32),
        ]
        controller.service.stream.return_value = (chunk for chunk in chunks)
        output = list(
            functions["generate_realtime"].fn(
                "Realtime", "Carter", "Hello world", 123, 1.5, 5
            )
        )
        self.assertEqual([len(item) for item in output], [3, 3, 3, 3])
        complete = output[-1][1]["value"]
        self.assertEqual(complete[0], 24000)
        np.testing.assert_array_equal(
            complete[1], np.array([3276, 6553, -3276, 32767], dtype=np.int16)
        )
        self.assertIn("complete", output[-1][2])
        demo.unload_model.assert_called_once()
        controller.service.load.assert_called_once_with("model", "voice")
        self.assertEqual(controller.service.stream.call_args.kwargs["seed"], 123)

    def test_owner_stop_and_disconnect_do_not_cancel_another_session(self):
        _, _, functions, controller = self.build()
        controller.service.stream.return_value = (
            chunk for chunk in [np.zeros(24000), np.zeros(24000)]
        )
        owner, peer = (
            SimpleNamespace(session_hash="owner"),
            SimpleNamespace(session_hash="peer"),
        )
        stream = functions["generate_realtime"].fn(
            "Realtime", "Carter", "Hello", 42, 1.5, 5, owner
        )
        next(stream)
        next(stream)
        controller.disconnect(peer)
        controller.stop_realtime(peer)
        self.assertFalse(controller.cancel.is_set())
        controller.stop_realtime(owner)
        self.assertTrue(controller.cancel.is_set())
        remaining = list(stream)
        self.assertEqual(len(remaining), 1)
        self.assertIsNone(remaining[0][0]["value"])
        self.assertIn("stopped", remaining[0][2])
        controller.service.stop.assert_called_once()

    def test_error_and_empty_input_are_visible_and_leave_no_active_owner(self):
        demo, _, _functions, controller = self.build()
        output = list(
            controller.generate_realtime("Realtime", "Carter", "", 42, 1.5, 5)
        )
        self.assertIn("Provide text", output[-1][2])
        demo.unload_model.assert_not_called()
        controller.service.load.side_effect = ValueError("invalid prompt")
        output = list(
            controller.generate_realtime("Realtime", "Carter", "Hello", 42, 1.5, 5)
        )
        self.assertIn("invalid prompt", output[-1][2])
        self.assertIsNone(controller.owner)
        self.assertIsNone(controller.cancel)


class RealtimeGradioLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def finish_generation(self, interface, fn, text="Hello"):
        state = SessionState(interface)
        iterator = None
        statuses = []
        for _ in range(8):
            result = await interface.process_api(
                fn,
                ["Realtime", "Carter", text, 42, 1.5, 5],
                state=state,
                iterator=iterator,
                session_hash="realtime-session",
                simple_format=True,
            )
            statuses.append(str(result["data"][2]))
            iterator = result["iterator"]
            if not result["is_generating"]:
                break
        self.assertIsNone(iterator)
        return statuses, result["data"]

    async def test_loading_failure_preserves_status_and_finishes_without_audio(self):
        _, interface, functions, controller = RealtimeUITests().build()
        controller.service.load.side_effect = ValueError("checkpoint mismatch")
        statuses, _ = await self.finish_generation(
            interface, functions["generate_realtime"]
        )
        self.assertTrue(any("checkpoint mismatch" in status for status in statuses))
        self.assertIsNone(controller.owner)
        self.assertIsNone(controller.cancel)
        stream = next(
            iter(
                next(
                    iter(interface.pending_streams["realtime-session"].values())
                ).values()
            )
        )
        self.assertTrue(stream.ended)
        self.assertEqual(stream.segments, [])

    async def test_validation_failure_opens_no_audio_segments_and_finishes(self):
        _, interface, functions, controller = RealtimeUITests().build()
        statuses, _ = await self.finish_generation(
            interface, functions["generate_realtime"], text=""
        )
        self.assertTrue(any("Provide text" in status for status in statuses))
        self.assertIsNone(controller.cancel)
        stream = next(
            iter(
                next(
                    iter(interface.pending_streams["realtime-session"].values())
                ).values()
            )
        )
        self.assertTrue(stream.ended)
        self.assertEqual(stream.segments, [])

    async def test_success_closes_stream_and_keeps_complete_download(self):
        import soundfile as sf

        _, interface, functions, controller = RealtimeUITests().build()
        controller.service.stream.return_value = (
            chunk for chunk in [np.full(24000, 0.1, dtype=np.float32)]
        )
        statuses, data = await self.finish_generation(
            interface, functions["generate_realtime"]
        )
        self.assertTrue(any("complete" in status for status in statuses))
        complete = data[1]["value"]["path"]
        audio, rate = sf.read(complete, dtype="int16")
        self.assertEqual(rate, 24000)
        np.testing.assert_array_equal(audio, np.full(24000, 3276, dtype=np.int16))
        stream = next(
            iter(
                next(
                    iter(interface.pending_streams["realtime-session"].values())
                ).values()
            )
        )
        self.assertTrue(stream.ended)
        self.assertEqual(len(stream.segments), 1)

    async def test_stop_during_loading_finishes_without_audio(self):
        _, interface, functions, controller = RealtimeUITests().build()
        controller.service.load.side_effect = lambda *_: controller.cancel.set()
        statuses, _ = await self.finish_generation(
            interface, functions["generate_realtime"]
        )
        self.assertTrue(any("stopped" in status for status in statuses))
        stream = next(
            iter(
                next(
                    iter(interface.pending_streams["realtime-session"].values())
                ).values()
            )
        )
        self.assertTrue(stream.ended)
        self.assertEqual(stream.segments, [])


if __name__ == "__main__":
    unittest.main()
