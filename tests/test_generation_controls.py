import json
from pathlib import Path
import queue
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main
from vibevoice.asr import discover_asr_models, run_transcription, validate_asr_model
from vibevoice.asr_worker import normalize_result
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.script import normalize_script, parse_script, resolve_seed


class SpeakerTests(unittest.TestCase):
    def test_short_long_case_colon_and_continuations(self):
        text = "[1] Hello\ncontinued.\n\n[2]: Hi: there\nsPeAkEr 1: Bye"
        self.assertEqual(parse_script(text, 2), [(0, "Hello continued."), (1, "Hi: there"), (0, "Bye")])
        self.assertEqual(parse_script("Speaker 0: Hello\nSpeaker 1: Hi", 2), [(0, "Hello"), (1, "Hi")])

    def test_plain_text_keeps_rotation(self):
        self.assertEqual(normalize_script("One\n\nTwo\nThree", 2), "Speaker 1: One\nSpeaker 2: Two\nSpeaker 1: Three")

    def test_invalid_labels_and_empty_turns_are_rejected(self):
        for script in ("[0] Hi", "[3] Hi", "[1", "Speaker x: Hi", "[1]", "[1] \n[2] Hi", ""):
            with self.subTest(script=script), self.assertRaises(ValueError):
                normalize_script(script, 2)

    def test_subset_and_out_of_order_speakers_keep_voice_index(self):
        processor = object.__new__(VibeVoiceProcessor)
        for script, expected in (("[2] Hi", [(1, " Hi")]), ("[3] Hi\n[1] Hello", [(2, " Hi"), (0, " Hello")])):
            self.assertEqual(processor._parse_script(normalize_script(script, 3)), expected)
            self.assertEqual(processor._parse_script(script), expected)
        processor.tokenizer = Mock()
        processor.system_prompt = "system"
        processor.tokenizer.encode.return_value = []
        processor.tokenizer.speech_start_id = 1
        processor.tokenizer.speech_end_id = 2
        processor._create_voice_prompt = Mock(return_value=([], [], []))
        # The preprocessing path must retain references through speaker 2,
        # rather than truncating to the number of distinct speakers (one).
        processor._process_single("[2] Hi", ["first.wav", "second.wav", "third.wav"])
        processor._create_voice_prompt.assert_called_once_with(["first.wav", "second.wav"])

    def test_text_file_short_labels_and_plain_single_speaker(self):
        processor = object.__new__(VibeVoiceProcessor)
        with tempfile.TemporaryDirectory() as directory:
            text_file = Path(directory) / "script.txt"
            text_file.write_text("[2] Hello\ncontinued\n[1] Bye", encoding="utf-8")
            self.assertEqual(
                processor._convert_text_to_script(str(text_file)),
                "Speaker 2: Hello continued\nSpeaker 1: Bye",
            )
            text_file.write_text("One\nTwo", encoding="utf-8")
            self.assertEqual(
                processor._convert_text_to_script(str(text_file)),
                "Speaker 1: One\nSpeaker 1: Two",
            )


class SeedTests(unittest.TestCase):
    def test_seed_validation_and_random_resolution(self):
        self.assertEqual(resolve_seed(123), 123)
        with patch("vibevoice.script.secrets.randbelow", return_value=321):
            self.assertEqual(resolve_seed(0), 322)
        for value in (None, -1, 2**32, 1.5, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                resolve_seed(value)

    def test_streamer_resets_rng_before_generation(self):
        demo = object.__new__(main.VibeVoiceDemo)
        demo.stop_generation = False
        demo.debug = False
        demo.model = Mock()
        demo.processor = Mock()
        streamer = Mock()
        with patch.object(main, "set_seed") as seed:
            demo._generate_with_streamer({}, 1.6, streamer, seed=567)
        seed.assert_called_once_with(567)
        demo.model.generate.assert_called_once()
        self.assertIsNone(demo.model.generate.call_args.kwargs["max_new_tokens"])

    def test_worker_request_contains_resolved_seed(self):
        demo = object.__new__(main.VibeVoiceDemo)
        demo.worker_process = True
        demo.model_loaded = True
        demo.request_queue = queue.Queue()
        demo.response_queue = queue.Queue()
        demo.response_queue.put(("success", "audio"))
        self.assertEqual(demo._generate_with_worker("Speaker 1: Hi", [], 1.6, 5, True, .95, .95, 0, "", 987), "audio")
        self.assertEqual(demo.request_queue.get()[-1], 987)

    def test_worker_applies_seed_and_completes_request(self):
        requests, responses = queue.Queue(), queue.Queue()
        requests.put(("generate", "Speaker 1: Hi", [], 1.6, 5, True, .95, .95, 0, "", 4321))
        requests.put(("shutdown", None))
        model, processor, streamer = Mock(), Mock(return_value={}), Mock()
        streamer.get_stream.return_value = iter([[0.1, 0.2]])
        with patch("vibevoice.model_loading.load_model_and_processor", return_value=(processor, model, None)), \
             patch("vibevoice.modular.streamer.AudioStreamer", return_value=streamer), \
             patch.object(main, "set_seed") as seed:
            main.model_worker_process(requests, responses, "fake-model", "cpu", 5, None, "sdpa")
        seed.assert_called_once_with(4321)
        self.assertEqual(responses.get()[0], "ready")
        self.assertEqual(responses.get()[0], "success")
        model.generate.assert_called_once()
        self.assertIsNone(model.generate.call_args.kwargs["max_new_tokens"])


class ASRTests(unittest.TestCase):
    def checkpoint(self, root):
        path = Path(root) / "tts" / "VibeVoice-ASR-HF"
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({"model_type": "vibevoice_asr"}))
        for name in ("model.safetensors", "processor_config.json", "tokenizer_config.json", "tokenizer.json", "chat_template.jinja"):
            (path / name).write_text("{}")
        return path

    def test_local_discovery_validates_asr_without_tts_or_downloads(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.checkpoint(root)
            settings = SimpleNamespace(models_dir=Path(root), tts_dir=Path(root) / "tts")
            self.assertEqual(discover_asr_models(settings), {model.name: str(model.resolve())})
            (model / "tokenizer.json").unlink()
            self.assertEqual(discover_asr_models(settings), {})
            with self.assertRaisesRegex(ValueError, "processor asset"):
                validate_asr_model(model)

    def test_worker_file_protocol_and_error_propagation(self):
        with tempfile.TemporaryDirectory() as root:
            model = self.checkpoint(root)
            audio = Path(root) / "upload.wav"
            audio.write_bytes(b"audio")
            payload = {"transcript": "[1] Hello", "segments": [{"speaker": 0, "start": 0., "end": 1., "text": "Hello"}]}
            def worker(command):
                request = json.loads(Path(command[-2]).read_text())
                self.assertEqual(request["context"], "Names")
                Path(command[-1]).write_text(json.dumps(payload))
                return SimpleNamespace(returncode=0)
            with patch("vibevoice.asr.asr_python", return_value=Path("python")), patch("vibevoice.asr.subprocess.run", side_effect=worker):
                self.assertEqual(run_transcription(audio, model, "cpu", "Names"), payload)
                payload = {"error": "out of memory"}
                with self.assertRaisesRegex(RuntimeError, "out of memory"):
                    run_transcription(audio, model, "cpu", "Names")

    def test_segment_schema_and_malformed_fallback(self):
        output = normalize_result([{"Start": 0, "End": 1.5, "Speaker": 1, "Content": "Hello"}], "raw")
        self.assertEqual(output["transcript"], "[2] Hello")
        self.assertEqual(output["segments"], [{"start": 0., "end": 1.5, "speaker": 1, "text": "Hello"}])
        for parsed in ("broken JSON", [{"Start": float("nan"), "End": 1, "Speaker": 0, "Content": "Hi"}], []):
            output = normalize_result(parsed, "raw")
            self.assertEqual(output["transcript"], "raw")
            self.assertEqual(output["segments"], [])
            self.assertTrue(output["warning"])


if __name__ == "__main__":
    unittest.main()
