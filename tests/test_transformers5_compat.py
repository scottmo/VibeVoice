import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import BitsAndBytesConfig, Qwen2Config
from transformers.cache_utils import DynamicCache
from unittest.mock import patch

from vibevoice.modular.configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceConfig,
    VibeVoiceDiffusionHeadConfig,
    VibeVoiceSemanticTokenizerConfig,
)
from vibevoice.modular.modeling_vibevoice_inference import (
    VibeVoiceForConditionalGenerationInference,
    _align_negative_cache_for_non_diffusion,
    _refresh_negative_cache_for_speech_start,
)
import vibevoice.modular.modeling_vibevoice_inference as inference
from vibevoice.runtime.model_loading import ModelLoadingSettings, load_model_and_processor
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor


def tiny_config(tie_word_embeddings=False):
    acoustic = VibeVoiceAcousticTokenizerConfig(
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
    )
    semantic = VibeVoiceSemanticTokenizerConfig(
        channels=1,
        vae_dim=2,
        fix_std=0.0,
        std_dist_type="none",
        encoder_n_filters=4,
        encoder_ratios=[2],
        encoder_depths="1-1",
    )
    decoder = Qwen2Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=16,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=tie_word_embeddings,
        use_cache=True,
    )
    diffusion = VibeVoiceDiffusionHeadConfig(
        hidden_size=16,
        head_layers=1,
        head_ffn_ratio=2,
        latent_size=2,
        speech_vae_dim=2,
        ddpm_num_steps=10,
        ddpm_num_inference_steps=1,
        ddpm_batch_mul=1,
    )
    return VibeVoiceConfig(
        acoustic_tokenizer_config=acoustic,
        semantic_tokenizer_config=semantic,
        decoder_config=decoder,
        diffusion_head_config=diffusion,
        dtype="float32",
    )


class Transformers5CompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "Q8 checkpoint regression requires CUDA")
    def test_legacy_q8_checkpoint_keeps_speech_modules_unquantized(self):
        import bitsandbytes as bnb

        # Match the real Q8 checkpoint: only LM layers and lm_head have INT8
        # weights/SCB; all five excluded speech components have BF16 weights.
        skips = [
            "prediction_head", "acoustic_connector", "semantic_connector",
            "acoustic_tokenizer", "semantic_tokenizer",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_dir, q8_dir = root / "original", root / "q8"
            original = VibeVoiceForConditionalGenerationInference(tiny_config()).eval()
            original.model.speech_scaling_factor.fill_(1.0)
            original.model.speech_bias_factor.zero_()
            original.save_pretrained(original_dir)
            quantized = VibeVoiceForConditionalGenerationInference.from_pretrained(
                original_dir, local_files_only=True, device_map="cuda", dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_8bit=True, llm_int8_threshold=3.0,
                    llm_int8_skip_modules=[f"model.{name}" for name in skips],
                ),
            ).eval()
            quantized.save_pretrained(q8_dir)
            expected_speech = {
                name: tensor.detach().cpu().clone()
                for name, tensor in quantized.state_dict().items()
                if any(name.startswith(f"model.{skip}.") for skip in skips)
            }
            config_file = q8_dir / "config.json"
            serialized = json.loads(config_file.read_text())
            serialized["quantization_config"]["llm_int8_skip_modules"] = skips
            config_file.write_text(json.dumps(serialized), encoding="utf-8")
            del quantized
            settings = ModelLoadingSettings(root)
            with (
                patch("vibevoice.runtime.model_loading.resolve_tokenizer_path", return_value=q8_dir),
                patch.object(VibeVoiceProcessor, "from_pretrained", return_value=object()),
            ):
                _, loaded, _ = load_model_and_processor(
                    str(q8_dir), settings, device="cuda", attn_implementation="sdpa",
                )

            # Exercise the reported failing path as well as the quantized LM
            # and output head, using synthetic voice input and no large model.
            with torch.no_grad():
                output = loaded(
                    input_ids=torch.tensor([[1, 7, 8, 9, 10]], device="cuda"),
                    attention_mask=torch.ones(1, 5, dtype=torch.long, device="cuda"),
                    speech_tensors=torch.zeros(1, 8, dtype=torch.bfloat16, device="cuda"),
                    speech_masks=torch.ones(1, 4, dtype=torch.bool, device="cuda"),
                    speech_input_mask=torch.tensor([[False, True, True, True, True]], device="cuda"),
                )
            self.assertTrue(torch.isfinite(output.logits).all())
            self.assertIsInstance(loaded.lm_head, bnb.nn.Linear8bitLt)
            self.assertIsInstance(loaded.lm_head.weight, bnb.nn.Int8Params)
            self.assertIsInstance(loaded.model.language_model.layers[0].self_attn.q_proj.weight, bnb.nn.Int8Params)
            for skip in skips:
                self.assertFalse(any(
                    isinstance(module, bnb.nn.Linear8bitLt)
                    for module in loaded.get_submodule(f"model.{skip}").modules()
                ))
            for name, expected in expected_speech.items():
                torch.testing.assert_close(loaded.state_dict()[name].cpu(), expected, rtol=0, atol=0)
            self.assertEqual(json.loads(config_file.read_text()), serialized)

    def test_tiny_tts_checkpoint_constructs_and_round_trips(self):
        model = VibeVoiceForConditionalGenerationInference(tiny_config()).eval()
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            loaded = VibeVoiceForConditionalGenerationInference.from_pretrained(
                directory, local_files_only=True, dtype=torch.float32,
            ).eval()
        self.assertEqual(loaded.config.dtype, torch.float32)
        self.assertEqual(loaded.config.decoder_config.vocab_size, 16)
        self.assertEqual(loaded.lm_head.weight.shape, (16, 16))

    def test_tied_and_untied_embedding_checkpoint_round_trips(self):
        for tied in (False, True):
            with self.subTest(tied=tied), tempfile.TemporaryDirectory() as directory:
                config = tiny_config(tie_word_embeddings=tied)
                self.assertEqual(config.tie_word_embeddings, tied)
                model = VibeVoiceForConditionalGenerationInference(config).eval()
                input_weight = model.model.language_model.embed_tokens.weight
                output_weight = model.lm_head.weight
                with torch.no_grad():
                    input_values = torch.arange(input_weight.numel(), dtype=input_weight.dtype).reshape_as(input_weight)
                    input_weight.copy_(input_values)
                    if tied:
                        output_values = input_values
                    else:
                        output_values = input_values + 1000
                        output_weight.copy_(output_values)
                expected_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }

                model.save_pretrained(directory, safe_serialization=True)
                loaded = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    directory, local_files_only=True, dtype=torch.float32,
                ).eval()
                loaded_input = loaded.model.language_model.embed_tokens.weight
                loaded_output = loaded.lm_head.weight
                for name, expected in expected_state.items():
                    torch.testing.assert_close(
                        loaded.state_dict()[name], expected, rtol=0, atol=0, equal_nan=True,
                        msg=lambda message: f"State tensor {name} did not round-trip: {message}",
                    )
                self.assertTrue(torch.equal(loaded_input, input_values))
                self.assertTrue(torch.equal(loaded_output, output_values))
                self.assertEqual(loaded_input.data_ptr() == loaded_output.data_ptr(), tied)

    def test_negative_cache_mutations_support_transformers5_layers(self):
        keys = torch.arange(8, dtype=torch.float32).view(2, 1, 4, 1)
        values = keys + 20
        cache = DynamicCache()
        cache.update(keys.clone(), values.clone(), 0)
        original_keys, original_values = keys.clone(), values.clone()
        attention_mask = torch.ones(2, 4, dtype=torch.long)
        input_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        kwargs = {"past_key_values": cache, "attention_mask": attention_mask}

        _refresh_negative_cache_for_speech_start(
            kwargs, input_ids, torch.tensor([1]), speech_start_id=9,
        )
        self.assertTrue(torch.equal(cache.layers[0].keys[1, :, -1, :], original_keys[1, :, 0, :]))
        self.assertTrue(torch.equal(cache.layers[0].values[1, :, -1, :], original_values[1, :, 0, :]))
        self.assertEqual(input_ids[1, -1].item(), 9)
        self.assertTrue(torch.equal(attention_mask[1], torch.tensor([0, 0, 0, 1])))

        _align_negative_cache_for_non_diffusion(
            kwargs, input_ids, torch.tensor([0]), torch.tensor([1]),
        )
        self.assertTrue(torch.equal(cache.layers[0].keys[0, :, 2:, :], original_keys[0, :, 1:-1, :]))
        self.assertTrue(torch.equal(cache.layers[0].values[0, :, 2:, :], original_values[0, :, 1:-1, :]))
        self.assertTrue(torch.equal(attention_mask[0], torch.tensor([1, 0, 1, 1])))

    def test_tiny_generation_runs_speech_diffusion_with_both_negative_cache_modes(self):
        tokenizer = SimpleNamespace(
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            speech_start_id=4,
            speech_diffusion_id=5,
            speech_end_id=6,
        )
        input_ids = torch.tensor([[1, 7, 8, 9, 10], [1, 7, 8, 9, 10]])
        speech_tensors = torch.zeros(2, 8)

        for refresh_negative in (True, False):
            with self.subTest(refresh_negative=refresh_negative):
                model = VibeVoiceForConditionalGenerationInference(tiny_config()).eval()
                model.set_ddpm_inference_steps(1)
                model.model.speech_scaling_factor.fill_(1.0)
                model.model.speech_bias_factor.zero_()
                calls = 0
                if refresh_negative:
                    targets = [
                        [5, 4], [4, 4], [5, 4], [4, 4], [6, 5], [4, 4],
                        [4, 6], [5, 2], [4, 4], [6, 2], [2, 2],
                    ]
                else:
                    targets = [
                        [5, 4], [4, 4], [5, 4], [4, 4], [6, 5], [4, 4],
                        [4, 6], [4, 4], [5, 2], [4, 4], [6, 2], [4, 4],
                        [2, 2], [4, 4],
                    ]

                def forced_logits(hidden):
                    nonlocal calls
                    target_ids = targets[min(calls, len(targets) - 1)]
                    calls += 1
                    logits = torch.full(
                        (*hidden.shape[:-1], model.config.decoder_config.vocab_size),
                        -100.0,
                        dtype=hidden.dtype,
                        device=hidden.device,
                    )
                    for sample_idx, target_id in enumerate(target_ids):
                        logits[sample_idx, ..., target_id] = 100.0
                    return logits

                model.lm_head.forward = forced_logits
                cache_mutations = []
                original_iter = inference._iter_cache_key_value_tensors

                def track_cache_mutations(cache):
                    for keys, values in original_iter(cache):
                        before_keys, before_values = keys.clone(), values.clone()
                        yield keys, values
                        cache_mutations.append(
                            not torch.equal(before_keys, keys) or not torch.equal(before_values, values)
                        )

                with patch.object(inference, "_iter_cache_key_value_tensors", side_effect=track_cache_mutations):
                    result = model.generate(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        speech_tensors=speech_tensors,
                        speech_masks=torch.ones(2, 4, dtype=torch.bool),
                        speech_input_mask=torch.tensor([[False, True, True, True, True]] * 2),
                        tokenizer=tokenizer,
                        max_new_tokens=8,
                        max_length_times=2,
                        refresh_negative=refresh_negative,
                        show_progress_bar=False,
                        return_speech=False,
                    )
                self.assertEqual(result.sequences[0, 5:12].tolist(), [5, 5, 6, 4, 5, 6, 2])
                self.assertTrue(cache_mutations and any(cache_mutations))
                self.assertGreaterEqual(calls, 4)


if __name__ == "__main__":
    unittest.main()
