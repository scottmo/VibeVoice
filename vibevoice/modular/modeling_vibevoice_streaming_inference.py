# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Adapted from VibeVoice revision 1541f590c7099820f10ea012f48d2399282df69f.

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from tqdm import tqdm
from transformers.generation import (
    GenerationMixin,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.auto import AutoModelForCausalLM

from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .modeling_vibevoice_streaming import (
    BinaryClassifier,
    VibeVoiceStreamingModel,
    VibeVoiceStreamingPreTrainedModel,
)
from .modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
from .streamer import AsyncAudioStreamer, AudioStreamer

TTS_TEXT_WINDOW_SIZE = 5
TTS_SPEECH_WINDOW_SIZE = 6


def _update_model_kwargs_for_generation(
    outputs: ModelOutput, model_kwargs: dict[str, Any], num_new_tokens: int = 1
) -> dict[str, Any]:
    """
    Update model_kwargs after adding new tokens (supports multi-token windows).

    Updates past_key_values, attention_mask, and cache_position for the next forward pass.
    """
    model_kwargs["past_key_values"] = outputs.past_key_values
    attention_mask = model_kwargs["attention_mask"]
    model_kwargs["attention_mask"] = torch.cat(
        [
            attention_mask,
            attention_mask.new_ones((attention_mask.shape[0], num_new_tokens)),
        ],
        dim=-1,
    )
    cache_pos = model_kwargs["cache_position"]
    model_kwargs["cache_position"] = torch.arange(
        cache_pos[-1] + 1, cache_pos[-1] + num_new_tokens + 1, device=cache_pos.device
    )
    return model_kwargs


@dataclass
class VibeVoiceCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: torch.FloatTensor | None = None


@dataclass
class VibeVoiceGenerationOutput(ModelOutput):
    """Output type for VibeVoice generation."""

    sequences: torch.LongTensor = None
    speech_outputs: list[torch.FloatTensor] | None = None
    reach_max_step_sample: torch.BoolTensor | None = None


class VibeVoiceStreamingForConditionalGenerationInference(
    VibeVoiceStreamingPreTrainedModel, GenerationMixin
):
    def __init__(self, config):
        super().__init__(config)
        self.model = VibeVoiceStreamingModel(config)
        self.tts_eos_classifier = BinaryClassifier(config.decoder_config.hidden_size)
        self.ddpm_inference_steps = (
            config.diffusion_head_config.ddpm_num_inference_steps
        )
        self.post_init()

    @property
    def noise_scheduler(self):
        return self.model.noise_scheduler

    @property
    def prediction_head(self):
        return self.model.prediction_head

    @property
    def speech_scaling_factor(self):
        return self.model.speech_scaling_factor

    @property
    def speech_bias_factor(self):
        return self.model.speech_bias_factor

    @property
    def acoustic_tokenizer(self):
        return self.model.acoustic_tokenizer

    @property
    def acoustic_connector(self):
        return self.model.acoustic_connector

    def tie_weights(self, missing_keys=None, recompute_mapping=True):
        return super().tie_weights(
            missing_keys=missing_keys, recompute_mapping=recompute_mapping
        )

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        """
        This model does not define an `lm_head` (vocabulary projection).
        """
        return

    def set_output_embeddings(self, new_embeddings):
        """
        No-op because there is no `lm_head`. Provided only to satisfy optional API calls.
        To enable, first create `self.lm_head` then allow assignment.
        """
        raise RuntimeError(
            "Output embeddings (lm_head) are not defined for this model. Create one before calling set_output_embeddings if needed."
        )

    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer)

    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = (
            num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        """Select the next query window and its positions from a native KV cache."""
        model_inputs = {"cache_position": cache_position}
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None and input_ids.shape[1] == 0:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0] :]
            elif inputs_embeds is not None or (
                cache_position is not None and cache_position[-1] >= input_ids.shape[1]
            ):
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif (
                cache_position is not None
                and input_ids.shape[1] != cache_position.shape[0]
            ):
                input_ids = input_ids[:, cache_position]
        use_embeds = inputs_embeds is not None and (
            past_key_values is None
            or (
                cache_position is not None
                and len(cache_position) == inputs_embeds.shape[1]
            )
        )
        if use_embeds:
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = inputs_embeds
        else:
            model_inputs["input_ids"] = (
                input_ids.clone(memory_format=torch.contiguous_format)
                if input_ids is not None
                else None
            )
            model_inputs["inputs_embeds"] = None
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
        if attention_mask is not None and kwargs.get("position_ids") is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids
        if kwargs.get("position_ids") is not None:
            if past_key_values is not None:
                seq_len = (
                    model_inputs["inputs_embeds"].shape[1]
                    if model_inputs.get("inputs_embeds") is not None
                    else model_inputs["input_ids"].shape[1]
                )
                model_inputs["position_ids"] = kwargs["position_ids"][
                    :, -seq_len:
                ].clone(memory_format=torch.contiguous_format)
            else:
                model_inputs["position_ids"] = kwargs.pop("position_ids").clone(
                    memory_format=torch.contiguous_format
                )
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value
        model_inputs.pop("labels", None)
        return model_inputs

    def _update_model_kwargs_for_generation(
        self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
    ):
        return _update_model_kwargs_for_generation(
            outputs, model_kwargs, num_new_tokens
        )

    def forward_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPast:
        """Single pass of the base text LM."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        if labels is not None:
            raise NotImplementedError(
                "Loss computation is not implemented in this version."
            )
        return BaseModelOutputWithPast(
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def forward_tts_lm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: tuple[tuple[torch.FloatTensor]] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        lm_last_hidden_state: torch.FloatTensor | None = None,
        tts_text_masks: torch.BoolTensor | None = None,
        **kwargs,
    ) -> tuple | VibeVoiceCausalLMOutputWithPast:
        """Single pass of the TTS LM."""
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        start_idx = inputs_embeds.shape[1] - lm_last_hidden_state.shape[1]
        inputs_embeds[:, start_idx:, :] = lm_last_hidden_state
        inputs_embeds = inputs_embeds + self.model.tts_input_types(
            tts_text_masks.long()
        )
        outputs = self.model.tts_language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        logits = self.tts_eos_classifier(hidden_states[:, -1, :])
        if labels is not None:
            raise NotImplementedError(
                "Loss computation is not implemented in this version."
            )
        return VibeVoiceCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def forward(self, *args, **kwargs):
        """Unified forward is intentionally disabled."""
        raise RuntimeError(
            "Unified forward is disabled. Use `forward_lm`, `forward_tts_lm`, or `generate` instead."
        )

    def _generation_inputs(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        return input_ids, {
            "attention_mask": attention_mask.to(self.device),
            "cache_position": torch.arange(input_ids.shape[1], device=self.device),
            "past_key_values": None,
            "use_cache": True,
        }

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids,
        attention_mask,
        tts_lm_input_ids,
        tts_lm_attention_mask,
        tts_text_ids,
        tokenizer,
        all_prefilled_outputs,
        max_new_tokens=None,
        audio_streamer: AudioStreamer | AsyncAudioStreamer | None = None,
        return_speech=True,
        cfg_scale=1.5,
        stop_check_fn: Callable[[], bool] | None = None,
        show_progress_bar=True,
        verbose=False,
    ) -> VibeVoiceGenerationOutput:
        """Encode a complete script in text windows and stream speech latents."""
        if input_ids.shape[0] != 1 or tts_text_ids.shape[1] == 0:
            raise ValueError("Realtime TTS requires one nonempty single-speaker script")
        tts_text_ids = tts_text_ids.to(self.device)
        if max_new_tokens is None:
            max_new_tokens = (
                self.config.decoder_config.max_position_embeddings
                - tts_lm_input_ids.shape[1]
            )
        if max_new_tokens < 1:
            raise ValueError("Realtime generation requires a positive token budget")
        max_length = tts_lm_input_ids.shape[1] + max_new_tokens
        input_ids, model_kwargs = self._generation_inputs(input_ids, attention_mask)
        tts_lm_input_ids, tts_lm_model_kwargs = self._generation_inputs(
            tts_lm_input_ids, tts_lm_attention_mask
        )
        negative_length = all_prefilled_outputs["neg_tts_lm"].last_hidden_state.shape[1]
        negative_ids = torch.full(
            (1, negative_length),
            tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            dtype=torch.long,
            device=self.device,
        )
        tts_lm_negative_input_ids, tts_lm_negative_model_kwargs = (
            self._generation_inputs(negative_ids, torch.ones_like(negative_ids))
        )
        acoustic_cache = VibeVoiceTokenizerStreamingCache()
        batch_size = input_ids.shape[0]
        if batch_size != 1:
            raise ValueError("Realtime TTS supports one speaker and one request")
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        audio_chunks = [[] for _ in range(batch_size)]
        tts_text_window_index = 0
        reach_max_step_sample = torch.zeros(batch_size, dtype=torch.bool, device=device)
        first_text_window_size = min(TTS_TEXT_WINDOW_SIZE, tts_text_ids.shape[1])
        outputs = all_prefilled_outputs["lm"]
        tts_lm_outputs = all_prefilled_outputs["tts_lm"]
        tts_lm_negative_outputs = all_prefilled_outputs["neg_tts_lm"]
        model_kwargs = _update_model_kwargs_for_generation(
            outputs, model_kwargs, num_new_tokens=first_text_window_size
        )
        tts_lm_model_kwargs = _update_model_kwargs_for_generation(
            tts_lm_outputs, tts_lm_model_kwargs, num_new_tokens=first_text_window_size
        )
        tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
            tts_lm_negative_outputs,
            tts_lm_negative_model_kwargs,
            is_encoder_decoder=False,
        )
        step = tts_lm_input_ids.shape[1]
        total_generated_speech_tokens = 0
        total_prefilled_text_tokens = 0
        if show_progress_bar:
            progress_bar = tqdm(
                total=max_length,
                desc=f"Prefilled {step} tokens, current step ({step} / {max_length})",
                initial=step,
                leave=False,
            )
        else:
            progress_bar = None
        while True:
            if stop_check_fn is not None and stop_check_fn():
                if verbose:
                    print(f"Generation stopped externally at step {step + 1}")
                if audio_streamer is not None:
                    audio_streamer.end()
                break
            if finished_tags.all():
                if hasattr(progress_bar, "set_description"):
                    progress_bar.set_description("Generation complete")
                break
            cur_input_tts_text_ids = tts_text_ids[
                :,
                tts_text_window_index * TTS_TEXT_WINDOW_SIZE : (
                    tts_text_window_index + 1
                )
                * TTS_TEXT_WINDOW_SIZE,
            ]
            next_text_window_size = tts_text_ids[
                :,
                (tts_text_window_index + 1) * TTS_TEXT_WINDOW_SIZE : (
                    tts_text_window_index + 2
                )
                * TTS_TEXT_WINDOW_SIZE,
            ].shape[1]
            tts_text_window_index += 1
            if cur_input_tts_text_ids.shape[1] > 0:
                if (
                    tts_lm_input_ids.shape[1] + cur_input_tts_text_ids.shape[1]
                    > max_length
                ):
                    reach_max_step_sample[~finished_tags] = True
                    break
                input_ids = torch.cat([input_ids, cur_input_tts_text_ids], dim=-1)
                tts_lm_input_ids = torch.cat(
                    [tts_lm_input_ids, cur_input_tts_text_ids], dim=-1
                )
                step += cur_input_tts_text_ids.shape[1]
                total_prefilled_text_tokens += cur_input_tts_text_ids.shape[1]
                if progress_bar is not None:
                    progress_bar.update(cur_input_tts_text_ids.shape[1])
                    progress_bar.set_description(
                        f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {max_length})"
                    )
                model_inputs = self.prepare_inputs_for_generation(
                    input_ids, **model_kwargs
                )
                outputs = self.forward_lm(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                model_kwargs = _update_model_kwargs_for_generation(
                    outputs, model_kwargs, num_new_tokens=next_text_window_size
                )
                tts_lm_model_inputs = self.prepare_inputs_for_generation(
                    tts_lm_input_ids, **tts_lm_model_kwargs
                )
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.ones_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": outputs.last_hidden_state,
                }
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs,
                    **tts_lm_additional_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False
                )
            diffusion_indices = torch.LongTensor([0])
            for cur_speech_index in range(TTS_SPEECH_WINDOW_SIZE):
                if finished_tags.all() or (
                    stop_check_fn is not None and stop_check_fn()
                ):
                    break
                if tts_lm_input_ids.shape[1] >= max_length:
                    reach_max_step_sample[~finished_tags] = True
                    break
                positive_condition = tts_lm_outputs.last_hidden_state[
                    diffusion_indices, -1, :
                ]
                negative_condition = tts_lm_negative_outputs.last_hidden_state[
                    diffusion_indices, -1, :
                ]
                speech_latent = self.sample_speech_tokens(
                    positive_condition, negative_condition, cfg_scale=cfg_scale
                ).unsqueeze(1)
                scaled_latent = speech_latent / self.model.speech_scaling_factor.to(
                    speech_latent.device
                ) - self.model.speech_bias_factor.to(speech_latent.device)
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent.to(self.model.acoustic_tokenizer.device),
                    cache=acoustic_cache,
                    sample_indices=diffusion_indices.to(
                        self.model.acoustic_tokenizer.device
                    ),
                    use_cache=True,
                    debug=False,
                )
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    if return_speech and (not finished_tags[idx]):
                        audio_chunks[idx].append(audio_chunk[i])
                if audio_streamer is not None:
                    audio_streamer.put(audio_chunk, diffusion_indices)
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                tts_lm_input_ids = torch.cat(
                    [tts_lm_input_ids, torch.ones_like(tts_lm_input_ids[:, -1:])],
                    dim=-1,
                )
                step += 1
                total_generated_speech_tokens += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Prefilled {total_prefilled_text_tokens} text tokens, generated {total_generated_speech_tokens} speech tokens, current step ({step} / {max_length})"
                    )
                tts_lm_model_inputs = self.prepare_inputs_for_generation(
                    tts_lm_input_ids, **tts_lm_model_kwargs
                )
                tts_lm_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(tts_lm_input_ids[:, -1:]),
                    "lm_last_hidden_state": acoustic_embed,
                }
                tts_lm_outputs = self.forward_tts_lm(
                    **tts_lm_model_inputs,
                    **tts_lm_additional_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                if (
                    cur_speech_index == TTS_SPEECH_WINDOW_SIZE - 1
                    and next_text_window_size > 0
                ):
                    tts_lm_model_kwargs = _update_model_kwargs_for_generation(
                        tts_lm_outputs,
                        tts_lm_model_kwargs,
                        num_new_tokens=next_text_window_size,
                    )
                else:
                    tts_lm_model_kwargs = self._update_model_kwargs_for_generation(
                        tts_lm_outputs, tts_lm_model_kwargs, is_encoder_decoder=False
                    )
                tts_lm_negative_input_ids = torch.cat(
                    [
                        tts_lm_negative_input_ids,
                        torch.ones_like(tts_lm_input_ids[:, -1:]),
                    ],
                    dim=-1,
                )
                tts_lm_negative_model_inputs = self.prepare_inputs_for_generation(
                    tts_lm_negative_input_ids, **tts_lm_negative_model_kwargs
                )
                tts_lm_negative_additional_inputs = {
                    "tts_text_masks": torch.zeros_like(
                        tts_lm_negative_input_ids[:, -1:]
                    ),
                    "lm_last_hidden_state": acoustic_embed,
                }
                tts_lm_negative_outputs = self.forward_tts_lm(
                    **tts_lm_negative_model_inputs,
                    **tts_lm_negative_additional_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                tts_lm_negative_model_kwargs = self._update_model_kwargs_for_generation(
                    tts_lm_negative_outputs,
                    tts_lm_negative_model_kwargs,
                    is_encoder_decoder=False,
                )
                tts_eos_logits = torch.sigmoid(
                    self.tts_eos_classifier(
                        tts_lm_outputs.last_hidden_state[diffusion_indices, -1, :]
                    )
                )
                if tts_eos_logits[0].item() > 0.5:
                    finished_tags[diffusion_indices] = True
                    if audio_streamer is not None:
                        audio_streamer.end(diffusion_indices)
            if reach_max_step_sample.any() or tts_lm_input_ids.shape[1] >= max_length:
                if verbose:
                    print(
                        f"Reached maximum generation length {max_length}, stopped it."
                    )
                reached_samples = torch.arange(batch_size, device=device)[
                    ~finished_tags
                ]
                if reached_samples.numel() > 0:
                    reach_max_step_sample[reached_samples] = True
                break
        if audio_streamer is not None:
            audio_streamer.end()
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                final_audio_outputs.append(None)
        if reach_max_step_sample is not None and reach_max_step_sample.any():
            print(f"Reached maximum generation length {max_length}, stopped it.")
        if progress_bar is not None:
            progress_bar.close()
        return VibeVoiceGenerationOutput(
            sequences=tts_lm_input_ids,
            speech_outputs=final_audio_outputs if return_speech else None,
            reach_max_step_sample=reach_max_step_sample,
        )

    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0).to(
            self.model.prediction_head.device
        )
        speech = torch.randn(condition.shape[0], self.config.acoustic_vae_dim).to(
            condition
        )
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(
                combined, t.repeat(combined.shape[0]).to(combined), condition=condition
            )
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]


AutoModelForCausalLM.register(
    VibeVoiceStreamingConfig, VibeVoiceStreamingForConditionalGenerationInference
)
__all__ = ["VibeVoiceStreamingForConditionalGenerationInference"]
