# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Adapted from VibeVoice revision 1541f590c7099820f10ea012f48d2399282df69f.

import torch
from torch import nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto import AutoModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from vibevoice.schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibevoice_streaming import VibeVoiceStreamingConfig
from .modular_vibevoice_diffusion_head import VibeVoiceDiffusionHead


class BinaryClassifier(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-06)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class VibeVoiceStreamingPreTrainedModel(PreTrainedModel):
    config_class = VibeVoiceStreamingConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        if isinstance(module, VibeVoiceDiffusionHead):
            module.initialize_weights()
            return
        if hasattr(self.config, "language_model_config") and hasattr(
            self.config.language_model_config, "initializer_range"
        ):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, "decoder_config") and hasattr(
            self.config.decoder_config, "initializer_range"
        ):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class VibeVoiceStreamingModel(VibeVoiceStreamingPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if hasattr(config, "dtype") and config.dtype is not None:
            if isinstance(config.dtype, str):
                dtype = getattr(torch, config.dtype)
            else:
                dtype = config.dtype
        else:
            dtype = torch.float32
        lm_config = config.backbone_config(tts=False)
        self.language_model = AutoModel.from_config(lm_config)
        self.language_model.norm = nn.Identity()
        tts_lm_config = config.backbone_config(tts=True)
        self.tts_language_model = AutoModel.from_config(tts_lm_config)
        self.tts_input_types = nn.Embedding(
            num_embeddings=2, embedding_dim=config.decoder_config.hidden_size
        )
        self.acoustic_tokenizer = AutoModel.from_config(
            config.acoustic_tokenizer_config
        ).to(dtype)
        # Cached prompts replace audio encoding in realtime inference.
        self.acoustic_tokenizer.encoder = None
        self.acoustic_connector = SpeechConnector(
            config.acoustic_vae_dim, lm_config.hidden_size
        ).to(dtype)
        self.register_buffer("speech_scaling_factor", torch.tensor(float("nan")))
        self.register_buffer("speech_bias_factor", torch.tensor(float("nan")))
        self.prediction_head = AutoModel.from_config(config.diffusion_head_config).to(
            dtype
        )
        with torch.device("cpu"):
            self.noise_scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
                beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
                prediction_type=config.diffusion_head_config.prediction_type,
            )

    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def set_speech_tokenizers(self, acoustic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()

    def forward(self, *args, **kwargs):
        """Intentionally not implemented."""
        raise RuntimeError(
            "VibeVoiceStreamingModel.forward is intentionally disabled. Use `model.language_model(...)` or `model.tts_language_model(...)` instead."
        )


AutoModel.register(VibeVoiceStreamingConfig, VibeVoiceStreamingModel)
__all__ = ["VibeVoiceStreamingModel", "VibeVoiceStreamingPreTrainedModel"]
