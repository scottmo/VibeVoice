# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Configuration for the split realtime text and speech backbones."""

import copy
from typing import ClassVar

from transformers import PretrainedConfig, Qwen2Config

from .configuration_vibevoice import (
    VibeVoiceAcousticTokenizerConfig,
    VibeVoiceDiffusionHeadConfig,
)


class VibeVoiceStreamingConfig(PretrainedConfig):
    model_type = "vibevoice_streaming"
    is_composition = True
    sub_configs: ClassVar[dict] = {
        "acoustic_tokenizer_config": VibeVoiceAcousticTokenizerConfig,
        "decoder_config": Qwen2Config,
        "diffusion_head_config": VibeVoiceDiffusionHeadConfig,
    }

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
        tts_backbone_num_hidden_layers=20,
        **kwargs,
    ):
        for name, value in (
            ("acoustic_tokenizer_config", acoustic_tokenizer_config),
            ("decoder_config", decoder_config),
            ("diffusion_head_config", diffusion_head_config),
        ):
            cls = self.sub_configs[name]
            if value is None:
                value = cls()
            elif isinstance(value, dict):
                value = cls(**value)
            elif not isinstance(value, cls):
                raise ValueError(f"Invalid realtime {name}")
            setattr(self, name, value)
        if (
            not 0
            < tts_backbone_num_hidden_layers
            < self.decoder_config.num_hidden_layers
        ):
            raise ValueError("Realtime requires both text and speech backbone layers")
        self.tts_backbone_num_hidden_layers = tts_backbone_num_hidden_layers
        self.acoustic_vae_dim = self.acoustic_tokenizer_config.vae_dim
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    def get_text_config(self, decoder=False):
        return self.decoder_config

    def backbone_config(self, *, tts):
        config = copy.deepcopy(self.decoder_config)
        split = config.num_hidden_layers - self.tts_backbone_num_hidden_layers
        config.num_hidden_layers = self.tts_backbone_num_hidden_layers if tts else split
        config.layer_types = (
            config.layer_types[split:] if tts else config.layer_types[:split]
        )
        return config
