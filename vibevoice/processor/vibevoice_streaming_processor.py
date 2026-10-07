# Copyright (c) Microsoft Corporation. Licensed under the MIT License.
"""Prepare single-speaker text using a prefilled realtime voice prompt."""

import json
from pathlib import Path

import torch
from transformers import BatchEncoding

from vibevoice.modular.modular_vibevoice_text_tokenizer import (
    VibeVoiceTextTokenizerFast,
)


class VibeVoiceStreamingProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_path, *, tokenizer_path, **kwargs):
        config_file = Path(model_path) / "preprocessor_config.json"
        config = json.loads(config_file.read_text(encoding="utf-8"))
        if config.get("speech_tok_compress_ratio", 3200) != 3200:
            raise ValueError(
                "Realtime processor requires 24 kHz / 3200-sample acoustic tokens"
            )
        tokenizer = VibeVoiceTextTokenizerFast.from_pretrained(
            str(tokenizer_path), **kwargs
        )
        return cls(tokenizer)

    def process_input_with_cached_prompt(self, text, cached_prompt, **kwargs):
        if not text or not text.strip():
            raise ValueError("Provide text for realtime speech")
        text_ids = self.tokenizer.encode(text.strip() + "\n", add_special_tokens=False)
        if not text_ids:
            raise ValueError("Realtime text produced no tokens")
        lm_length = cached_prompt["lm"]["last_hidden_state"].shape[1]
        tts_length = cached_prompt["tts_lm"]["last_hidden_state"].shape[1]
        return BatchEncoding(
            {
                "input_ids": torch.full(
                    (1, lm_length), self.tokenizer.pad_id, dtype=torch.long
                ),
                "tts_lm_input_ids": torch.full(
                    (1, tts_length), self.tokenizer.pad_id, dtype=torch.long
                ),
                "tts_text_ids": torch.tensor([text_ids], dtype=torch.long),
                "attention_mask": torch.ones(1, lm_length, dtype=torch.long),
                "tts_lm_attention_mask": torch.ones(1, tts_length, dtype=torch.long),
            }
        )
