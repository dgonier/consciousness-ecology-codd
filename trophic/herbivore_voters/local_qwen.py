"""Local Qwen3-4B herbivore voter.

Mirrors the apex LocalQwenVoter but emits a synthesis paragraph instead
of a directional vote. Shares the singleton ModelHost so multiple
herbivore species running on Qwen3-4B don't reload weights.
"""
from __future__ import annotations

import os

import torch

from ..config import DEFAULT_CONFIG
from ..model_host import ModelHost
from .base import HerbivoreVoter, HerbivoreSynthesis
from .api_voters import _parse_synthesis, DEFAULT_HERB_SYSTEM


class LocalQwenHerbivore(HerbivoreVoter):
    def __init__(
        self,
        host: ModelHost | None = None,
        max_new_tokens: int = 256,
        species_template: str | None = None,
    ):
        self.host = host or ModelHost.get(DEFAULT_CONFIG.model)
        self.max_new_tokens = max_new_tokens
        self.species_template = species_template
        self.herb_id = "local_qwen3_4b_herb"

    def is_available(self) -> bool:
        return self.host._model is not None

    def synthesize(self, scenario_text: str) -> HerbivoreSynthesis:
        system = self.species_template or DEFAULT_HERB_SYSTEM
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": scenario_text},
        ]
        tok = self.host._tok
        try:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", enable_thinking=False,
            ).to(self.host.device)
        except TypeError:
            ids = tok.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True,
                return_tensors="pt",
            ).to(self.host.device)
        am = torch.ones_like(ids)
        with torch.no_grad():
            out = self.host._model.generate(
                input_ids=ids,
                attention_mask=am,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new = out[0, ids.shape[1]:]
        text = tok.decode(new, skip_special_tokens=True)
        diet, synth, hint = _parse_synthesis(text)
        return HerbivoreSynthesis(
            herb_id=self.herb_id,
            species_id="?",
            diet_tag=diet,
            synthesis=synth,
            confidence=None,
            raw_text=text,
            provider_meta={"model": "Qwen3-4B (local)"},
            direction_hint=hint,
        )
