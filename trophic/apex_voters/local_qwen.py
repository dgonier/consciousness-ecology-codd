"""Local Qwen3-4B as an apex voter, with token-level perplexity.

Reads the EvidencePacket, runs Qwen with the benchmark XML prompt, and
extracts:
  - direction from the parsed XML
  - confidence from the model's emitted <confidence> tag
  - perplexity = mean negative log-prob over the answer's direction token
    (computed from the same forward, no extra cost)

The perplexity signal is what the aggregator uses to weight votes:
when the model is uncertain about its own direction emission (high
perplexity / low logprob on the chosen token), down-weight its vote.
"""
from __future__ import annotations

import math

import torch

from ..config import DEFAULT_CONFIG
from ..model_host import ModelHost
from ..training.xml_schema import parse_prediction
from .base import ApexVoter, EvidencePacket, VoterResponse
from .parsing import extract_reasoning, extract_citations


class LocalQwenVoter(ApexVoter):
    voter_id = "local_qwen3_4b"

    def __init__(self, host: ModelHost | None = None, max_new_tokens: int = 96):
        self.host = host or ModelHost.get(DEFAULT_CONFIG.model)
        self.max_new_tokens = max_new_tokens

    def is_available(self) -> bool:
        return self.host._model is not None

    def vote(self, evidence: EvidencePacket) -> VoterResponse:
        # Split the rendered text into system + user using the SYSTEM
        # block as a separator. evidence.text is "<SYSTEM>\n\n<rest>".
        # For the chat template we want to pass them as separate messages.
        text = evidence.text
        # The first "blank line + content" after SYSTEM is the user body.
        # SYSTEM is the first ~5 lines until the first blank-line separator.
        from .evidence import SYSTEM
        if text.startswith(SYSTEM):
            user_body = text[len(SYSTEM):].lstrip()
            system = SYSTEM
        else:
            system = SYSTEM
            user_body = text

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_body},
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
                return_dict_in_generate=True,
                output_scores=True,
            )
        sequences = out.sequences
        new_ids = sequences[0, ids.shape[1]:]
        raw_text = tok.decode(new_ids, skip_special_tokens=True)
        parsed = parse_prediction(raw_text)
        direction = parsed.direction
        confidence = parsed.confidence

        # Compute log-perplexity over the direction-token region. We find
        # the index of the direction token in the generated sequence (the
        # token whose decode is "up" or "down") and take its log-prob.
        # If we can't find it, fall back to mean perplexity over all
        # generated tokens.
        scores = out.scores  # tuple of [vocab] tensors per generated step
        target_word = direction or ""
        chosen_logprob = None
        if target_word and scores:
            target_ids = set()
            for s in (target_word, " " + target_word, target_word.upper(), " " + target_word.upper()):
                enc = tok(s, add_special_tokens=False).input_ids
                if enc:
                    target_ids.add(enc[0])
            for step_idx, step_logits in enumerate(scores):
                logits = step_logits[0]  # [vocab]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                best = int(logits.argmax().item())
                if best in target_ids:
                    chosen_logprob = float(logprobs[best].item())
                    break
        if chosen_logprob is None and scores:
            # Mean log-prob of the actually-generated tokens
            total = 0.0
            n = 0
            for step_idx, step_logits in enumerate(scores):
                logits = step_logits[0]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                tok_id = int(new_ids[step_idx].item()) if step_idx < new_ids.numel() else None
                if tok_id is None:
                    break
                total += float(logprobs[tok_id].item())
                n += 1
            chosen_logprob = total / max(n, 1) if n > 0 else 0.0
        # Perplexity = exp(-mean_logprob). Lower = more confident.
        perplexity = math.exp(-chosen_logprob) if chosen_logprob is not None else float("inf")

        reasoning = extract_reasoning(raw_text)
        citations = extract_citations(reasoning)
        return VoterResponse(
            voter_id=self.voter_id,
            direction=direction,
            confidence=confidence,
            perplexity=perplexity,
            raw_text=raw_text,
            reasoning=reasoning,
            evidence_citations=citations,
            provider_meta={"model": "Qwen3-4B (local)", "max_new_tokens": self.max_new_tokens},
        )
