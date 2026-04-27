"""Single-process Qwen3-4B host. Hidden-state-first.

The trophic system is hidden-state-native. Tokens appear at exactly two
boundaries: (a) the very first forward pass that converts a raw text input
into hidden states, and (b) the optional decode-for-legibility step at the
end of an agent's forward pass. Everything in between is vectors.

Three primary methods:
  - text_to_hidden(text)
        forward Qwen on `text`, return the *full sequence* of last-layer
        hidden states (not pooled). Used by producers.
  - encode_role_prefix(text)
        like text_to_hidden but tokenized as a system prompt — gives an
        agent its frozen role-prefix vectors, computed once per agent kind.
  - forward_with_prefix(role_prefix, channel_output, query_text=None,
                        decode=True)
        run Qwen with input embeddings = [role_prefix ⊕ channel_output ⊕
        embedded(query_text)] and return (last_hidden_pooled, decoded_text
        or None). The pooled hidden state is the agent's broadcast.

Mock paths return deterministic stand-in tensors so the system runs
without GPU/weights. Channels still get exercised end-to-end in mock mode
because they're real torch ops on real tensors.
"""
from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import ModelConfig, DEFAULT_CONFIG


@dataclass
class ForwardResult:
    last_hidden: torch.Tensor    # [hidden_size] pooled, on CPU, fp32
    decoded_text: str | None
    null_logit: float | None = None  # set by callers that want to surface gate state


class ModelHost:
    _instance: Optional["ModelHost"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or DEFAULT_CONFIG.model
        self._tok = None
        self._model = None
        self._embed_layer = None
        self._hidden_size: int = 2560  # Qwen3-4B; corrected on real load
        self._device = "cpu"
        if not self.cfg.mock:
            self._load()
        else:
            self._device = "cpu"

    @classmethod
    def get(cls, cfg: ModelConfig | None = None) -> "ModelHost":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the cached singleton — used in tests when switching configs."""
        with cls._lock:
            cls._instance = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.cfg.dtype, torch.float16)
        self._tok = AutoTokenizer.from_pretrained(self.cfg.chat_model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.cfg.chat_model_id,
            torch_dtype=dtype,
            device_map=self.cfg.device,
        )
        self._model.eval()
        self._hidden_size = self._model.config.hidden_size
        self._device = next(self._model.parameters()).device
        # Cache embed layer for tokenless prefix construction.
        self._embed_layer = self._model.get_input_embeddings()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def device(self) -> torch.device | str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if self.cfg.mock:
            return torch.float32
        return next(self._model.parameters()).dtype

    def input_embedding_norm(self) -> float:
        """Mean L2 norm of input-embedding rows. Used to scale Channel output
        to live in the same range Qwen expects to see at its input layer.
        """
        if self.cfg.mock:
            return 1.0
        with torch.no_grad():
            w = self._embed_layer.weight.detach().float()
            n = w.norm(dim=-1).mean().item()
        return float(n)

    # ------------------------------------------------------------------
    # text → hidden states
    # ------------------------------------------------------------------

    def text_to_hidden(self, text: str, pool: str | None = None) -> torch.Tensor:
        """Forward `text` through Qwen, return last-layer hidden states.

        pool=None  → full sequence  [seq_len, hidden_size]
        pool="mean" → [hidden_size]
        pool="last" → [hidden_size]
        """
        if self.cfg.mock:
            return self._mock_text_hidden(text, pool)

        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True, use_cache=False)
        seq = out.hidden_states[-1][0]  # [seq, hidden]
        seq = seq.float().cpu()
        if pool is None:
            return seq
        if pool == "mean":
            return seq.mean(dim=0)
        if pool == "last":
            return seq[-1]
        raise ValueError(pool)

    def encode_role_prefix(self, system_text: str) -> torch.Tensor:
        """Compute frozen role-prefix vectors. [seq_len, hidden_size] on CPU fp32."""
        return self.text_to_hidden(system_text, pool=None).detach()

    # ------------------------------------------------------------------
    # forward with prefix (the load-bearing one)
    # ------------------------------------------------------------------

    def freeze_base_model(self) -> None:
        """Set requires_grad=False on all Qwen parameters.

        Channels stay independently trainable. Called once at training setup.
        """
        if self.cfg.mock or self._model is None:
            return
        for p in self._model.parameters():
            p.requires_grad = False

    def forward_with_prefix_train(
        self,
        role_prefix: torch.Tensor,
        channel_output: torch.Tensor,
        query_text: str | None = None,
        pool: str = "mean",
    ) -> torch.Tensor:
        """Differentiable forward returning the pooled hidden state as a live
        tensor on the model's device. Gradients flow through channel_output;
        role_prefix and Qwen weights are detached/frozen.

        No decoding (we don't need tokens during training; the loss is on
        hidden states). Returns [hidden] on device, dtype = model dtype.
        """
        if self.cfg.mock:
            return self._mock_forward_train(role_prefix, channel_output, query_text, pool)

        device = self._device
        dtype = self.dtype
        # role_prefix is frozen — detach to avoid spurious grads.
        rp = role_prefix.detach().to(device=device, dtype=dtype)
        # channel_output keeps autograd; only cast/move.
        co = channel_output.to(device=device, dtype=dtype)
        parts = [rp, co]
        if query_text:
            q_ids = self._tok(query_text, return_tensors="pt", add_special_tokens=False).input_ids
            q_ids = q_ids.to(device)
            q_emb = self._embed_layer(q_ids)[0].detach()
            parts.append(q_emb)
        inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        out = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        seq = out.hidden_states[-1][0]
        if pool == "mean":
            pooled = seq.mean(dim=0)
        elif pool == "last":
            pooled = seq[-1]
        else:
            raise ValueError(pool)
        return pooled  # live tensor, on device, model dtype

    def teacher_forcing_loss(
        self,
        role_prefix: torch.Tensor,
        channel_output: torch.Tensor,
        query_text: str,
        target_text: str,
        content_token_weight: float = 1.0,
    ) -> torch.Tensor:
        """Compute next-token CE loss on target_text portion of the sequence.

        Sequence layout: [role_prefix ⊕ channel_output ⊕ query_embeds ⊕ target_embeds].
        CE is computed only over positions that predict target tokens. Gradients
        flow through channel_output (live tensor); everything else is detached
        or frozen.

        When `content_token_weight` > 1.0, target tokens identified as
        *content* (ticker symbols, direction words, percent magnitudes,
        sigma values, horizon numbers) are weighted higher than template
        tokens. This concentrates the gradient on grounding rather than
        on producing fluent template text.

        Returns a scalar loss tensor on the model's device.
        """
        if self.cfg.mock:
            return self._mock_teacher_forcing_loss(
                role_prefix, channel_output, query_text, target_text
            )

        device = self._device
        dtype = self.dtype

        rp = role_prefix.detach().to(device=device, dtype=dtype)         # [P, H]
        co = channel_output.to(device=device, dtype=dtype)               # [M, H] (live)

        if query_text:
            q_ids = self._tok(query_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
            q_emb = self._embed_layer(q_ids)[0].detach()                  # [Q, H]
        else:
            q_emb = torch.zeros(0, rp.shape[1], device=device, dtype=dtype)

        # Tokenize target with EOS appended so the model learns to stop.
        tgt_ids = self._tok(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        eos = torch.tensor([[self._tok.eos_token_id]], device=device)
        tgt_ids = torch.cat([tgt_ids, eos], dim=1)
        tgt_emb = self._embed_layer(tgt_ids)[0].detach()                  # [T, H]

        inputs_embeds = torch.cat([rp, co, q_emb, tgt_emb], dim=0).unsqueeze(0)  # [1, P+M+Q+T, H]
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        out = self._model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = out.logits[0]  # [P+M+Q+T, vocab]

        # Predicting target token i happens at sequence position
        # (P+M+Q+i-1) — i.e. the position just before the target token.
        T = tgt_ids.shape[1]
        prefix_len = rp.shape[0] + co.shape[0] + q_emb.shape[0]
        # logits at positions [prefix_len-1 .. prefix_len-1+T-1] predict
        # target tokens [0 .. T-1].
        pred_logits = logits[prefix_len - 1 : prefix_len - 1 + T, :]      # [T, vocab]
        labels = tgt_ids[0]                                                # [T]

        if content_token_weight == 1.0:
            return torch.nn.functional.cross_entropy(pred_logits, labels)

        # Weighted CE: build per-position weights (1.0 for template tokens,
        # `content_token_weight` for content tokens).
        weights = self._content_weights(target_text, labels, content_token_weight)
        per_token_loss = torch.nn.functional.cross_entropy(
            pred_logits, labels, reduction="none"
        )                                                                  # [T]
        weighted = (per_token_loss * weights).sum() / weights.sum().clamp(min=1.0)
        return weighted

    def _content_weights(
        self,
        target_text: str,
        labels: torch.Tensor,
        content_weight: float,
    ) -> torch.Tensor:
        """Build a per-token weight vector: content tokens weighted higher.

        Identifies content tokens by re-tokenizing each *content phrase*
        from the target text and matching token id sequences in `labels`.
        Phrases recognized as content:
          - All-caps tickers (3-5 chars)
          - Direction words: up, down, flat
          - Percent magnitudes: +1.50%, -0.37%, etc.
          - Sigma values: σ≈1.72%
          - Horizon numbers: 60, 120, 30
        Everything else (PREDICTION:, likely to move, over next, minutes,
        CONFIDENCE:, etc.) is template — weight 1.0.
        """
        import re

        device = labels.device
        T = labels.shape[0]
        weights = torch.ones(T, dtype=torch.float32, device=device)

        # Find all content phrases in target_text and tokenize each.
        content_patterns = [
            r"\b[A-Z]{2,5}\b",                       # tickers
            r"\b(?:up|down|flat)\b",                  # direction words
            r"[+\-]?\d+\.\d+%",                      # percent magnitudes
            r"σ≈\d+\.\d+%",                          # sigma values
            r"\b(?:30|60|90|120|240)\b",             # horizon numbers
        ]
        content_phrases = set()
        for pat in content_patterns:
            for m in re.finditer(pat, target_text):
                content_phrases.add(m.group(0))

        # For each content phrase, find its token IDs and mark matching
        # positions in `labels` as content.
        for phrase in content_phrases:
            # Tokenize the phrase (no special tokens; with leading space and
            # without, since BPE handles them differently).
            for prefix in ("", " "):
                phrase_ids = self._tok(
                    prefix + phrase, return_tensors="pt", add_special_tokens=False
                ).input_ids[0].tolist()
                if not phrase_ids:
                    continue
                # Sliding window match in labels.
                L = len(phrase_ids)
                lab_list = labels.tolist()
                for i in range(T - L + 1):
                    if lab_list[i:i + L] == phrase_ids:
                        weights[i:i + L] = content_weight

        return weights

    def _mock_teacher_forcing_loss(
        self,
        role_prefix: torch.Tensor,
        channel_output: torch.Tensor,
        query_text: str,
        target_text: str,
    ) -> torch.Tensor:
        # Mock: a simple gradient-flowing loss so trainer mechanics can be
        # tested without weights. Pull channel_output toward a deterministic
        # target derived from target_text.
        import hashlib
        h = self.hidden_size
        digest = int(hashlib.sha256(target_text.encode()).hexdigest()[:12], 16)
        rng = torch.Generator().manual_seed(digest)
        target = torch.randn(channel_output.shape[0], h, generator=rng)
        return torch.nn.functional.mse_loss(channel_output, target)

    def text_to_hidden_pooled_target(self, text: str, pool: str = "mean") -> torch.Tensor:
        """Detached target hidden state for training. Cached at call site."""
        if self.cfg.mock:
            return self._mock_text_hidden(text, pool=pool).detach()

        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True, use_cache=False)
        seq = out.hidden_states[-1][0]
        if pool == "mean":
            return seq.mean(dim=0).detach()
        if pool == "last":
            return seq[-1].detach()
        raise ValueError(pool)

    def _mock_forward_train(
        self,
        role_prefix: torch.Tensor,
        channel_output: torch.Tensor,
        query_text: str | None,
        pool: str,
    ) -> torch.Tensor:
        # Keep gradient flow through channel_output; everything else detached.
        rp = role_prefix.detach()
        co = channel_output  # live
        parts = [rp, co]
        if query_text:
            parts.append(self._mock_text_hidden(query_text, pool=None).detach())
        cat = torch.cat(parts, dim=0)
        if pool == "mean":
            return cat.mean(dim=0)
        if pool == "last":
            return cat[-1]
        raise ValueError(pool)

    def forward_with_prefix(
        self,
        role_prefix: torch.Tensor,        # [P, hidden]
        channel_output: torch.Tensor,     # [M, hidden]  (already projected through E)
        query_text: str | None = None,
        decode: bool = True,
        max_new_tokens: int | None = None,
        pool: str = "mean",
    ) -> ForwardResult:
        if self.cfg.mock:
            return self._mock_forward_with_prefix(
                role_prefix, channel_output, query_text, decode, pool
            )

        # Build inputs_embeds = [role_prefix ⊕ channel_output ⊕ query_embeds].
        device = self._device
        dtype = self.dtype
        parts = [
            role_prefix.to(device=device, dtype=dtype),
            channel_output.to(device=device, dtype=dtype),
        ]
        if query_text:
            q_ids = self._tok(query_text, return_tensors="pt", add_special_tokens=False).input_ids
            q_ids = q_ids.to(device)
            q_emb = self._embed_layer(q_ids)[0]  # [Q, hidden]
            parts.append(q_emb)
        inputs_embeds = torch.cat(parts, dim=0).unsqueeze(0)  # [1, P+M+Q, hidden]
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

        with torch.no_grad():
            out = self._model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            seq = out.hidden_states[-1][0]
            if pool == "mean":
                pooled = seq.mean(dim=0)
            elif pool == "last":
                pooled = seq[-1]
            else:
                raise ValueError(pool)
            pooled = pooled.float().cpu()

            decoded = None
            if decode:
                gen = self._model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens or self.cfg.max_new_tokens,
                    do_sample=False,
                    repetition_penalty=1.3,
                    pad_token_id=self._tok.eos_token_id,
                )
                # When inputs_embeds is used, generate returns only new tokens.
                decoded = self._tok.decode(gen[0], skip_special_tokens=True).strip()
                # Early-stop at the first complete top-level XML close tag
                # so eval visualizations don't include thinking-mode tail.
                for stop in ("</prediction>", "</synthesis>"):
                    idx = decoded.find(stop)
                    if idx >= 0:
                        decoded = decoded[: idx + len(stop)]
                        break

        return ForwardResult(last_hidden=pooled, decoded_text=decoded)

    # ------------------------------------------------------------------
    # generate_chat — text-in, text-out (used by SelfJudge)
    # ------------------------------------------------------------------

    def generate_chat(
        self,
        system: str,
        user: str,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        enable_thinking: bool = False,
    ) -> str:
        """Standard chat completion. Used for SelfJudge etc.

        temperature=0.0 → greedy. Anything > 0 → sample with that temperature
        (used for self-consistency probes that need variance).

        enable_thinking=False disables Qwen3's <think>...</think> reasoning
        block (default for judging tasks where we want JSON only).
        """
        if self.cfg.mock:
            return self._mock_generate_chat(system, user, temperature)

        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        # Qwen3 chat template supports enable_thinking kwarg.
        try:
            prompt = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            prompt = self._tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tok.eos_token_id,
        }
        if temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = float(temperature)
            gen_kwargs["top_p"] = 0.9
        else:
            gen_kwargs["do_sample"] = False
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self._tok.decode(gen, skip_special_tokens=True).strip()

    def _mock_generate_chat(self, system: str, user: str, temperature: float) -> str:
        digest = hashlib.sha256(
            (system + "||" + user + "||" + str(round(temperature, 3))).encode()
        ).hexdigest()[:6]
        if temperature > 0.0:
            # Inject a tiny per-call random component when sampling so
            # consistency probes see >0 variance.
            import os as _os
            digest = digest + "-" + _os.urandom(2).hex()
        return f'{{"score": 0.5, "rationale": "[mock:{digest}] {user[:60]}"}}'

    # ------------------------------------------------------------------
    # mock paths (deterministic, CPU-only)
    # ------------------------------------------------------------------

    def _mock_text_hidden(self, text: str, pool: str | None) -> torch.Tensor:
        # Produce a short deterministic sequence so Channels see real shapes.
        seq_len = max(4, min(16, len(text) // 8))
        h = self._hidden_size
        digest = int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)
        rng = np.random.default_rng(digest)
        seq = rng.normal(0.0, 1.0 / np.sqrt(h), size=(seq_len, h)).astype(np.float32)
        seq_t = torch.from_numpy(seq)
        if pool is None:
            return seq_t
        if pool == "mean":
            return seq_t.mean(dim=0)
        if pool == "last":
            return seq_t[-1]
        raise ValueError(pool)

    def _mock_forward_with_prefix(
        self,
        role_prefix: torch.Tensor,
        channel_output: torch.Tensor,
        query_text: str | None,
        decode: bool,
        pool: str,
    ) -> ForwardResult:
        h = self._hidden_size
        # Pool by averaging everything together — gives a deterministic vector
        # that depends on the actual prefix and channel output (so Channels
        # exercising the system in mock mode produce different broadcasts).
        parts = [role_prefix, channel_output]
        if query_text:
            parts.append(self._mock_text_hidden(query_text, pool=None))
        cat = torch.cat(parts, dim=0)
        if pool == "mean":
            pooled = cat.mean(dim=0)
        elif pool == "last":
            pooled = cat[-1]
        else:
            raise ValueError(pool)
        decoded = None
        if decode:
            digest = hashlib.sha256(pooled.detach().cpu().numpy().tobytes()).hexdigest()[:8]
            qhint = (query_text or "")[:80]
            decoded = f"[mock-decode:{digest}] {qhint}"
        return ForwardResult(last_hidden=pooled.float().cpu(), decoded_text=decoded)
