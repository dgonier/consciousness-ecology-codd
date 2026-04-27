"""Singleton wrapper for Qwen2.5-Math-1.5B-Instruct.

This is a *tool* used by the Interrogator herbivore — not a peer agent in
the trophic stack. The Interrogator decides what to ask, calls
MathHost.solve(question), and uses the answer in its eventual synthesis
broadcast.

Loaded once on cuda; bf16. Fits in ~3GB alongside Qwen3-4B.

Tool-integrated reasoning (Qwen2.5-Math's native pattern of emitting
Python code blocks for arithmetic) is *not* executed in v1 — we just read
the model's chain-of-thought and final answer. Adding a sandboxed
executor is a future upgrade.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Optional

import torch

from .config import DEFAULT_CONFIG, ModelConfig


SYSTEM_PROMPT = (
    "You are a precise quantitative analyst. Given numerical data and a"
    " question, perform the calculation step by step and emit the final"
    " answer in a structured form.\n\n"
    "Format your response as:\n"
    "WORK: <brief calculation>\n"
    "ANSWER: <single number or short structured value>\n\n"
    "Be concise. Do not narrate. Do not use Python. Do not use markdown."
    " Just compute and answer."
)


@dataclass
class MathResult:
    """Parsed result from a Math node call."""
    work: str                       # the calculation steps
    answer: str                     # the final answer (string form)
    answer_num: float | None        # numeric form if extractable
    raw: str                        # full unparsed response


def _parse_math_response(raw: str) -> MathResult:
    # Strip <think>...</think> blocks (some Qwen variants emit these).
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if not cleaned:
        cleaned = raw

    # Pull WORK and ANSWER sections.
    work_match = re.search(r"WORK:\s*(.+?)(?=ANSWER:|$)", cleaned, re.DOTALL)
    answer_match = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", cleaned, re.DOTALL)

    work = work_match.group(1).strip() if work_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    # If structured parse failed, fall back to "last number in the response".
    if not answer:
        nums = re.findall(r"-?\d+\.?\d*", cleaned)
        if nums:
            answer = nums[-1]

    # Try to extract a single numeric value from the answer.
    num_match = re.search(r"-?\d+\.?\d*", answer)
    answer_num = float(num_match.group(0)) if num_match else None

    return MathResult(work=work, answer=answer, answer_num=answer_num, raw=raw)


class MathHost:
    _instance: Optional["MathHost"] = None
    _lock = threading.Lock()

    def __init__(self, cfg: ModelConfig | None = None):
        self.cfg = cfg or DEFAULT_CONFIG.model
        self._tok = None
        self._model = None
        self._device = "cpu"
        if not self.cfg.mock:
            self._load()

    @classmethod
    def get(cls, cfg: ModelConfig | None = None) -> "MathHost":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls._instance = None

    def _load(self) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        dtype = dtype_map.get(self.cfg.dtype, torch.bfloat16)
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=self.cfg.device,
        )
        self._model.eval()
        self._device = next(self._model.parameters()).device

    def solve(
        self,
        question: str,
        max_new_tokens: int = 512,
    ) -> MathResult:
        """Run the math model on a single question. Returns parsed result.

        The question text should be self-contained — including any numbers
        the math node needs to see. The Interrogator herbivore is
        responsible for assembling the question with relevant data inline.
        """
        if self.cfg.mock:
            return self._mock_solve(question)

        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
        inputs = self._tok(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tok.eos_token_id,
            )
        gen = out[0, inputs["input_ids"].shape[1]:]
        raw = self._tok.decode(gen, skip_special_tokens=True).strip()
        return _parse_math_response(raw)

    def freeze(self) -> None:
        """Freeze the math model's parameters. Call before training the
        cross-model E so gradients flow only through E."""
        if self.cfg.mock or self._model is None:
            return
        for p in self._model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        if self._model is None:
            return 1536
        return self._model.config.hidden_size

    @property
    def device(self) -> torch.device | str:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        if self.cfg.mock or self._model is None:
            return torch.float32
        return next(self._model.parameters()).dtype

    @property
    def embed_layer(self) -> torch.nn.Module:
        return self._model.get_input_embeddings()

    def forward_with_prefix_embeds(
        self,
        prefix_embeds: torch.Tensor,            # [P, hidden_math] in Math's space
        max_new_tokens: int = 256,
        return_hidden: bool = False,
    ) -> tuple[str, torch.Tensor | None]:
        """Run Math model from a sequence of input embeddings (already in
        Math's embedding space — caller projected through E_analyst→math).
        Returns (decoded_text, last_pooled_hidden_state | None).

        Used for the token-bypass path between Analyst and Math.
        """
        if self.cfg.mock:
            # Mock: just return a stub text + zero hidden
            stub_hidden = (
                torch.zeros(self.hidden_size) if return_hidden else None
            )
            return ("[mock]", stub_hidden)

        device = self._device
        dtype = self.dtype
        embeds = prefix_embeds.to(device=device, dtype=dtype).unsqueeze(0)
        attention_mask = torch.ones(
            embeds.shape[:2], dtype=torch.long, device=device
        )
        with torch.no_grad():
            gen = self._model.generate(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=self._tok.eos_token_id,
                output_hidden_states=return_hidden,
                return_dict_in_generate=return_hidden,
            )
        if return_hidden:
            # The generate call returns a structure; extract sequences and
            # last-layer hidden state at the last generated position.
            sequences = gen.sequences
            decoded = self._tok.decode(sequences[0], skip_special_tokens=True).strip()
            # gen.hidden_states is a tuple per generated step;
            # last step, last layer, last token position.
            last_hidden = gen.hidden_states[-1][-1][0, -1, :].float().cpu()
            return decoded, last_hidden
        else:
            decoded = self._tok.decode(gen[0], skip_special_tokens=True).strip()
            return decoded, None

    def text_to_input_embeds(self, text: str) -> torch.Tensor:
        """Tokenize text and look up its input embeddings in Math's space.
        Used to assemble a hybrid prefix: [system_text_embeds ⊕ projected_query].
        Returns [seq_len, hidden_math] on Math's device.
        """
        if self.cfg.mock:
            # Mock: deterministic sequence
            import hashlib
            digest = int(hashlib.sha256(text.encode()).hexdigest()[:12], 16)
            rng = torch.Generator().manual_seed(digest)
            n = max(4, min(16, len(text) // 8))
            return torch.randn(n, self.hidden_size, generator=rng) * 0.1
        ids = self._tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(self._device)
        return self.embed_layer(ids)[0].detach()

    def _mock_solve(self, question: str) -> MathResult:
        # Mock: try to compute trivial arithmetic from the question text;
        # otherwise return a stub.
        import re
        # Look for "(close - open) / open * 100" style or just two numbers.
        nums = re.findall(r"-?\d+\.?\d*", question)
        if len(nums) >= 2:
            try:
                a, b = float(nums[0]), float(nums[1])
                pct = (b - a) / a * 100.0 if a != 0 else 0.0
                return MathResult(
                    work=f"({b} - {a}) / {a} * 100 = {pct:.4f}",
                    answer=f"{pct:.2f}",
                    answer_num=pct,
                    raw=f"WORK: ({b} - {a}) / {a} * 100 = {pct:.4f}\nANSWER: {pct:.2f}",
                )
            except (ValueError, ZeroDivisionError):
                pass
        return MathResult(work="", answer="0", answer_num=0.0,
                          raw=f"[mock] {question[:80]}")
