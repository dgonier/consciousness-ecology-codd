"""DSPy LM adapter that drives the local Qwen3-4B via trophic.model_host.

This is the bridge that lets DSPy's structured-output Signatures hit the
exact same model the apex uses, on the same GPU, with shared weights.

Usage:
    import dspy
    from trophic.beliefs.dspy_qwen_lm import QwenLocalLM
    from trophic.model_host import ModelHost
    from trophic.config import GlobalConfig
    host = ModelHost(GlobalConfig().model)
    dspy.configure(lm=QwenLocalLM(host, max_tokens=1024))
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

import dspy


class _ChatMessage:
    """OpenAI-style message stub (object with .content + .role attrs)."""

    def __init__(self, content: str, role: str = "assistant"):
        self.content = content
        self.role = role
        self.tool_calls: list = []
        self.refusal: Optional[str] = None


class _Choice:
    def __init__(self, content: str, index: int = 0):
        self.message = _ChatMessage(content)
        self.index = index
        self.finish_reason = "stop"
        self.logprobs = None


class _Usage(dict):
    """Subclasses dict so DSPy code that does `dict(response.usage)` or
    iterates over keys works, while attribute access still works for
    OpenAI-style code (`response.usage.prompt_tokens`)."""

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        super().__init__()
        self["prompt_tokens"] = prompt_tokens
        self["completion_tokens"] = completion_tokens
        self["total_tokens"] = prompt_tokens + completion_tokens

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class _OpenAIShapedResponse:
    """Minimal OpenAI ChatCompletion-shaped response object so DSPy's
    response handling code (which expects `.choices[i].message.content`)
    works without modification.
    """

    def __init__(self, content: str, model_name: str, prompt_tokens: int = 0):
        self.id = f"qwen-local-{uuid.uuid4().hex[:8]}"
        self.created = int(time.time())
        self.model = model_name
        self.object = "chat.completion"
        self.choices = [_Choice(content)]
        self.usage = _Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=len(content.split()),  # rough
        )

    def model_dump(self, *args, **kwargs) -> dict:
        # DSPy may call model_dump on responses; provide a serializable view
        return {
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "object": self.object,
            "choices": [
                {
                    "index": c.index,
                    "finish_reason": c.finish_reason,
                    "message": {"content": c.message.content, "role": c.message.role},
                }
                for c in self.choices
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens,
            },
        }


class QwenLocalLM(dspy.BaseLM):
    """DSPy LM that calls trophic.model_host.ModelHost.generate_chat()."""

    def __init__(
        self,
        host,                    # ModelHost
        model_name: str = "qwen3-4b-local",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        enable_thinking: bool = False,
        cache: bool = True,
    ):
        super().__init__(
            model=model_name,
            model_type="chat",
            temperature=temperature,
            max_tokens=max_tokens,
            cache=cache,
        )
        self.host = host
        self.enable_thinking = enable_thinking

    @property
    def supports_function_calling(self) -> bool:
        return False

    @property
    def supports_reasoning(self) -> bool:
        return False

    @property
    def supports_response_schema(self) -> bool:
        return False

    @property
    def supported_params(self) -> set[str]:
        return {"temperature", "max_tokens"}

    def forward(self, prompt: Optional[str] = None, messages: Optional[list[dict]] = None, **kwargs) -> Any:
        # DSPy passes either a single `prompt` or `messages=[{role, content}, ...]`.
        # Collapse into (system, user) for Qwen's chat template.
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        system_parts: list[str] = []
        user_parts: list[str] = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                # multimodal; stringify
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            role = m.get("role", "user")
            if role == "system":
                system_parts.append(str(content))
            else:
                user_parts.append(str(content))
        system_str = "\n".join(system_parts).strip()
        user_str = "\n".join(user_parts).strip()

        merged = {**self.kwargs, **kwargs}
        max_new = int(merged.get("max_tokens", 1024))
        temp = float(merged.get("temperature", 0.0))

        text = self.host.generate_chat(
            system=system_str or "You are a helpful assistant.",
            user=user_str,
            max_new_tokens=max_new,
            temperature=temp,
            enable_thinking=self.enable_thinking,
        )

        prompt_tokens = len((system_str + " " + user_str).split())
        return _OpenAIShapedResponse(
            content=text or "",
            model_name=self.model,
            prompt_tokens=prompt_tokens,
        )
