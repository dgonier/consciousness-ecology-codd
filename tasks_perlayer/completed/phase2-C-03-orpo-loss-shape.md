# Mission 03: orpo-loss-shape

**Handle**: phase2-C
**Phase**: 2 (parallel after phase1-A)
**Mission file**: `phase2-C-03-orpo-loss-shape.md`
**Dependencies**: phase1-A:01
**Blocks**: phase3-A:05

---

## Before You Start

```bash
# 1. Confirm phase 1 is done.
grep "phase1-A:01:DONE" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 2. Read the Hexis ORPO recipe.
sed -n '150,235p' /home/dgonier/experiments/scripts/train_action_m.py
# Pay attention to compute_log_probs (lines ~153-179) and the ORPO loss block (~220-225).

# 3. Read the existing teacher_forcing_loss to know what shape to match for callers.
grep -n "def teacher_forcing_loss\|content_token_weight" /home/dgonier/ecology_experiment/trophic/trophic/model_host.py | head

# 4. Inbox check.
grep -nE "@phase2-C|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

# 5. Flip status.
sed -i 's/^phase2-C:03:PENDING$/phase2-C:03:RUNNING/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md
```

## Goal

Replace the trophic SFT loss (currently teacher-forced NTP only) with an **ORPO-shaped** loss that scores both a preferred and a rejected response. This adds an active "push the prior away from the wrong answer" signal that teacher-forcing alone misses — the missing ingredient that causes constant-direction collapse on StockNet (the model satisfies "match AAPL down" by tweaking hidden states without disrupting its "always emit MSFT up" prior).

The ORPO contract `compute_orpo_loss(host, role_prefix, primary_context, preferred_target, rejected_target, m_tensors_per_layer, d_star, lambda_or)` is published in scratchpad's INTERFACE CONTRACTS — implement it exactly.

## Files to Create / Modify

Create:
- `trophic/training/orpo.py` — `compute_log_probs`, `compute_orpo_loss`, helper that draws a rejected sample from the host
- `tests/test_orpo.py` — at least 5 tests

Do NOT modify (yet):
- `trophic/training/sft.py` (phase2-D rewires the runner; you stop at producing the ORPO function)
- `trophic/model_host.py` (your function should call `host._model` directly, not extend host's API)

## Implementation Steps

### 1. ORPO loss

```python
# trophic/training/orpo.py
"""ORPO-shaped loss for trophic.

Per Hexis Phase A (train_action_m.py): teacher-force log-probs of both a
preferred and a rejected response under the same prefix; compute

    L = NTP(preferred) + lambda_or * -logsigmoid(log_odds_pref - log_odds_rej)

where log_odds(x) = log(p) - log(1-p) using the response's average
log-probability as p.

This forces the predator to assign higher likelihood to the preferred
response AND lower likelihood to the rejected one, which prevents the
"always-up constant" trap that teacher-forcing alone falls into.
"""
from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from ..model_host import ModelHost


def compute_log_probs(
    host: ModelHost,
    prefix_text: str,                    # role-prefix + primary context (curated slot)
    response_text: str,                  # the response to score
    max_ctx: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Teacher-forced log-probs of `response_text` given `prefix_text`.

    Returns (avg_log_prob, ntp_loss) where:
      - avg_log_prob is the mean log-probability of response tokens (a tensor with grad).
      - ntp_loss = -avg_log_prob (CE).

    Hexis-equivalent: matches train_action_m.py:153-179.
    """
    tok = host._tok
    device = host.device

    resp_ids = tok(response_text, return_tensors="pt",
                   max_length=512, truncation=True,
                   add_special_tokens=False).input_ids
    resp_len = resp_ids.shape[1]

    prompt_ids = tok(prefix_text, return_tensors="pt",
                     max_length=max(max_ctx - resp_len, 256),
                     truncation=True).input_ids
    p_len = prompt_ids.shape[1]

    ids = torch.cat([prompt_ids, resp_ids], dim=1).to(device)
    am = torch.ones_like(ids)

    out = host._model(input_ids=ids, attention_mask=am)
    logits = out.logits  # [1, L, V]
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target = ids[:, 1:]
    token_lp = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # [1, L-1]

    mask = torch.zeros_like(token_lp)
    mask[:, p_len - 1:] = 1.0
    denom = mask.sum().clamp(min=1)
    avg_lp = (token_lp * mask).sum() / denom
    ntp = -avg_lp
    return avg_lp, ntp


def compute_orpo_loss(
    host: ModelHost,
    prefix_text: str,
    preferred_target: str,
    rejected_target: str,
    lambda_or: float = 0.5,
    max_ctx: int = 4096,
) -> dict[str, torch.Tensor]:
    """Hexis Phase A loss shape, parameterized by prefix_text.

    The caller is responsible for installing whatever per-layer M hooks /
    d* hooks they want active for the inner forward passes BEFORE calling
    this function, and removing them after. compute_orpo_loss is hook-
    agnostic — it just calls host._model.

    Returns dict with:
        loss: scalar tensor (the thing to backward)
        ntp_pref: scalar
        ntp_rej: scalar
        log_odds_pref: scalar
        log_odds_rej: scalar
        win: bool — whether NTP(pref) < NTP(rej) on this example
    """
    lp_w, ntp_w = compute_log_probs(host, prefix_text, preferred_target, max_ctx=max_ctx)
    lp_l, ntp_l = compute_log_probs(host, prefix_text, rejected_target, max_ctx=max_ctx)

    # log odds: log p - log(1-p). p = exp(avg_lp). Clamp p < 0.9999 to avoid -inf.
    log_odds_w = lp_w - torch.log1p(-torch.exp(lp_w).clamp(max=0.9999))
    log_odds_l = lp_l - torch.log1p(-torch.exp(lp_l).clamp(max=0.9999))
    L_or = -F.logsigmoid(log_odds_w - log_odds_l)
    L = ntp_w + lambda_or * L_or
    return {
        "loss": L,
        "ntp_pref": ntp_w.detach(),
        "ntp_rej": ntp_l.detach(),
        "log_odds_pref": log_odds_w.detach(),
        "log_odds_rej": log_odds_l.detach(),
        "win": (ntp_w.detach() < ntp_l.detach()).item(),
    }


def sample_rejected_response(
    host: ModelHost,
    prefix_text: str,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
) -> str:
    """Greedy-sample a response from the host model under the given prefix.

    Used to provide a "rejected" example for ORPO when no manual negative is
    available. NOTE: caller must install hooks BEFORE calling this if they
    want the sample to come from the hooked model — otherwise it samples
    from the bare frozen base (which gives a more contrastive negative,
    arguably a stronger signal).
    """
    tok = host._tok
    ids = tok(prefix_text, return_tensors="pt").input_ids.to(host.device)
    am = torch.ones_like(ids)
    with torch.no_grad():
        out = host._model.generate(
            input_ids=ids,
            attention_mask=am,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            pad_token_id=tok.eos_token_id,
        )
    new = out[0, ids.shape[1]:]
    return tok.decode(new, skip_special_tokens=True)
```

### 2. Tests

```python
# tests/test_orpo.py
"""Unit tests for the ORPO loss. Uses a tiny stub host so tests run fast and
don't load Qwen3-4B."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _StubModel(nn.Module):
    """Behaves enough like a HF causal LM that compute_log_probs runs.

    Returns logits with intentional bias: the token id of `bias_target_token`
    receives +bias_amt. This lets us craft scenarios where one response is
    systematically more likely than another and verify ORPO's gradient sign.
    """
    def __init__(self, vocab_size, hidden=8, bias_target_token=None, bias_amt=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.bias_target_token = bias_target_token
        self.bias_amt = bias_amt
        self.embed = nn.Embedding(vocab_size, hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None):
        h = self.embed(input_ids)
        logits = self.head(h)
        if self.bias_target_token is not None:
            logits = logits.clone()
            logits[..., self.bias_target_token] = logits[..., self.bias_target_token] + self.bias_amt

        class _Out: pass
        o = _Out()
        o.logits = logits
        return o

    def generate(self, input_ids, attention_mask=None, max_new_tokens=10,
                 do_sample=False, temperature=1.0, pad_token_id=0):
        # Greedy: append argmax repeatedly.
        cur = input_ids
        for _ in range(max_new_tokens):
            o = self.forward(cur, attention_mask=None)
            nxt = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, nxt], dim=1)
        return cur


class _StubTok:
    """Trivial tokenizer: ASCII-1-token-per-char."""
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
    def __call__(self, text, return_tensors=None, max_length=None, truncation=False,
                 add_special_tokens=True):
        ids = [min(ord(c), self.vocab_size - 1) for c in text][:max_length] if max_length else \
              [min(ord(c), self.vocab_size - 1) for c in text]
        if not ids: ids = [0]
        t = torch.tensor([ids], dtype=torch.long)
        class _R: pass
        r = _R()
        r.input_ids = t
        return r
    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if int(i) > 0)


class _StubHost:
    def __init__(self, model, tok, device, dtype=torch.float32):
        self._model = model
        self._tok = tok
        self.device = device
        self.dtype = dtype


def _make_host(bias_target_token=None, bias_amt=0.0):
    tok = _StubTok()
    model = _StubModel(vocab_size=256, bias_target_token=bias_target_token, bias_amt=bias_amt)
    return _StubHost(model, tok, torch.device("cpu"))


def test_compute_log_probs_returns_finite():
    from trophic.training.orpo import compute_log_probs
    host = _make_host()
    lp, ntp = compute_log_probs(host, "Hello", " world!")
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ntp).all()
    assert ntp.item() >= 0  # NTP loss is non-negative


def test_compute_log_probs_grad():
    from trophic.training.orpo import compute_log_probs
    host = _make_host()
    lp, ntp = compute_log_probs(host, "Hi", " there")
    ntp.backward()
    g = host._model.head.weight.grad
    assert g is not None
    assert g.abs().sum() > 0


def test_orpo_prefers_preferred():
    """When the model is biased toward the preferred response's tokens,
    NTP(pref) < NTP(rej) and the win flag is True."""
    from trophic.training.orpo import compute_orpo_loss
    # Bias toward 'A' (token 65). Preferred is "AAAA"; rejected is "ZZZZ".
    host = _make_host(bias_target_token=65, bias_amt=10.0)
    out = compute_orpo_loss(host, "Q:", "AAAA", "ZZZZ", lambda_or=0.5)
    assert out["win"] is True
    # Loss is finite and positive
    assert torch.isfinite(out["loss"])


def test_orpo_loss_decreases_when_bias_grows():
    from trophic.training.orpo import compute_orpo_loss
    out_low = compute_orpo_loss(_make_host(bias_target_token=65, bias_amt=2.0),
                                 "Q:", "AAAA", "ZZZZ")
    out_high = compute_orpo_loss(_make_host(bias_target_token=65, bias_amt=10.0),
                                  "Q:", "AAAA", "ZZZZ")
    assert out_high["loss"].item() < out_low["loss"].item(), \
        f"bias 10 ({out_high['loss']:.3f}) should give lower loss than bias 2 ({out_low['loss']:.3f})"


def test_orpo_loss_grad_flows():
    """A backward call on the loss should populate gradients on host._model."""
    from trophic.training.orpo import compute_orpo_loss
    host = _make_host()
    out = compute_orpo_loss(host, "Q:", "answer", "wrong", lambda_or=0.5)
    out["loss"].backward()
    assert host._model.head.weight.grad is not None
    assert host._model.head.weight.grad.abs().sum() > 0


def test_sample_rejected_response_runs():
    """Greedy-sample produces SOME string."""
    from trophic.training.orpo import sample_rejected_response
    host = _make_host(bias_target_token=65, bias_amt=10.0)
    s = sample_rejected_response(host, "prompt:", max_new_tokens=5)
    assert isinstance(s, str)
```

## Acceptance Criteria

- [ ] `trophic/training/orpo.py` exists with `compute_log_probs`, `compute_orpo_loss`, `sample_rejected_response`.
- [ ] `compute_orpo_loss` returns the dict shape specified in the docstring.
- [ ] All 6 new tests pass.
- [ ] Pre-existing 89 + phase1-A's 8 tests still pass (you should be at 103).
- [ ] No modifications to `sft.py`, `model_host.py`, or any agent file.
- [ ] `compute_orpo_loss` is hook-agnostic (you can install M / d* hooks before calling and remove them after; the function never touches hook state).

## Testing Conditions (exit verification)

```bash
cd /home/dgonier/ecology_experiment/trophic

# 1. ORPO tests pass
.venv/bin/python -m pytest tests/test_orpo.py -v
# Expected: 6 passed

# 2. Pre-existing + phase 1 tests still pass
.venv/bin/python -m pytest tests/ -x --ignore=tests/test_orpo.py
# Expected: 97 passed

# 3. Smoke check: ORPO loss is finite + grad flows on real Qwen
.venv/bin/python -c "
import torch
from trophic.config import DEFAULT_CONFIG
from trophic.model_host import ModelHost
from trophic.training.orpo import compute_orpo_loss

host = ModelHost.get(DEFAULT_CONFIG.model)
# Need grad on the head — we don't actually train host, but just to confirm grad flows.
# Unfreeze just for the smoke test, then refreeze.
for p in host._model.parameters():
    p.requires_grad_(False)
last_layer = list(host._model.parameters())[-1]
last_layer.requires_grad_(True)

prefix = 'Predict the direction:'
out = compute_orpo_loss(host, prefix, '<prediction>up</prediction>', '<prediction>down</prediction>', lambda_or=0.5)
print(f'loss={out[\"loss\"].item():.4f} ntp_pref={out[\"ntp_pref\"].item():.4f} ntp_rej={out[\"ntp_rej\"].item():.4f} win={out[\"win\"]}')
out['loss'].backward()
print(f'last-param grad norm: {last_layer.grad.norm().item():.4e} (should be > 0)')
assert torch.isfinite(out['loss']) and last_layer.grad.norm().item() > 0
print('OK')
"
# Expected: prints loss/NTPs/win flag, grad norm > 0, "OK"
```

## Coordination

You're parallel with **phase2-B** (d*) and **phase2-D** (consumer rewire). Files don't overlap. The only shared concern is the `compute_orpo_loss` signature — phase 3 will call it. If you change the signature, post `@all` BEFORE the change.

Note for **phase2-D**: when you wire the new path in sft.py, the dispatch should call `compute_orpo_loss(host, prefix_text=role_prefix_text + "\n" + curated_slot + query_text, preferred_target=oracle, rejected_target=...)`. **You** should NOT touch sft.py — D handles that.

For the rejected sample: phase 3's training loop should call `sample_rejected_response(host, prefix)` once per scenario per step, using the BARE host (no M hooks installed) so the rejected is a genuine "what would the prior model say" baseline. This is a stronger contrast than sampling from the hooked model.

## When Done

```bash
grep -nE "@phase2-C|@all|@phase2" /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

sed -i 's/^phase2-C:03:RUNNING$/phase2-C:03:DONE/' \
  /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md

cat >> /home/dgonier/ecology_experiment/trophic/tasks_perlayer/scratchpad.md <<EOF
[$(date +%Y-%m-%d\ %H:%M)] phase2-C > @phase3-A: ORPO loss landed at trophic/training/orpo.py. compute_orpo_loss(host, prefix_text, preferred_target, rejected_target, lambda_or=0.5) returns dict with 'loss' for backward + diagnostic fields. Hook-agnostic — install M/d* before calling, remove after.
EOF

mv /home/dgonier/ecology_experiment/trophic/tasks_perlayer/phase2-C-03-orpo-loss-shape.md \
   /home/dgonier/ecology_experiment/trophic/tasks_perlayer/completed/
```
