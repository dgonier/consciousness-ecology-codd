"""ORPO-shaped loss for trophic.

Per Hexis Phase A (train_action_m.py): teacher-force log-probs of both a
preferred and a rejected response under the same prefix; compute

    L = NTP(preferred) + lambda_or * -logsigmoid(log_odds_pref - log_odds_rej)

where log_odds(x) = log(p) - log(1-p) using the response's average
log-probability as p.

This forces the predator to assign higher likelihood to the preferred
response AND lower likelihood to the rejected one, which prevents the
"always-up constant" trap that teacher-forcing alone falls into.

The functions in this module are HOOK-AGNOSTIC: callers install whatever
M / d* hooks they want on host._model BEFORE calling, and remove them
after. compute_orpo_loss never touches hook state.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_log_probs(
    host,
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

    resp_ids = tok(
        response_text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        add_special_tokens=False,
    ).input_ids
    resp_len = resp_ids.shape[1]

    prompt_ids = tok(
        prefix_text,
        return_tensors="pt",
        max_length=max(max_ctx - resp_len, 256),
        truncation=True,
    ).input_ids
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
    host,
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
        ntp_pref: scalar (detached)
        ntp_rej: scalar (detached)
        log_odds_pref: scalar (detached)
        log_odds_rej: scalar (detached)
        win: bool — whether NTP(pref) < NTP(rej) on this example
    """
    lp_w, ntp_w = compute_log_probs(host, prefix_text, preferred_target, max_ctx=max_ctx)
    lp_l, ntp_l = compute_log_probs(host, prefix_text, rejected_target, max_ctx=max_ctx)

    # log odds: log p - log(1-p). p = exp(avg_lp). Clamp p < 0.9999 to avoid log1p(-1)=-inf.
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
        "win": bool((ntp_w.detach() < ntp_l.detach()).item()),
    }


def sample_rejected_response(
    host,
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
