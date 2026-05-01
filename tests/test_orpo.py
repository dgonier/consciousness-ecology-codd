"""Unit tests for the ORPO loss. Uses a tiny stub host so tests run fast and
don't load Qwen3-4B.

The stub host gives compute_orpo_loss everything it touches: ._model (a
forward-able nn.Module returning .logits), ._tok (a callable that returns an
object with .input_ids), and .device. That lets us cover both correctness of
the math and gradient flow without GPU or HF model load.
"""
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
            logits[..., self.bias_target_token] = (
                logits[..., self.bias_target_token] + self.bias_amt
            )

        class _Out:
            pass

        o = _Out()
        o.logits = logits
        return o

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens=10,
        do_sample=False,
        temperature=1.0,
        pad_token_id=0,
    ):
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

    def __call__(
        self,
        text,
        return_tensors=None,
        max_length=None,
        truncation=False,
        add_special_tokens=True,
    ):
        ids = [min(ord(c), self.vocab_size - 1) for c in text]
        if max_length is not None:
            ids = ids[:max_length]
        if not ids:
            ids = [0]
        t = torch.tensor([ids], dtype=torch.long)

        class _R:
            pass

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
    model = _StubModel(
        vocab_size=256, bias_target_token=bias_target_token, bias_amt=bias_amt
    )
    return _StubHost(model, tok, torch.device("cpu"))


def test_compute_log_probs_returns_finite():
    from trophic.training.orpo import compute_log_probs

    host = _make_host()
    lp, ntp = compute_log_probs(host, "Hello", " world!")
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ntp).all()
    # NTP loss is non-negative (it's -avg_log_prob, and avg_log_prob <= 0).
    assert ntp.item() >= 0


def test_compute_log_probs_grad():
    from trophic.training.orpo import compute_log_probs

    host = _make_host()
    lp, ntp = compute_log_probs(host, "Hi", " there")
    ntp.backward()
    g = host._model.head.weight.grad
    assert g is not None
    assert g.abs().sum() > 0


def test_orpo_returns_expected_keys():
    """Output dict has exactly the documented schema."""
    from trophic.training.orpo import compute_orpo_loss

    host = _make_host()
    out = compute_orpo_loss(host, "Q:", "yes", "no", lambda_or=0.5)
    expected = {"loss", "ntp_pref", "ntp_rej", "log_odds_pref", "log_odds_rej", "win"}
    assert set(out.keys()) == expected
    assert isinstance(out["win"], bool)
    assert torch.isfinite(out["loss"])


def test_orpo_prefers_preferred():
    """When the model is biased toward the preferred response's tokens,
    NTP(pref) < NTP(rej) and the win flag is True."""
    from trophic.training.orpo import compute_orpo_loss

    # Bias toward 'A' (token 65). Preferred is "AAAA"; rejected is "ZZZZ".
    host = _make_host(bias_target_token=65, bias_amt=10.0)
    out = compute_orpo_loss(host, "Q:", "AAAA", "ZZZZ", lambda_or=0.5)
    assert out["win"] is True
    assert torch.isfinite(out["loss"])
    # NTP(pref) should be strictly less than NTP(rej).
    assert out["ntp_pref"].item() < out["ntp_rej"].item()


def test_orpo_loss_decreases_when_bias_grows():
    """Stronger bias toward preferred tokens should yield a lower combined loss."""
    from trophic.training.orpo import compute_orpo_loss

    out_low = compute_orpo_loss(
        _make_host(bias_target_token=65, bias_amt=2.0),
        "Q:",
        "AAAA",
        "ZZZZ",
    )
    out_high = compute_orpo_loss(
        _make_host(bias_target_token=65, bias_amt=10.0),
        "Q:",
        "AAAA",
        "ZZZZ",
    )
    assert out_high["loss"].item() < out_low["loss"].item(), (
        f"bias 10 ({out_high['loss']:.3f}) should give lower loss than "
        f"bias 2 ({out_low['loss']:.3f})"
    )


def test_orpo_loss_grad_flows():
    """A backward call on the loss should populate gradients on host._model."""
    from trophic.training.orpo import compute_orpo_loss

    host = _make_host()
    out = compute_orpo_loss(host, "Q:", "answer", "wrong", lambda_or=0.5)
    out["loss"].backward()
    assert host._model.head.weight.grad is not None
    assert host._model.head.weight.grad.abs().sum() > 0


def test_orpo_loss_no_inf_at_high_prob():
    """If avg log-prob is essentially 0 (p≈1), the 0.9999 clamp must keep
    log_odds finite. We test by giving an extreme bias toward a single token
    that the response is composed entirely of."""
    from trophic.training.orpo import compute_orpo_loss

    host = _make_host(bias_target_token=65, bias_amt=1000.0)
    out = compute_orpo_loss(host, "Q:", "AAAA", "ZZZZ", lambda_or=0.5)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["log_odds_pref"])
    assert torch.isfinite(out["log_odds_rej"])


def test_sample_rejected_response_runs():
    """Greedy-sample produces a string."""
    from trophic.training.orpo import sample_rejected_response

    host = _make_host(bias_target_token=65, bias_amt=10.0)
    s = sample_rejected_response(host, "prompt:", max_new_tokens=5)
    assert isinstance(s, str)
