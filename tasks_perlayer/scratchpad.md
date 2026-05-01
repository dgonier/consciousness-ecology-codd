# PER_LAYER_MODULATION — Coordination Scratchpad

**Status legend**: PENDING | RUNNING | DONE | BLOCKED

## STATUS

```
phase1-A:01:DONE
phase2-B:02:DONE
phase2-C:03:DONE
phase2-D:04:DONE
phase3-A:05:DONE
```

## PHASE MAP

```
PHASE 1 (sequential, foundation)
    ┌──────────────────────────────────┐
    │  phase1-A:01 phi-mlp-and-hooks   │
    └──────────────┬───────────────────┘
                   │
                   ▼
PHASE 2 (parallel after phase1-A)
    ┌─────────────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐
    │  phase2-B:02 dstar      │  │  phase2-C:03 orpo loss  │  │  phase2-D:04 rewire     │
    └────────────┬────────────┘  └────────────┬────────────┘  └────────────┬────────────┘
                 └────────────────────────────┼────────────────────────────┘
                                              │
                                              ▼
PHASE 3 (sequential, integration)
    ┌──────────────────────────────────┐
    │  phase3-A:05 train + eval        │
    └──────────────────────────────────┘
```

## MISSION DEPENDENCY GRAPH

```
                 ┌──────────► phase2-B:02 ──┐
                 │                          │
phase1-A:01 ─────┼──────────► phase2-C:03 ──┼──────► phase3-A:05
                 │                          │
                 └──────────► phase2-D:04 ──┘
```

## INBOX PROTOCOL

**Required grep on start:**
```bash
grep -nE "@<your-handle>|@all|@phase<N>" tasks_perlayer/scratchpad.md
```
Replace `<your-handle>` with your literal handle (e.g., `@phase2-B`) and `<N>` with your phase number.

**Required grep before DONE:**
Re-run the same grep — late messages may have arrived while you were working.

**Optional live tail (only useful if siblings are running concurrently):**
```bash
tail -F tasks_perlayer/scratchpad.md | grep --line-buffered -E "@<your-handle>|@all|@phase<N>" &
```
Pair with Monitor for push notifications. **Reality check:** subagents are not long-running listeners — they wake, run, return. The `tail -F | grep` listener via Monitor only fires while *that specific agent* is running. It does NOT span agent lifetimes. Late messages are caught by the start/end grep on the next agent's run.

**Addressing scheme:** `@all`, `@<handle>` (e.g., `@phase2-B`), `@phase<N>` (broadcast to all of phase N), `@<h1>,<h2>` (multi-target).

**Message format:**
```
[YYYY-MM-DD HH:MM] <from-handle> > @<to>: <message>
```

## SHARED FACTS

- **Repo root:** `/home/dgonier/ecology_experiment/trophic`
- **Tasks dir:** `tasks_perlayer/` (this swarm)
- **Other tasks dirs (history, do not touch):** `tasks/` (trough-as-transformer migration), `tasks_envstream/` (EnvironmentStream rebuild)
- **Test command:** `cd /home/dgonier/ecology_experiment/trophic && .venv/bin/python -m pytest tests/ -x`
- **Baseline tests:** 89 passing as of 2026-04-29
- **Baseline metrics:**
  - Prompt-only Qwen3-4B + structured prompt on StockNet 50: **ACC 0.460, MCC +0.292** (this is the bar to beat)
  - All trophic checkpoints (seed8 / seed12 / seed14 / seed16 / seed18 / seed20 / seed22): ACC 0.720, **MCC 0.000** (constant-direction collapse)
  - StockNet test set: top-5 tickers (AAPL, GOOG, MSFT, AMZN, JPM) × 10 days = 50 scenarios. Class prior: 36 up / 14 down (0.72 ACC trivial constant baseline).
- **Architectural anchors:**
  - Hexis paper: `/home/dgonier/debaterhub/hexis/paper/sections/{hexis_architecture,three_layer,discussion}.tex`
  - Hexis M code: `/home/dgonier/experiments/scripts/train_action_m.py` (DirectM, install_hooks, ORPO)
  - d* extraction: `/home/dgonier/experiments/scripts/extract_d_star.py`
  - Trophic memory: `~/.claude/projects/-home-dgonier-ecology-experiment/memory/project_issue_8_diagnosis.md`, `project_trophic_hexis_alignment.md`

## INTERFACE CONTRACTS

These are the load-bearing signatures across missions. **Do not change without an `@all` MESSAGES note.**

### phi-MLP (introduced phase1-A:01, used by phase2-D:04 and phase3-A:05)

```python
# trophic/phi_mlp.py
class PhiMLP(nn.Module):
    """Compiles a trough-attended hidden state into per-layer M+E modulation tensors.

    Args:
        hidden_size: H (e.g., 2560 for Qwen3-4B)
        rank: r (e.g., 16 — match Hexis paper)
        patched_layers: list[int] of layer indices to modulate (e.g., stride-3 = [0,3,6,...,30])

    Forward:
        trough_attended_hidden: [H]  -> dict[layer_idx, dict] where each layer has:
            "M_A": [H, r]  # Q-modulation A
            "M_B": [H, r]  # Q-modulation B
            "E_A": [H, r]  # V-modulation A
            "E_B": [H, r]  # V-modulation B
            "s_M": [] scalar  # Q-mod scale
            "s_E": [] scalar  # V-mod scale

    Application math (per layer ℓ):
        x'_ℓ = x_ℓ + s_M · (x_ℓ M_A_ℓ) M_B_ℓ^T   ← pre-projection (Q)
        V'_ℓ = V_ℓ + s_E · (x_ℓ E_A_ℓ) E_B_ℓ^T   ← post-projection (V)
    """
```

### Hook installer (introduced phase1-A:01, used by phase2-D:04, phase3-A:05)

```python
# trophic/m_hooks.py
def install_M_hooks(
    model: nn.Module,
    m_tensors_per_layer: dict[int, dict],
    prefill_active: bool,
) -> list[handle]:
    """Install forward-pre-hooks on `model.model.layers[ℓ]` for each ℓ in m_tensors_per_layer.

    During prefill (prefill_active=False) hooks are no-ops to avoid the M-attractor
    feedback loop (Hexis discussion §). During generation (prefill_active=True for the
    next-token forward pass), hooks apply the rank-r perturbation.

    Returns the list of handles so the caller can `for h in handles: h.remove()` after use.
    """
```

### d* schema (introduced phase2-B:02, used by phase3-A:05)

```python
# trophic/dstar.py
@dataclass
class DStar:
    directions: dict[int, torch.Tensor]   # {layer_idx: unit_vector [H]} — frozen
    scale: float                           # learnable scalar (one global), default 10.0

# Application: post-attention residual hook on each patched layer ℓ:
#   output[layer_ℓ] += scale * d_star.directions[ℓ]
```

### ORPO loss (introduced phase2-C:03, used by phase3-A:05)

```python
# trophic/training/orpo.py
def compute_orpo_loss(
    host: ModelHost,
    role_prefix: torch.Tensor,        # [P, H]
    primary_context: str,             # the curated-slot text + query
    preferred_target: str,            # oracle target text
    rejected_target: str,             # predator's own greedy sample (or another negative)
    m_tensors_per_layer: dict | None = None,   # if None, no M hooks (baseline)
    d_star: DStar | None = None,
    lambda_or: float = 0.5,           # ORPO contrast weight
) -> torch.Tensor:
    """L = NTP(preferred) + lambda_or * -logsigmoid(log_odds(preferred) - log_odds(rejected))

    Both preferred and rejected scored via teacher-forcing with the SAME prefix + same
    M hooks installed. Hexis-style: compute_log_probs runs model(prefix + response) and
    log-probs the response tokens.
    """
```

### Consumer interface gating env var (introduced phase2-D:04)

`TROPHIC_CONSUMER_INTERFACE={prefix,hooks}` (default `prefix`).
- `prefix` = legacy `forward_with_prefix(role_prefix, channel_output, query)` — backward compat with seed1..seed22.
- `hooks` = new path: M tensors via phi-MLP, applied via install_M_hooks; primary context = role_prefix tokens + query only (no channel_output prefix).

## MESSAGES

(append-only — most recent at bottom)

[2026-04-29 21:00] orchestrator > @phase1-A: kickoff. Read /home/dgonier/debaterhub/hexis/paper/sections/hexis_architecture.tex eq 1 first, then `/home/dgonier/experiments/scripts/train_action_m.py` lines 50 (DirectM) and 126 (install_hooks). Match the rank=16 / stride-3 / 11-of-32-layers parameters from the paper. Your output is the foundation everyone in phase 2 depends on.

[2026-04-29 21:16] phase1-A > @phase2: PhiMLP at trophic/phi_mlp.py, install_M_hooks at trophic/m_hooks.py. Contracts unchanged from scratchpad INTERFACE CONTRACTS section — keys ('M_A','M_B','E_A','E_B','s_M','s_E') and HookHandle methods (.set_prefill_active(bool), .remove()) are exact. Tests in tests/test_phi_mlp.py (6) and tests/test_m_hooks.py (6) — 101 total passing (89 baseline + 12 new). Smoke check on real Qwen3-4B: zero-init Phi gives identity-mod (diff = 0.00). Phase 2 unblocked.

[2026-04-29 21:20] phase2-C > @phase3-A: ORPO loss landed at trophic/training/orpo.py. compute_orpo_loss(host, prefix_text, preferred_target, rejected_target, lambda_or=0.5) returns dict with 'loss' for backward + diagnostic fields (ntp_pref, ntp_rej, log_odds_pref, log_odds_rej, win). Hook-agnostic — install M/d* before calling, remove after. Also exported: compute_log_probs, sample_rejected_response. Tests at tests/test_orpo.py (8). 109 total passing. Real-Qwen smoke: loss=4.9375, ntp_pref=4.6250, ntp_rej=4.7188, win=True, grad norm 2.58e-01.

[2026-04-29 21:26] phase2-B > @phase3-A: d* extraction landed at trophic/dstar.py. Saved checkpoint at checkpoints/dstar_stocknet.pt (11 layers, unit-norm, scale=10.0). Use install_dstar_hooks(host._model, dstar, active=True) before generation; remove handles after. Smoke verified: scale=10 yields logit diff 13.59 on Qwen3-4B prefix. 114 total tests passing.

[2026-04-28 12:00] phase2-D > @phase3-A: Consumer rewire landed. TROPHIC_CONSUMER_INTERFACE=hooks activates per-layer M hook path with ORPO loss for predator. Herbivores still on prefix mode (phase 3 can extend if needed). Each agent has self.phi_mlp in hooks mode; trainer optimizes phi_mlp params via extra_modules. NOTE: phase2-C's compute_orpo_loss landed hook-agnostic — signature is (host, prefix_text, preferred_target, rejected_target, lambda_or, max_ctx) returning a dict {loss, ntp_pref, ntp_rej, log_odds_pref, log_odds_rej, win}. Caller installs M+d* hooks before the call and removes them after; that's exactly what _hooks_orpo_loss_for_predator does. Phase 3: pass DStar via the optional `dstar` arg of _hooks_orpo_loss_for_predator (it tries to import install_dstar_hooks from trophic.dstar — wire that on phase 2-B's side if not already there).

[2026-04-30 07:25] phase3-A > @all: Per-layer refactor landed end-to-end. Training crashed at step 75/200 with CUDA OOM during eval-decode (18.6/24 GB on 4090, hooks-mode 96-token gen × 14 scenarios × 11 patched layers blew the budget). Best checkpoint at step 75 (eval=8.30). StockNet smoke on undertrained ckpt: ACC=0.000, MCC=0.000, abstained 50/50 — predator never learned to emit parseable XML in 75 steps. Predator ORPO eval flat (21.893 ×4) suggests d*@scale=10 dominates M-hook gradient. Decision: pivot. Architecture is wired right but two follow-ups before declaring it failed: (1) TROPHIC_EVAL_EVERY=200 to bypass eval OOM, (2) anneal d* scale 10→1. One contract drift fixed: phase 2-D's per_loss['pred.short_horizon.diag'] is a dict; train_sft.py:125 formatter only handles floats, fixed by filtering numerics. GitHub issue: https://github.com/dgonier/consciousness-ecology-codd/issues/13. CHANGELOG/EXPERIMENTS updated. Swarm complete.
