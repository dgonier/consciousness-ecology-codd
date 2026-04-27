"""Channel — the trophic-edge cross-attention module.

The same architecture sits at every trophic edge (producer→herbivore,
herbivore→predator, predator→apex_when_built). Asymmetry of the trophic
relationship is encoded in K/Q/V assignment + a learnable null-attention
slot:

  - Q comes from the hunter (herbivore for tier 1, predator for tier 2)
  - K, V come from the prey (the available food items)
  - K_null, V_null are learnable per-edge: when no real prey clears the
    null gate, the hunter abstains (returns the null V and a high
    null_prob, which the runner uses to short-circuit the synthesis step).

W_Q, W_K, W_V are learnable linear projections. Identity-initialized so an
untrained Channel passes the producer's pooled hidden state through nearly
unchanged — the system runs forward-only without any training and the
behavior is interpretable from tick 1.

Multi-head attention. Default 8 heads.

Notes:
  - We do NOT softmax-normalize across (real_keys + null) by default in v1.
    Instead we compute attention weights with the null logit included in
    the softmax. This means null competes on equal footing with real
    items. If real items are weak, null wins; the hunter abstains.
  - The hunter's "stomach size" (top-K) is enforced *after* attention by
    keeping the top-K real items by attention weight (excluding null) and
    re-mixing their values weighted by their attention. Items beyond top-K
    are treated as rejected for decomposer attribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChannelOutput:
    output: torch.Tensor              # [out_seq_len, hidden_size]
    null_prob: float                  # softmax mass that landed on null
    selected_indices: list[int]       # indices into the prey list that were "eaten"
    selected_weights: list[float]     # corresponding attention weights
    rejected_indices: list[int]       # diet-filtered prey not selected


class Channel(nn.Module):
    """Per-edge cross-attention with null abstain slot.

    Args:
      hidden_size: d_model of both prey and hunter (same model in v1.5).
      n_heads:     attention heads.
      out_seq_len: how many "broadcast vectors" the channel emits to the
                   hunter (its prefix length, after role_prefix). Default 1.
      capacity:    stomach size (max number of prey items the hunter
                   actually consumes after attention). Items beyond this
                   are recorded as rejected even if their weight was
                   nonzero.
      null_temperature: scalar multiplier on the null logit before
                   softmax. Higher → null harder to win → less abstaining.
                   Initialized to 0 (null competes neutrally) so we can
                   observe baseline abstention behavior with identity W's.
    """

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 8,
        out_seq_len: int = 8,
        capacity: int = 6,
        null_temperature: float = 0.0,
        target_norm: float = 1.0,
        proj_inner: int | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        assert hidden_size % n_heads == 0, (hidden_size, n_heads)
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.out_seq_len = out_seq_len
        self.capacity = capacity
        self.target_norm = target_norm

        # Local generator for reproducible init. Doesn't touch global RNG state.
        gen = torch.Generator()
        if seed is not None:
            gen.manual_seed(int(seed))
        else:
            gen.seed()

        # W_Q / W_K initialized small (not identity) so attention scores at
        # init are near-zero and softmax is uniform — null wins lightly,
        # real keys compete on equal terms. Identity-init blew up downstream
        # in fp16 because producer hidden states have large native magnitude.
        self.W_Q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size, bias=False)
        # W_V initialized as scaled identity so the channel passes value
        # information through (slightly damped) at init.
        self.W_V = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            std_qk = 1.0 / hidden_size ** 0.5
            self.W_Q.weight.copy_(torch.empty_like(self.W_Q.weight).normal_(0.0, std_qk, generator=gen))
            self.W_K.weight.copy_(torch.empty_like(self.W_K.weight).normal_(0.0, std_qk, generator=gen))
            self.W_V.weight.copy_(torch.eye(hidden_size) * 0.5)

        # Null slot. Initialized small-random so it competes weakly at first.
        self.K_null = nn.Parameter(
            torch.empty(hidden_size).normal_(0.0, 0.02, generator=gen)
        )
        self.V_null = nn.Parameter(
            torch.empty(hidden_size).normal_(0.0, 0.02, generator=gen)
        )
        # Learned scalar that biases the null logit (higher → less abstain).
        self.null_bias = nn.Parameter(torch.tensor(float(null_temperature)))

        # Output projection: small MLP from attended hidden → out_seq_len
        # distinct vectors. A single linear ⇒ all out_seq_len positions are
        # rank-1 copies of one direction; an MLP per-position lets each
        # position carry distinguishable content. Inner width defaults to
        # 2x hidden for adequate capacity.
        inner = proj_inner if proj_inner is not None else hidden_size * 2
        self.proj_in = nn.Linear(hidden_size, inner, bias=False)
        self.proj_out = nn.Linear(inner, hidden_size * out_seq_len, bias=False)
        with torch.no_grad():
            self.proj_in.weight.copy_(
                torch.empty_like(self.proj_in.weight).normal_(0.0, 1.0 / hidden_size ** 0.5, generator=gen)
            )
            self.proj_out.weight.copy_(
                torch.empty_like(self.proj_out.weight).normal_(0.0, 1.0 / inner ** 0.5, generator=gen)
            )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, hidden] -> [n_heads, N, head_dim]
        N, H = x.shape
        return x.view(N, self.n_heads, self.head_dim).transpose(0, 1)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: [n_heads, N, head_dim] -> [N, hidden]
        return x.transpose(0, 1).contiguous().view(x.shape[1], self.hidden_size)

    def forward(
        self,
        hunter_state: torch.Tensor,    # [hidden] or [Q, hidden]
        prey_states: torch.Tensor,     # [M, hidden] — M prey items
    ) -> ChannelOutput:
        if hunter_state.dim() == 1:
            hunter_state = hunter_state.unsqueeze(0)  # [1, hidden]
        Q_in = hunter_state                                   # [Q, H]
        M = prey_states.shape[0]
        H = self.hidden_size

        # Pad to a single hunter query vector for v1 simplicity.
        # (Multi-query hunters are a v2 concern.)
        if Q_in.shape[0] != 1:
            Q_in = Q_in.mean(dim=0, keepdim=True)

        Q = self.W_Q(Q_in)                                    # [1, H]
        if M > 0:
            K_real = self.W_K(prey_states)                    # [M, H]
            V_real = self.W_V(prey_states)                    # [M, H]
        else:
            K_real = torch.zeros(0, H, device=Q_in.device, dtype=Q_in.dtype)
            V_real = torch.zeros(0, H, device=Q_in.device, dtype=Q_in.dtype)

        # Append null slot.
        K_all = torch.cat([K_real, self.K_null.to(Q_in).unsqueeze(0)], dim=0)  # [M+1, H]
        V_all = torch.cat([V_real, self.V_null.to(Q_in).unsqueeze(0)], dim=0)  # [M+1, H]

        # Multi-head attention.
        Q_h = self._split_heads(Q)        # [n_heads, 1, head_dim]
        K_h = self._split_heads(K_all)    # [n_heads, M+1, head_dim]
        V_h = self._split_heads(V_all)    # [n_heads, M+1, head_dim]

        scores = torch.einsum("hqd,hkd->hqk", Q_h, K_h) / math.sqrt(self.head_dim)  # [heads,1,M+1]
        # Bias the null logit (last index) by the learned scalar, averaged
        # across heads to keep semantics interpretable.
        scores[..., -1] = scores[..., -1] + self.null_bias

        weights = F.softmax(scores, dim=-1)           # [heads, 1, M+1]
        attended = torch.einsum("hqk,hkd->hqd", weights, V_h)  # [heads, 1, head_dim]
        merged = self._merge_heads(attended)          # [1, hidden]

        # MLP projects to out_seq_len distinct vectors so the model has
        # multiple positions of bandwidth to anchor on, not a single
        # mystery vector. GELU between the two linears.
        z = F.gelu(self.proj_in(merged))                       # [1, inner]
        out = self.proj_out(z)                                 # [1, hidden*out_seq_len]
        output = out.view(self.out_seq_len, self.hidden_size)  # [out_seq_len, hidden]

        # Aggregate weights across heads for reporting / capacity selection.
        avg_w = weights.mean(dim=0).squeeze(0)         # [M+1]
        null_prob = float(avg_w[-1].item())
        real_w = avg_w[:-1] if M > 0 else torch.zeros(0)

        if M > 0:
            sorted_w, sorted_idx = torch.sort(real_w, descending=True)
            keep_n = min(self.capacity, M)
            sel_idx = sorted_idx[:keep_n].tolist()
            sel_w = sorted_w[:keep_n].tolist()
            rej_idx = sorted_idx[keep_n:].tolist()
        else:
            sel_idx, sel_w, rej_idx = [], [], []

        # Norm-match each output position to target_norm so the spliced
        # vectors live in the same range Qwen's input embeddings do.
        cur_norm = output.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        output = output / cur_norm * self.target_norm

        return ChannelOutput(
            output=output,
            null_prob=null_prob,
            selected_indices=sel_idx,
            selected_weights=sel_w,
            rejected_indices=rej_idx,
        )
