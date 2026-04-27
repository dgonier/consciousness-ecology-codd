# Trough-as-Transformer: Architecture Specification

## Core Reframing

The "trough" at each trophic tier boundary **is** the attention operation — not a database that feeds into attention. The SubstratePool concept collapses into cross-attention mechanics with ecological lifecycle semantics on K/V membership.

### Geometry

At each tier boundary (producer→herbivore, herbivore→predator, predator→apex):

- **V matrix**: All outputs (hidden states) from the lower tier, stacked into a container matrix
- **K matrix**: Projections from the same lower-tier agents (what they "advertise")
- **Q matrix**: Projections from the upper-tier consumers (what they're looking for)

Standard cross-attention: `Attention(Q, K, V) = softmax(QK^T / τ) · V`

The full trophic chain is a stack of cross-attention layers:

```
input → [producer K/V trough] → herbivore Q attends → [herbivore K/V trough] → predator Q attends → [predator K/V trough] → apex Q attends → output
```

---

## Ecological Attention Mechanics

### Attention Distribution = Fitness Landscape

The softmax over Q·K for a given consumer query is a probability distribution over lower-tier organisms. This is not metaphor — it is literally a fitness measure:

- **High attention weight** = high fitness in the niche defined by that query
- **Aggregate desirability** = sum/mean of attention weights received across all consumers in a tick = ecological fitness
- No separate reputation table needed — reputation IS cumulative attention received

### Attention Decay (Selection Pressure)

Track cumulative attention per K slot across ticks:

```python
# Per K slot, rolling attention score
cumulative_attention[k_id] = α * cumulative_attention[k_id] + (1 - α) * tick_attention[k_id]

# Death threshold
if cumulative_attention[k_id] < ε for n consecutive ticks:
    kill(k_id)
    reallocate_slot(k_id)
```

A K that never gets attended to is an organism not participating in energy transfer. It dies. Its slot gets reallocated.

### Attention Temperature as Selection Pressure

Temperature τ on the softmax controls harshness of selection:

| τ | Distribution | Ecological Analog | Training Phase |
|---|---|---|---|
| High | Flat, egalitarian | All producers roughly equal | Early (exploration) |
| Low | Peaked, winner-take-all | Best K dominates, others starve | Late (exploitation) |

**Anneal τ during training**: start warm (diverse ecosystem), cool over time (sharpen selection). This is simulated annealing mapped onto ecological selection pressure — the math is identical.

---

## Slot Reallocation Strategies

When a K slot dies, four options with different dynamics:

### 1. Mutation (Asexual Reproduction)
Clone highest-fitness K in tier, add noise.
```python
new_k = best_k + ε * torch.randn_like(best_k)
new_v = best_v + ε * torch.randn_like(best_v)  # or keep V, mutate K only
```
- **Pro**: Simple, biologically grounded
- **Con**: Diversity collapse risk — converges to monoculture

### 2. Crossover (Genetic Algorithm)
Interpolate or component-wise crossover of two high-fitness K vectors.
```python
parent_a, parent_b = top_k_by_fitness(2)
mask = torch.rand_like(parent_a.k) > 0.5
new_k = torch.where(mask, parent_a.k, parent_b.k)
```
- **Pro**: More diversity than mutation
- **Con**: Linear interpolation in hidden-state space is geometrically questionable

### 3. Niche-Aware Spawning (Demand-Driven Speciation) ⭐ RECOMMENDED
Find underserved Q vectors (consumers with low max attention), initialize new K to be high-similarity with them.
```python
# Find the consumer Q with lowest max attention score
underserved_q = Q[torch.argmin(max_attention_per_q)]
# Initialize new K to serve that niche
new_k = project_to_k_space(underserved_q) + ε * torch.randn(...)
```
- **Pro**: Fills ecosystem gaps, directly addresses coverage
- **Con**: More complex, requires tracking per-Q satisfaction

### 4. Distillation (Population Knowledge Transfer)
New slot gets K initialized from fitness-weighted mixture of all current K vectors.
```python
weights = softmax(fitness_scores / τ_spawn)
new_k = (weights.unsqueeze(-1) * all_k_vectors).sum(0)
```
- **Pro**: New organism "knows" what the population knows
- **Con**: Converges toward mean, less explorative

---

## Multi-Head Attention as Emergent Dietary Niches

Use multi-head attention in the trough. Each head defines a different niche:

```
Head 1 → learns to route technical signals
Head 2 → learns to route fundamental signals
Head 3 → learns cross-domain correlations
Head n → ...
```

A producer's K gets projected into each head's subspace — it might be highly fit in head 1 (technical niche) and irrelevant in head 2 (fundamental niche).

**This replaces hardcoded `diet_tags` with differentiable specialization.**

| Property | Current (SQL diet_tags) | Multi-Head Trough |
|---|---|---|
| Specialization mechanism | Hardcoded categorical | Learned from gradients |
| Generalist organisms | Not representable | High attention across multiple heads |
| Specialist organisms | One tag per agent | Concentrated in one head |
| New niche discovery | Manual tag creation | Emerges during training |
| Differentiable | No | Yes |

---

## K/V Decomposition: Advertisement vs Nutritional Content

- **K** = what the producer "advertises" (determines whether consumer pays attention)
- **V** = actual content delivered when attention is paid

This creates natural pressure for **honest signaling**:

- K attracts consumer to useless V → predator's downstream score suffers → gradient penalizes misleading K
- K honestly represents V → consumer gets useful food → positive gradient reinforces K

If gradients DON'T flow back through K (detached for stability), dishonest signaling can persist. Attention decay partially corrects this (consumer Q drifts away from bad K over training), but it's slower than end-to-end gradients.

---

## Decomposer as Attention Bias

Instead of a separate `lessons` table, the decomposer produces a **bias vector** added to Q·K scores:

```python
# After apex judges prediction
attention_scores = Q @ K.T / τ

# Decomposer traces which producers contributed to good/bad predictions
decomposer_bias = decomposer(apex_judgment, broadcast_lineage)

# Bias modifies next tick's attention
biased_scores = attention_scores + decomposer_bias
weights = softmax(biased_scores)
```

This collapses global state into the attention mechanism:
- **Reputation** → attention bias magnitude
- **Lessons** → modifications to the bias
- **parsed_fields** → not needed if agents attend to V directly

---

## Cross-Trough Skip Connections

Full chain is deep: input → producer attn → herbivore attn → predator attn → apex. Deep attention stacks have gradient flow problems.

**Residual connections across tiers:**

```
apex_input = predator_output + α * skip(producer_output)
```

Ecologically: apex predator can observe the entire ecosystem, not just immediate prey tier.

- If skip weight α is low → intermediate tiers are adding useful abstraction
- If skip weight α is high → intermediate tiers are bottlenecking, skip is primary gradient path
- α is diagnostic: tells you about herbivore/predator quality

---

## Open Design Decisions

### 1. Gradient Flow Strategy

| Option | Mechanism | Training Signal for K/V | Stability | Richness |
|---|---|---|---|---|
| **End-to-end backprop** | Gradients flow through trough attention | Direct, fast | Lower (deep chain) | High |
| **Pure ecological** | Attention decay only | Indirect, slow | Higher | Lower |
| **Hybrid** | Gradients for V, ecological selection for K | K adapts by survival, V adapts by gradient | Medium | Medium |
| **Detached tiers** | Each tier boundary is a stop_gradient | Each tier trains on own loss | Highest | Each tier independent |

### 2. When Does Attention Decay Fire?

- **Every tick**: Very aggressive selection. Risk of premature convergence.
- **Every N ticks**: Buffered. Allows organisms time to find their niche.
- **Triggered by population saturation**: Only cull when the pool is full and a new organism wants to spawn.

### 3. Cross-Tick Persistence of V

- **Ephemeral**: V vectors cleared each tick. Fresh trough every cycle. Simplest.
- **Decaying**: V vectors persist but with exponential weight decay on their attention scores. Older food is less attractive.
- **Persistent until consumed**: V vectors live until claimed. Current SubstratePool behavior, now as attention entries.

### 4. Population Size

- **Fixed**: N slots per tier, always full. Dead slots immediately respawned. Simple.
- **Dynamic**: Population grows/shrinks based on ecosystem health metrics (total attention entropy, downstream task performance).
- **Carrying capacity**: Max N slots, but can go below. New organisms spawn only when fitness pressure warrants.

---

## Recommended First Implementation

**Combination of the most promising ideas, ordered by implementation difficulty:**

### Phase 1: Trough as Cross-Attention (Replaces SubstratePool)
- Stack producer hidden states into V matrix at tier boundary
- K and V projections are learned parameters
- Herbivore Q attends via standard cross-attention
- Softmax temperature τ as a hyperparameter
- Track per-K attention scores per tick

### Phase 2: Attention Decay + Slot Reallocation
- Rolling cumulative attention per K slot
- Death threshold ε with N-tick patience
- Niche-aware spawning for dead slots (demand-driven)
- Log attention distributions for diagnostics

### Phase 3: Multi-Head Dietary Niches
- Replace single-head with multi-head trough attention
- Remove hardcoded diet_tags
- Monitor per-head specialization (what emerges?)
- Measure specialist vs generalist distribution

### Phase 4: Temperature Annealing + Decomposer Bias
- Cosine or linear schedule for τ (warm→cool)
- Decomposer writes attention bias vector after apex judgment
- Skip connections across tier boundaries
- Full gradient flow analysis (where does learning happen?)

---

## Relationship to Current Codebase

| Current Component | Trough-as-Transformer Equivalent |
|---|---|
| `SubstratePool` (SQLite) | Cross-attention K/V matrix at tier boundary |
| `Broadcast.channel_embedding` | V vector (row in V matrix) |
| `diet_tags` + SQL filter | Multi-head attention head specialization |
| `claim()` semantics | Attention weight > threshold → consumed |
| `rotted` flag | Attention decay below ε → slot death |
| `decoded_text` | Debug annotation (unchanged, stored alongside) |
| `Channel` cross-attention | Becomes the trough attention itself (they merge) |
| `null_gate` | Abstention head or zero-attention output |
| Reputation table (proposed) | Cumulative attention score (emergent) |
| Lessons table (proposed) | Decomposer attention bias vector |
| `parsed_fields` (proposed) | May not be needed — agents attend to V directly |

---

## Key Insight

The trough doesn't *use* attention. The trough *is* attention — with lifecycle semantics (birth, competition, death, reproduction) on the K/V population that standard transformers don't have. Every ecological property (async operation, contested resources, agent provenance, population dynamics) maps to a modification of standard cross-attention mechanics rather than requiring a separate database layer.

The thing that makes this different from standard mixture-of-experts (which it superficially resembles) is that MoE has a fixed set of experts with a learned router. This has a **dynamic set of experts where membership is a learned outcome of the selection process**. Experts are born, compete, and die. That is a fundamentally different optimization dynamic.