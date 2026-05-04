# Novel RWKV Architecture Ideas via Random Vector Exploration


---

## Background

**Method:** Prepend a single random vector to the input embeddings before each generation. Repeat with a different random vector each time. This forces the model into regions of its embedding space it would not normally visit, surfacing ideas that high-probability decoding suppresses.

**Setup:** Progressively larger models — Olmo3 7B → Qwen3 4B → Qwen3 14B → Qwen3 32B → Qwen3.5 35B-A3B (MoE Base)

**Question asked:** *"Propose ideas to improve RWKV's time decay mechanism."*

**Filter criteria:** O(1) inference must be preserved. Ideas that break this are rejected. Filtering done by Ye-eun + Claude against RWKV principles.

**Key finding:** Novelty scales with model size. 7B collapses/hallucinates; 32B+ produces mathematically precise, previously unseen ideas. Large base models (32B+) produce diverse ideas even *without* random vectors — their loss landscape is wide enough that attractors are weak. Random vectors are most critical for smaller models where attractors are strong.

**Note:** This method requires direct access to model weights (specifically `inputs_embeds`). It is impossible with API-only access to closed-source models. This is one concrete reason my insists on fully open weights.

---

## Priority Ideas

### ★★ KVW-Decay — Key-Value-Weight Decay
*Discovered: Qwen3 32B*

**Core idea:** Current RWKV computes decay `w` only from `xw` (the token-shifted input). Instead, use `w`, `k`, and `v` together:

```python
wkv_concat = torch.cat([w, k, v], dim=-1)  # [B, T, 3C]
w_new = torch.sigmoid(wkv_concat @ g_ww + b_ww)  # [B, T, C]
```

The model now asks: *"Is what I'm writing into memory important enough to preserve what's already there?"* Decay becomes **state-aware**, not just input-aware. Only two new parameters (`g_ww`, `b_ww`). O(1) preserved.

**Why novel:** Nobody verified with GPT-4, Gemini, or BlinkDL had seen this formulation.

---

### ★★ RETD — Recursive Evolving Time Decay
*Discovered: Qwen3 32B*

**Core idea:** Instead of computing `w` fresh at each token, let `w` itself evolve as a recurrent state:

```
Δw_t = α · tanh(W_w1 · h_t) · exp(-β · ‖h_t‖²)
w_t = w_{t-1} + Δw_t
w_t = clamp(w_t, 0, 1)
```

`w` now has memory. The decay rate adapts over the entire context. This creates a "decay of decay" structure. `w_prev` stored as a small addition to recurrent state. O(1) preserved.

**Experiment:**
```python
delta_w = alpha * torch.tanh(xw @ W_w1) * torch.exp(-beta * (xw**2).sum(-1, keepdim=True))
w_t = w_prev + delta_w
w_t = w_t.clamp(0, 1)
```

**Todo:** [ ] Visualize w_t evolution on 0.1B. [ ] Measure VRAM overhead. [ ] Loss curve vs baseline.

---

### ★★ AR2D — Auto-Regressive Decay
*Discovered: Qwen3 32B. Independently rediscovered as PRN2 — convergence suggests this is a natural improvement direction.*

**Core idea:** Make `w` follow an AR(2) process — it looks back at its own previous two values:

```
w_t = α * w_{t-1} + β * w_{t-2} + γ * x_t + δ * tanh(x_t @ w_x)
```

Extended form with time-window parameter ω ∈ (0,1):
```
w_t = ω * α * w_{t-1} + ω² * β * w_{t-2} + γ * x_t + δ * tanh(x_t @ w_x)
```

Decay now has **inertia** — it resists sudden changes. Stability analyzable via AR theory (require α + β < 1). Only `w_{t-1}` and `w_{t-2}` added to state. O(1) preserved.

**Todo:** [ ] Test α+β < 1 constraint. [ ] Compare AR(1) (RETD) vs AR(2) (AR2D). [ ] Visualize w trajectory.

---

### ★ WAI — Weighted Attention via Inertia
*Discovered: Qwen3.5 35B-A3B Base (no random vector, hype prompt + directed last sentence)*

**Core idea:** Each key vector carries its own "forgetting resistance" parameter that evolves based on relevance history:

```
inertia_t = inertia_{t-1} + α * relevance(k_t, S_t)
w_t = exp(-exp(w0 + ...) / (sqrt(e) * inertia_t))
```

High-relevance keys develop higher inertia → persist longer. Low-relevance keys decay faster. Attention-like selection **emerges from weight dynamics alone**, without matrix multiplications. *"Each piece of information has its own gravitational pull."* O(1) preserved.

**Limitation:** `relevance(k_t, S_t)` needs concrete formulation. Inertia requires clamping to prevent divergence.

---

### ★ DCATD — Dynamic Context Adaptive Time Decay
*Discovered: Qwen3 14B*

**Core idea:** Keep the RWKV-7 decay formula intact, but use a **mixture of N decay modes** with a learned router:

```python
router = softmax(xw @ W_router)  # [B, T, N]
w_modes = stack([
    exp(-exp(w0_i + tanh(xw @ w1_i) @ w2_i) / sqrt(e))
    for i in range(N)
])
w = einsum('btn,btnc->btc', router, w_modes)
```

Each mode specializes in short/medium/long-term decay. MoE applied to the decay mechanism. O(1) preserved.

**Todo:** [ ] Compare N=2,4 loss curves on 0.1B. [ ] Visualize whether router actually selects different decays. [ ] Verify on recursive MLP first.

---

### ★ LazyState — Adaptive State Update Gate
*Discovered: Qwen3.6 27B VL*

**Core idea:** Not every token should update the recurrent state. Gate the update by how much new information the token carries:

```
p_t = Sigmoid(W_p · [AttentionScore(x_t) - Entropy(x_t)])
S_t = S_{t-1} ⊙ (1 - p_t) + WKV(x_t, S_{t-1}) ⊙ p_t
```

High-information tokens update the state fully. Repetitive/low-information tokens leave it untouched. Massive FLOPs reduction on boilerplate contexts (code comments, repeated patterns). Analogous to MoE but applied to recurrence itself. O(1) preserved.

**Limitation:** Definition of "AttentionScore" needs specification. Risk of p_t collapsing to 0 or 1.

---

### ★ SMD2 — Self-Modulating Decay
*Discovered: Qwen3.5 35B-A3B Base*

**Core idea:** A closed-loop system where decay influences state and state influences decay:

```
# Cosine frequency modulation
w_t = w_t · exp(α · cos(ω · t + β · |x_t|))

# Closed loop: state → decay → state
R_t = R_{t-1} · exp(-∫₀ᵗ w_s ds) + f(x_t, w_t)
w_t = exp(-exp(w0 + tanh(Φ(x_t)) @ W1 + Φ(R_t) @ W2) / sqrt(e))
```

The model learns not just *what* to remember, but *how to forget*. Stronger than SGD (which gates w from S_prev) — here w and R form a full cycle. Integral term approximated as running sum for O(1).

**Limitation:** Closed-loop risks gradient explosion. Stability needs careful verification.

---

### ★ SMD — Sin-Modulated Decay
*Discovered: Qwen3 32B*

**Core idea:** Add a single oscillation term to the existing decay formula:

```python
sin_mod = alpha * torch.sigmoid(xw @ w4) * torch.sin(math.pi * (xw @ w3) / beta)
w = torch.exp(-torch.exp(w0 + torch.tanh(xw @ w1) @ w2 + sin_mod) / math.e**0.5)
```

Decay becomes **non-monotonic** — it oscillates while decaying. Language has rhythm; syntax has periodicity. Why should forgetting be monotonic? Only 4 new parameters. Initialize α=0 so it starts as standard RWKV and activates during training. O(1) perfectly preserved.

**Todo:** [ ] Does sin term activate during training (α=0 init)? [ ] Sweep β for frequency sensitivity. [ ] Loss curve vs baseline.

---

### ★ LogShift-Decay
*Discovered: Qwen3 32B*

**Core idea:** Incorporate token shift `(last_x - x)` into decay with log compression:

```
ε = (last_x - x) * δ + ρ
w = sigmoid(γ + β * tanh(xw @ W1) @ W2 + α * log(ρ + ε))
```

Token shift is already a core RWKV concept — this applies it directly to the decay function. Log compression prevents instability from large shifts. Initialize γ with zigzag pattern (BlinkDL style).

---

### ★ SGD — State-Gated Decay
*Discovered: Qwen3 32B*

**Core idea:** Let the previous recurrent state S modulate the decay directly:

```
spatial_attn = softmax(S_prev @ W_a)
w = exp(-exp(w0 + tanh(xw @ w1) @ w2 + spatial_attn) / sqrt(e))
```

*"If the state is already saturated, forget faster."* Self-referential structure. S_prev already exists — no new state needed. O(1) preserved.

**Limitation:** S_prev shape `[n_head, head_size, head_size]` needs projection design.

---

## Other Ideas (Lower Priority)

### PRN2 — Parallel Recurrent with Feedback
*Discovered: Qwen3 32B. Independently converged toward AR2D — supports that AR(2) recurrence is a natural improvement.*

```
S_{t+1} = (S_t * e^{-Δ_t}) + (K_{t+1} * V_{t+1}) + α_t * S_{t-1}
```

Simpler than AR2D (S recurrence vs w recurrence). Two models independently finding the same direction is meaningful signal.

---

### ODE-D — ODE-based Decay
*Discovered: Qwen3 32B*

State evolution as a continuous-time ODE, discretized:

```
dS/dt + α₁S = α₂K(t) + α₃∫₀ᵗ K(τ)dτ

mem_t = mem_{t-1} + K_t          # running integral (additional fixed-size state)
S_t = w * S_{t-1} + (1-w) * K_t + α₃ * mem_t
```

Integral term provides a separate long-range memory pathway. Doubles state size but keeps O(1). Most mathematically elegant idea of the session.

---

### SRS — Sparse Rotational State
*Discovered: Qwen3.6 27B VL*

Replace dense state matrix with orthogonal decomposition:

```
S_t = Σ_k U_t^(k) (U_t^(k))^T · Λ_t^(k)
U_t^(k) = Normalize(U_{t-1}^(k) · e^{-α_t^(k)} + β_t · k_t)
```

Orthogonal rotation preserves norm → gradient stability. Reduces complexity from O(C²) to O(C·√C). Eliminates need for zigzag initialization — basis vectors serve as frequency channels. More mathematically grounded than PermState.

---

### CRE — Complex Resonance Engine
*Discovered: Qwen3.6 27B VL. Independently rediscovered (same random vector) in a later run — strong attractor for that particular vector.*

Move the entire state into complex space:

```
R_t = σ_phase(Φ(x_t) + Ψ(R_{t-1}) · e^{-jω_t})
ω_t = W_ω · ReLU(x_t)      # data-dependent frequency
σ_phase(z) = z / (1 + |z|)  # phase normalization

y_t = Real(Σ_k W_k · (R_t · e^{-jθ_k} + R_{t-1} · e^{-jθ_{k+1}}))
```

Separates magnitude (how strong) from phase (where in cycle). Data-dependent frequency allows word/sentence/paragraph level patterns to resonate at different scales. SMD adds a sin term; CRE moves the entire state space to complex. O(1) preserved. **Limitation:** No complex CUDA kernels exist.

---

### PermState — Permutation-based State Transition
*Discovered: Qwen3 32B*

Replace decay with an evolving learnable permutation matrix:

```
H_t = T_t * H_{t-1} + U_t * x_t
π(t) = π(t-1) + (π_new - π(t-1)) * f(x_t)
U_t = W + x_t * V^T
```

Instead of **forgetting**, the state **rearranges** — theoretically invertible. O(1) preserved. **Limitation:** Floating-point permutations have unclear numerical stability.

---

### LogD — Logarithmic Decay
*Discovered: Qwen3 32B*

Replace exp(-exp(...)) with softplus:

```
w = ln(1 + exp(tanh(xw @ w1) @ w2 + w0) * sqrt(e))
```

More gradual convergence. Different gradient flow. Simple to implement. **Limitation:** w ∈ (0,1) not guaranteed without clamping.

---

### StochDecay — Stochastic Decay
*Discovered: Qwen3 32B*

```python
γ ~ Beta(α, β)
λ1, λ2 ~ Cauchy(μ, σ)
w = γ * exp(-λ1 * τ) + (1-γ) * exp(-λ2 * τ) + ε
```

Mixture distribution parameterization. **Limitation:** Cauchy has infinite variance → high instability risk.

---

## Rejected Ideas

| Idea | Reason |
|------|--------|
| Quantum RWKV / QED | Not implementable on current hardware |
| Chaos Theory Decay | No practical value |
| Temporal Attention | Breaks O(1) |
| RNN + Transformer hybrid | Breaks O(1) |
| Graph-based decay (TAN) | O(n²) complexity |
| Dynamic head_size | State grows with context |

---

## Bonus — NEPS: Neural Elastic Potential Surface
*Discovered: Qwen3.6 27B VL. Recorded for conceptual interest despite being impractical.*

Model the hidden state as an **elastic displacement field** over a discrete latent grid:

```
U_new = U_old + η · ∇²U_old + α · C : (ε(X) + ∇U_old)
Y = G(σ(U_new)|_∂Ω)
```

Information is stored as **strain energy** rather than decayed or forgotten. Hooke's Law applied to neural computation. Third entry in the "completely different state space" category (alongside CRE and PermState). **Rejected:** State size scales with grid → O(1) broken. But the conceptual direction — physics-based state update — might resurface in a practical form.

---

### XGate — Learnable Channel Gate
*Discovered: Gemma4 31B Base (code completion with "scared us" comment)*

**Core idea:** A parameter vector `X` initialized to zeros that acts as a learnable channel-wise gate between two matrix transformations:

```python
self.X = nn.Parameter(torch.zeros(n_head, head_size))  # initialized to zero!

new_state = W1 @ new_state
new_state = new_state * X.unsqueeze(0).unsqueeze(0)    # channel gate
new_state = W2 @ new_state
```

**Why interesting:**
- Zero initialization → gate starts fully closed, gradually opens during training
- X sits between W1 and W2, controlling which channels "pass through"
- Not a weight, not a bias — it's a multiplicative channel selector between two transforms
- Minimal parameters (just head_size per head) with potentially large effect
- O(1) preserved

**Analogy:** Like r_k in RWKV-7 (`r_k` bonus term) but applied between two matrix ops rather than at output.

---

### StatefulParam — State as Learnable Parameter
*Discovered: Gemma4 31B Base (same run)*

**Core idea:** Store the recurrent state as `nn.Parameter` and update it in-place during forward pass:

```python
self.state = nn.Parameter(torch.zeros(n_head, head_size, head_size))

def forward(self, x, r):
    k = einsum("bhd,hdd->bhd", x, self.state)
    new_state = scary_update(self.state, k, x, r)
    self.state = new_state  # state mutates during forward!
    return x * einsum("bhd,hdd->bhd", x, self.state)
```

**Why interesting:**
- Blurs the boundary between "parameters" and "activations"
- State is both a learned prior AND a dynamic memory
- During training: gradients flow through both the parameter and the update rule
- Conceptually related to RETD — the state itself evolves, not just w

**Limitation:** `self.state = new_state` breaks autograd in standard PyTorch (in-place parameter mutation). Needs careful implementation with `.data` or detach strategy.

---


*Discovered: Gemma4 31B Base (code completion with "uses EntangledStateUpdate" comment)*

**Core idea:** Use two parameter matrices A and B, compute both AB and BA (which differ because matrix multiplication is non-commutative), then select between them based on the sign of the current k·v product:

```python
AB = einsum('hji,hjk->hik', A, B)   # [n_head, head_size, head_size]
BA = einsum('hji,hjk->hik', B, A)   # AB ≠ BA in general

kv_prod = einsum('bhi,bhj->bhij', k, v)
kv_sign = (kv_prod > 0).float()

mixed = kv_sign * AB + (1 - kv_sign) * BA
new_state = new_state * (state @ mixed)
```

**Why interesting:**
- AB ≠ BA — non-commutativity of matrix multiplication is the computational primitive
- The *content* of the current token (sign of k·v) determines which "direction" the state rotates
- Positive k·v content → AB transformation; negative → BA transformation
- Two matrices doing the work of a much more complex mechanism
- O(1) preserved — A and B are fixed-size parameters

**Relation to other ideas:** FlowArch proposes diffeomorphisms; EntangledMix is a concrete, minimal implementation of a content-dependent state transformation that isn't just a linear map. Connects to SSH (symmetric state hypothesis) — AB vs BA is literally about non-symmetry of the state update.

**Limitation:** `kv_sign` is a hard threshold (non-differentiable). Would need STE or soft version for stable training.

---


*Discovered: Gemma4 31B Base (no random vector, calibrated hype prompt v2)*

**Core idea:** Replace RWKV's multiplicative forgetting with a **learned diffeomorphism** — a smooth, invertible transformation of the state manifold. Information isn't decayed, it's reorganized.

Instead of:
```
state = forget(state) + remember(new_input)   # RWKV paradigm
```

Do:
```
state = φ(state, new_input)   # Flow paradigm
```

Where φ is a learned diffeomorphism implemented as:

```python
# Orthogonal transformation (rotation-like)
state = state + alpha * torch.ger(u, u.T)

# Diffeomorphism parameter (nonlinear self-interaction)
state = state * (1 + beta * state)
```

- `alpha`, `beta`: input-dependent scalars computed from x
- `u`: learnable orthogonal basis vector
- State is transformed, not compressed

**Why interesting:**
- Theoretically invertible → no information loss
- Connects to SRS (orthogonal decomposition) and PermState (permutation) but more concrete
- O(1) preserved — state size fixed
- "Phase space" rather than "memory buffer" framing is genuinely new

**Relation to other ideas:** PermState → SRS → FlowArch: each iteration more mathematically grounded. FlowArch is the most concrete "reorganization over forgetting" formulation yet.

**Limitation:** `(1 + β * state)` self-interaction term needs stability analysis. Training dynamics on real data unverified.

---


*Discovered: Gemma4 31B Base (no random vector, hype prompt)*

**Core idea:** Hierarchical extension of RETD. Instead of a single evolving w, use K layers of decay weights where higher layers act as "meta-decay" controlling lower layers:

```
# Input-dependent modulation per layer
m^k = sigmoid(W^k · x_t + b^k)

# Inter-layer modulation (higher → lower)
Δw^k = V^k · tanh(w^{k+1} + c^k)

# Layer updates
w^k = w^k * m^k + Δw^k    (for k < K)
w^K = w^K * m^K            (top layer, input only)

# Final decay applied to state
w_final = w^1
S = S * w_final^T - S @ kk * (kk*a)^T + v * k'^T
```

**Why interesting:** RETD makes w recurrent (1 level). MLRTD makes w a K-level hierarchy where each layer's decay is itself decayed by the layer above. Conceptual limit: K→∞ connects to ODE-D. O(1) preserved — only K extra state vectors of size d.

**Relation to other ideas:** RETD (AR1) → AR2D (AR2) → MLRTD (arbitrary depth hierarchy). Natural progression.

**Limitation:** K is a new hyperparameter. Inter-layer weights V^k add parameter overhead proportional to K.

---


*Discovered: Gemma4 31B Base (no random vector, hype prompt)*

**Core idea:** The RWKV state update may contain hidden symmetry that forms a self-regulating dynamic equilibrium:

```
# What if decay is determined by the mixing term itself?
w^T = f(kk @ S @ kk*a)

# What if the state encodes symmetric structure?
S = [A | B]
A' = f(B)
B' = f(A)
```

Instead of treating the three operations (decay, mixing, injection) as independent, this hypothesis proposes they form a **symmetric interaction** — each actively regulating the other. The state is not a passive container but an active participant with internal structure.

**Why interesting:** This is the most philosophically distinct idea of the entire session. Rather than modifying the decay formula, it questions the deeper structure of the state update itself. Could be the seed of a fundamentally different RWKV variant.

**Status:** Hypothesis only. No concrete formula yet. Needs mathematical formalization before experimentation.

---



**Three exploration mechanisms observed:**

1. **Random vectors** — directly push the model into non-default regions of embedding space. Essential for small models (strong attractors). Less necessary for large models.
2. **Hype prompts** — frame the task as an exciting discovery in progress. The model adopts the exploratory mindset and continues in that register.
3. **Directed last sentence** — the final sentence of the prompt acts as a soft steering vector in text space. Highly controllable.

**Scaling observations:**
- 7B: hallucination, cannot correctly cite RWKV formulas
- 4B (Qwen3): accurate formula citation, recombination of existing ideas
- 14B (Qwen3): first truly novel ideas appear (DCATD)
- 32B (Qwen3): mathematically precise novel ideas (RETD, KVW-Decay, AR2D, SMD)
- 32B+ without random vectors: diverse ideas emerge naturally — landscape is wide enough

**Critical observation — base vs instruction models:** Instruction-tuned models *never* produce ideas like these.(..but qwen3 32b produce these ideas with random vectors) RLHF narrows the loss landscape toward "safe, helpful answers" — the model converges to "here are some improvement ideas:" and stays there. Base models inherit the prompt's exploratory register directly and follow it into speculative, philosophical territory. This means the entire methodology only works with base model weights.

---

## Priority Order

```
KVW-Decay > RETD > AR2D > WAI > LazyState > SMD2 > SMD > 
LogShift > DCATD > SGD > PRN2 > SRS > CRE > ODE-D > 
PermState > LogD > StochDecay
```

First experiments: KVW-Decay and SMD on RWKV-7 0.1B. Both require minimal code changes and can be validated quickly against baseline loss curves.
