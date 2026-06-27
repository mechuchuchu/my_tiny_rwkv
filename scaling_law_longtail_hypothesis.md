# Why Scaling Laws Work: A Long-Tail Absorption Hypothesis

> A conversation-derived hypothesis connecting power-law data distributions, long-tail knowledge absorption, and the validity of LLM scaling laws.

---

## Core Claim

> "An important reason scaling laws persist is that larger models absorb rare patterns and weak signals more effectively. As long as the data distribution remains sufficiently heavy-tailed and effective novelty continues to be supplied, scaling is likely to remain productive for a long time."

---

## The Argument Chain

### 1. Internet data follows a power-law distribution
Web documents, topics, code repositories, and language usage commonly exhibit Zipf/power-law properties.

**Important caveat:** frequency distribution ≠ information distribution. Rare does not always mean novel.
- 100k fanfic variations → tail, but low information density
- Duplicate GitHub forks → tail, but low novelty
- SEO pages → tail, but near-zero entropy

### 2. LLMs fail on the long tail
From Kandpal et al. (2023) *"Large Language Models Struggle to Learn Long-Tail Knowledge"*:
- QA accuracy strongly correlates with the number of relevant pre-training documents
- To reach human-level accuracy on facts appearing in <100 documents, models would need ~10^18 parameters (dense, extrapolated)
- This 10^18 estimate is based on dense architectures — sparse/MoE/linear-RNN may differ

### 3. Larger params → better tail absorption
More precisely:

$$\text{absorption ability} \propto \text{capacity} \times \text{optimization budget}$$

Not params alone. From Mohri et al. (2026) *"A Bitter Lesson for Data Filtering"*:
- Unfiltered Common Crawl (240T tokens) beats filtered datasets — but only with large enough models AND sufficient training steps
- Small models cannot cross this threshold regardless of data

### 4. Tail sources are growing rapidly — and they enforce novelty structurally
Key sources (arXiv, GitHub, AO3) are:
- **arXiv CC-BY**: ~3% in 2018 → ~40% in 2026 (empirically verified via OAI-PMH)
- **GitHub**: MIT/Apache dominant, open by default
- **AO3**: CC-based fan fiction community

These sources *structurally enforce novelty*:
- New arXiv papers must differ from prior work to be accepted
- New news is new by definition
- Creative writing generates novel combinations

**Caveat:** new document ≠ new information. Code reuse and self-citation are common. Tail generation rate ≠ effective information generation rate.

### 5. UncheatableEval implicitly measures long-tail absorption
UncheatableEval (Jellyfish042) uses post-cutoff data from arXiv, GitHub, AO3, BBC News — sources that:
- Are legally safe (CC/open licenses)
- Are rapidly growing
- Structurally enforce novelty
- Sit at the extreme tail of the power-law distribution

This means UncheatableEval simultaneously measures:
1. Contamination resistance (post-cutoff)
2. Long-tail absorption ability
3. True generalization (compression = intelligence)

The result that RWKV ≈ Transformer on this benchmark suggests tail absorption is primarily a scale phenomenon, not an architecture phenomenon.

### 6. Therefore: scaling law validity is partially grounded in tail dynamics
As long as:
- Data distribution remains heavy-tailed
- Effective novelty continues to be generated
- Models grow in capacity × compute

...scaling laws are likely to remain productive.

---

## Known Weaknesses

### The power-law truncation problem
Real distributions are truncated power laws:
$$P(X > x) \propto x^{-\alpha}$$
with finite support due to finite human population, finite knowledge production, and finite attention.

Even if the tail keeps growing, *effective novelty density* may decrease:
$$I(n) \sim \log n \quad \text{(diminishing returns)}$$

### The abstraction reuse problem
If a model has already learned "REST API design patterns," a million new REST API repos don't provide new information — they're recomposed from existing latent representations.

$$\Delta \text{performance} \rightarrow 0 \quad \text{as abstraction layers saturate}$$

**Counter-argument:** abstraction layers themselves shift over time. RWKV challenged the transformer paradigm. New architectural primitives generate genuinely new tail content that cannot be recomposed from prior abstractions.

### Scaling law has an irreducible floor
$$L(N) \approx A \cdot N^{-\alpha} + B$$

The $B$ term reflects irreducible noise, evaluation saturation, architecture limits, and optimization limits. Infinite scaling does not guarantee infinite improvement.

---

## Related Observations

- **Indirect long-tail absorption via citation**: Even paywalled papers (e.g., Lipinski's Rule of Five, cited 20,000+ times) get absorbed indirectly through CC-licensed papers that cite and re-express their content.
- **Human review as entropy stabilization**: Synthetic data without human review is "the same function repeated" — it cannot escape the model's own distributional boundary. Human review removes low-probability artifacts and injects out-of-distribution signal.
- **HuggingFace as a long-tail reservoir**: HF datasets with <10 downloads represent extreme tail knowledge. HF blog posts are CC0 by default (per ToS). The community generates genuine long-tail content daily.
- **RAG vs. parametric memory**: RAG retrieves but cannot recombine across tail sources. Parametric absorption enables cross-tail reasoning. Both are necessary; neither is sufficient alone.

---

## Summary

| Claim | Strength |
|-------|----------|
| Data follows power law | Strong |
| LLMs fail on long tail | Empirically proven |
| Params → tail absorption | True, but compute matters too |
| Tail sources growing + novelty enforced | Mostly true, with caveats |
| Power law persists indefinitely | Weak — likely truncated |
| Scaling law permanently valid | Too strong — "likely persistent" is more accurate |

---

## Source Conversation
Derived from a discussion between Seo Yeeun and Claude (Anthropic), June 27, 2026.

Key papers referenced:
- Kandpal et al. (2023). *Large Language Models Struggle to Learn Long-Tail Knowledge.* ICML 2023.
- Mohri, Duchi & Hashimoto (2026). *A Bitter Lesson for Data Filtering.* arXiv:2605.19407.
- Sutton (2019). *The Bitter Lesson.*
- dynomight (blog). *Scaling.*
