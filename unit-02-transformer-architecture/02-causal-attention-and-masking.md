# Chapter 2.2: Causal Attention and Masking

> **Upstream Dependencies:** 2.1 RoPE, segment IDs, $T$ | **Downstream Effects:** 2.3/2.4 MLP; memory $O(T^2)$ vs. flash kernels | **MaxText Location:** `MaxText/layers/attentions.py`; Splash/Flash kernels

*Unit 2 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Each token must gather information from earlier tokens, weighted by relevance; the mask enforces the factorization of 1.2.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive the $1/\sqrt{d_k}$ factor: for unit-variance independent entries $\text{Var}(q\cdot k)=d_k$.
> 2. Write the causal mask $M_{ij}=0$ if $j\le i$ else $-\infty$, and the combined segment mask for packed sequences.
> 3. Derive FLOPs $\approx 4T^2d$ per layer for scores and values, and the online-softmax identity used by flash-style kernels.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 2 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/layers/attentions.py`; Splash/Flash kernels. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- A fully masked row yields NaN softmax.
- Forgetting the segment mask when packing.
- Attention logit growth causing entropy collapse.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Attention entropy per head; max attention logit; fraction of heads attending mostly to the first token.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** If $d_k$ is doubled without changing the scale factor, how does the variance of the logits change and what happens to softmax entropy at init?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
