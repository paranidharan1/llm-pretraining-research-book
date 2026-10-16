# Chapter 2.1: Rotary Position Embeddings (RoPE)

> **Upstream Dependencies:** $d_k$, base $\theta$, position index $t$ | **Downstream Effects:** 2.2 attention scores; long-context behavior | **MaxText Location:** `MaxText/layers/embeddings.py` (`RotaryEmbedding`)

*Unit 2 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Su et al. (2021) – RoFormer](https://arxiv.org/abs/2104.09864)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Attention is permutation-invariant without position; RoPE injects *relative* position by rotating $Q,K$.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Write the 2-D rotation $R(\phi)$ and show $R(m\theta)^\top R(n\theta)=R((n-m)\theta)$.
> 2. Derive $\langle R_m q,R_n k\rangle=q^\top R_{n-m}k$: score depends on offset only.
> 3. Define $\theta_i=\theta^{-2i/d_k}$ and explain the frequency spectrum (fast vs. slow dimensions).

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 2 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/layers/embeddings.py` (`RotaryEmbedding`). Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Applying RoPE to values or to the input embedding.
- Mismatched pairing convention (adjacent vs. half-split) when loading checkpoints.
- Assuming trained-length RoPE extrapolates.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Per-position validation loss beyond training length; long-context retrieval accuracy.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict what happens to per-position loss at $2\times$ the trained $T$ if you change the base $\theta$ from $10^4$ to $10^6$ at inference only.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
