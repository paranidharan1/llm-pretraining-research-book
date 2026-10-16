# Chapter 2.5: Residual Stream and Pre-RMSNorm

> **Upstream Dependencies:** Initialization scale, depth $L$, sublayer outputs | **Downstream Effects:** 3.1 gradient flow; 3.4 BF16 range of activations | **MaxText Location:** `MaxText/layers/normalizations.py` (`RMSNorm`), `models.py` decoder layer

*Unit 2 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [He et al. (2015)](https://arxiv.org/abs/1512.03385)
- [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467)
- [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Depth makes optimization hard; the identity path keeps a direct route for signal and gradient from output to input.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive $\partial x_L/\partial x_l=\prod_{j=l}^{L-1}(I+\partial F_j/\partial x_j)$ and expand to show the identity term.
> 2. Show Pre-Norm residual variance grows roughly linearly with depth if sublayer outputs are uncorrelated.
> 3. Derive RMSNorm and its gradient; show scale invariance $\text{RMSNorm}(cx)=\text{RMSNorm}(x)$.
> 4. Derive a depth-scaled init, e.g. output projections scaled by $1/\sqrt{2L}$.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 2 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/layers/normalizations.py` (`RMSNorm`), `models.py` decoder layer. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Norm in BF16 (compute statistics in FP32).
- Pre-LN residual norm growth hurting late layers.
- Forgetting the final norm before unembedding.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Per-layer residual RMS; ratio of sublayer output norm to residual norm; gradient norm per layer.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** If each of $L=24$ sublayers adds an independent unit-variance vector, what is the residual variance at the output, and what init scaling would keep it at $O(1)$?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
