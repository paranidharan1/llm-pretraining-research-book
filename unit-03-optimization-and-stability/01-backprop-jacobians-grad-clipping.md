# Chapter 3.1: Backprop, Jacobians and Gradient Clipping

> **Upstream Dependencies:** Forward activations, loss gradient (1.3), residual structure (2.5) | **Downstream Effects:** 3.2 optimizer inputs; 4.3 noise scale | **MaxText Location:** `train.py` (`jax.value_and_grad`); Optax `clip_by_global_norm`

*Unit 3 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Gradients are the only signal the optimizer sees; one bad batch can produce a catastrophic update.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive reverse-mode accumulation through a linear layer: $\partial\mathcal{L}/\partial W=\delta x^\top$, $\partial\mathcal{L}/\partial x=W^\top\delta$.
> 2. Derive the Jacobian product for the residual chain and bound $\lVert\partial x_L/\partial x_l\rVert$.
> 3. Derive norm clipping and show it preserves direction and caps step length.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 3 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `train.py` (`jax.value_and_grad`); Optax `clip_by_global_norm`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Clipping per-parameter vs. globally.
- Clip threshold chosen so clipping triggers constantly (it then acts as an LR change).
- Reading post-clip norm as the true gradient norm.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Pre- and post-clip global `grad_norm`; clip fraction; per-layer gradient norms.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** If clip threshold $\tau=1$ and the typical pre-clip norm is $0.4$ with occasional $5\times$ spikes, what is the effective update scale on a spike step?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
