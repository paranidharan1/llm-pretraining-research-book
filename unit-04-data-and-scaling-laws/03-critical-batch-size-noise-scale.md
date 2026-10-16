# Chapter 4.3: Critical Batch Size and Gradient Noise Scale

> **Upstream Dependencies:** Loss level, dataset diversity | **Downstream Effects:** Global batch size and LR scaling; 5.x per-device batch | **MaxText Location:** Global batch = `per_device_batch_size` $\times$ chips $\times$ `max_target_length`

*Unit 4 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** More batch reduces gradient variance, but only up to the point where noise is already smaller than signal.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Model minibatch gradient $G_B=G+\epsilon$, $\mathbb{E}\lVert G_B\rVert^2=\lVert G\rVert^2+\text{tr}(\Sigma)/B$.
> 2. Derive the optimal step and the tradeoff $\left(S/S_{\min}-1\right)\left(E/E_{\min}-1\right)=1$.
> 3. Derive an unbiased estimator of $B_{\text{noise}}$ from gradient norms at two batch sizes.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 4 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** Global batch = `per_device_batch_size` $\times$ chips $\times$ `max_target_length`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Treating $B_{\text{crit}}$ as constant (it rises as loss falls).
- Scaling batch without LR re-tuning.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Estimated $B_{\text{noise}}$ over training; steps-to-target vs. batch size.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** If $B_{\text{crit}}=2\text{M}$ tokens and you train at $8\text{M}$, roughly how many steps and how many tokens, relative to $B=B_{\text{crit}}$, are needed to hit the same loss?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
