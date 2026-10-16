# Chapter 3.4: Mixed Precision: BF16 and FP32

> **Upstream Dependencies:** TPU MXU hardware | **Downstream Effects:** Memory footprint, throughput, rounding errors in all earlier chapters | **MaxText Location:** `config.dtype`, `config.weight_dtype`, `matmul_precision`

*Unit 3 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** 16-bit math is $\sim$2$\times$ cheaper in bandwidth and memory, but reduced precision loses small contributions.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive BF16 machine epsilon $2^{-7}$ vs. FP32 $2^{-23}$.
> 2. Derive the swamping condition $|\Delta|<\tfrac12\epsilon|w|$ and apply it to a typical update-to-weight ratio.
> 3. Show error of summing $n$ terms in low precision grows $O(n\epsilon)$; why accumulations are FP32.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 3 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `config.dtype`, `config.weight_dtype`, `matmul_precision`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Storing master weights in BF16.
- Router, norms, softmax, and loss reductions in BF16.
- Assuming BF16 needs loss scaling (it generally does not).

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- NaN/Inf counts; logit magnitude; update-to-weight ratio vs. $\epsilon_{\text{mach}}$.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** With $\eta=3\times10^{-4}$ and weights of magnitude $0.02$, would a typical Adam update survive being added to a BF16 weight?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
