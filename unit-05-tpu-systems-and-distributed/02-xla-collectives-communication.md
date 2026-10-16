# Chapter 5.2: XLA Collectives and Communication

> **Upstream Dependencies:** Sharding annotations (5.1) | **Downstream Effects:** Exposed communication; 5.3 MFU | **MaxText Location:** XLA GSPMD pass; XProf trace

*Unit 5 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Lepikhin et al. (2020) – GShard](https://arxiv.org/abs/2006.16668)
- [Xu et al. (2021) – GSPMD](https://arxiv.org/abs/2105.04663)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** When a matmul needs data that lives on another chip, XLA inserts a collective; it is free only if hidden behind compute.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive ring All-Gather cost: $(n-1)/n\cdot S/\text{BW}$ for per-chip output size $S$.
> 2. Show All-Reduce $=$ Reduce-Scatter $+$ All-Gather, each moving $(n-1)/n\cdot S$.
> 3. Derive the arithmetic-intensity condition for overlap: compute time $\ge$ comm time as a function of per-device batch.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 5 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** XLA GSPMD pass; XProf trace. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Assuming compute and communication always overlap.
- All-to-All cost ignored in MoE.
- Reading step time instead of the trace.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- XProf: exposed collective time %, op breakdown, HBM bandwidth.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Estimate the minimum per-device batch (tokens) at which FSDP All-Gather of one layer hides behind its matmul on your chip (use peak FLOP/s and ICI bandwidth).
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
