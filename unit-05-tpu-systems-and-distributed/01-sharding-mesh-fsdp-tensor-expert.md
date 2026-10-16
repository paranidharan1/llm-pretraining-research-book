# Chapter 5.1: Sharding: Mesh, FSDP, Tensor and Expert Parallelism

> **Upstream Dependencies:** TPU slice topology; 2.x layer shapes; 3.2 optimizer state | **Downstream Effects:** HBM per chip; 5.2 collectives; 5.3 MFU | **MaxText Location:** `sharding.py` / config `mesh_axes`, `logical_axis_rules`, `ici_*_parallelism`

*Unit 5 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Rajbhandari et al. (2020) – ZeRO](https://arxiv.org/abs/1910.02054)
- [Xu et al. (2021) – GSPMD](https://arxiv.org/abs/2105.04663)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** A model's weights, optimizer state and activations must fit in chip HBM; sharding is the mapping from tensor axes to device-mesh axes.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Compute bytes per parameter: FP32 weight 4 + grad 4 + Adam $m,v$ 8 = 16 B (before activations); total for 5B.
> 2. Show per-chip memory under FSDP over $n$ chips: $16N/n$ plus gathered layer buffers and activations.
> 3. Describe PartitionSpec for a $[d,d_{ff}]$ weight under (fsdp, tensor) and the matmul collectives needed.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 5 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `sharding.py` / config `mesh_axes`, `logical_axis_rules`, `ici_*_parallelism`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Confusing `fsdp` and `data` axes.
- Tensor parallel across slow DCN links.
- Forgetting activation memory.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Peak HBM; ICI link utilization; per-axis collective time.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Will a 5B MoE fit on 16 chips with 95 GB HBM each under pure FSDP at 8K sequence and per-device batch 4? Estimate state memory first.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
