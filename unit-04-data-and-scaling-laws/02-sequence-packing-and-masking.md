# Chapter 4.2: Sequence Packing and Document Masking

> **Upstream Dependencies:** Document length distribution | **Downstream Effects:** 2.2 attention mask; effective tokens per batch; MFU | **MaxText Location:** `input_pipeline/_grain_data_processing.py`; `segmentation_ids`, positions

*Unit 4 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Raffel et al. (2020) – T5](https://arxiv.org/abs/1910.10683)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Fixed-shape batches are required on TPU; padding wastes FLOPs; packing fills the window but must not let documents attend to each other.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Compute padding waste for a length distribution and a window $T$.
> 2. Write the combined mask $M_{ij}=-\infty$ unless $j\le i$ and $\text{seg}_i=\text{seg}_j$.
> 3. Derive per-segment position reset for RoPE and loss masking at document boundaries.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 4 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `input_pipeline/_grain_data_processing.py`; `segmentation_ids`, positions. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Packing without segment masks.
- RoPE positions not reset per document.
- Loss normalized by padded length.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Padding waste %; fraction of tokens in truncated docs; loss at boundary positions.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict the BPB change from turning off the segment mask but keeping packing.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
