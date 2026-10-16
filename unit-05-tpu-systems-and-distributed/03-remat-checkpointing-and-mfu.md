# Chapter 5.3: Rematerialization, Checkpointing and MFU

> **Upstream Dependencies:** HBM capacity, `remat_policy` | **Downstream Effects:** Step time, max $T$, per-device batch, hardware efficiency | **MaxText Location:** `config.remat_policy`; `max_utils.py` TFLOP calculator

*Unit 5 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Chen et al. (2016)](https://arxiv.org/abs/1604.06174)
- [Chowdhery et al. (2022) – PaLM](https://arxiv.org/abs/2204.02311)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Activations dominate memory at long $T$; recomputing them trades FLOPs for memory, allowing larger batches.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive activation memory per layer $\approx c\cdot T\cdot d\cdot B$ bytes and total $O(L)$; checkpointing reduces to $O(\sqrt L)$.
> 2. Derive MFU $=\dfrac{6N_{\text{active}}\cdot\text{tokens/s}+\text{attention FLOPs}}{n_{\text{chips}}\cdot\text{peak}}$.
> 3. Show full remat adds about one extra forward ($\approx+33\%$ compute) and why it is excluded from MFU.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 5 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `config.remat_policy`; `max_utils.py` TFLOP calculator. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Counting recompute as useful FLOPs.
- Using total instead of active parameters for MoE.
- Forgetting the attention FLOPs term at long $T$.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- MFU %; tokens/sec/chip; peak HBM; recompute share of step time.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Compute MFU for 3.0e5 tokens/s on 64 chips for a 1.4B-active MoE given a chip peak you look up; which remat policy would you try next?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
