# Chapter 1.4: Bits-Per-Byte and Perplexity

> **Upstream Dependencies:** 1.3 cross-entropy loss; 1.1 bytes-per-token | **Downstream Effects:** All ablation comparisons (Unit 4 scaling fits, experiment template) | **MaxText Location:** Evaluation loop in `train.py` / downstream eval scripts

*Unit 1 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Why per-token loss is not comparable across tokenizers and BPB is: it measures compression of the *same bytes*.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive $\text{BPB}=\dfrac{\sum_t\ell_t}{\ln2\cdot\text{Bytes}}=\dfrac{\bar\ell}{\ln 2}\cdot\dfrac{\text{tokens}}{\text{bytes}}$.
> 2. Derive $\text{PPL}=e^{\bar\ell}$ and interpret it as an effective branching factor.
> 3. Show two models with the same BPB but different tokenizers have different PPL.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 1 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** Evaluation loop in `train.py` / downstream eval scripts. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Comparing PPL across tokenizers.
- Counting bytes after normalization/BOS insertion inconsistently.
- Averaging per-document BPB instead of total nats over total bytes.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Validation BPB on a fixed byte set; bytes-per-token of the eval set.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Run A: $V=32\text{K}$, loss 2.40 nats/token, 4.0 bytes/token. Run B: $V=128\text{K}$, loss 2.70, 4.8 bytes/token. Which has better BPB?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
