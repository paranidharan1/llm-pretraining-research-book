# Chapter 4.1: Deduplication and Quality Filtering

> **Upstream Dependencies:** Raw web dumps | **Downstream Effects:** 4.2 packing; validation BPB; memorization | **MaxText Location:** Offline pipeline before Grain/TFRecord creation

*Unit 4 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Lee et al. (2022)](https://arxiv.org/abs/2107.06499)
- [Penedo et al. (2023) – RefinedWeb](https://arxiv.org/abs/2306.01116)
- [Penedo et al. (2024) – FineWeb](https://arxiv.org/abs/2406.17557)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Repeated text over-weights some content and enables memorization; low-quality text spends tokens on little signal.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive MinHash: $\Pr[h(A)=h(B)]=J(A,B)$.
> 2. Derive LSH banding: with $b$ bands of $r$ rows, collision probability $1-(1-s^r)^b$ and the resulting threshold $\approx(1/b)^{1/r}$.
> 3. Express effective epochs under duplication: a document duplicated $c$ times is seen $c$ times per epoch.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 4 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** Offline pipeline before Grain/TFRecord creation. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Dedup threshold too aggressive removing useful content.
- Train/eval overlap leakage.
- Quality classifier bias.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Retention rate; duplicate cluster-size histogram; train-vs-val gap; benchmark contamination checks.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict the effect on validation BPB of removing near-duplicates that made up 20% of the corpus at fixed token budget.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
