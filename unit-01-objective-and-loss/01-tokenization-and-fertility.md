# Chapter 1.1: Tokenization and Fertility

> **Upstream Dependencies:** Raw corpus statistics | **Downstream Effects:** 1.2 (T, V), 2.x embedding size, 1.4 (bytes-per-token) | **MaxText Location:** `assets/` tokenizer model; `input_pipeline/` preprocessing

*Unit 1 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Sennrich et al. (2016)](https://arxiv.org/abs/1508.07909)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Why a neural net cannot consume strings; why subwords beat words (open vocabulary) and characters (long $T$). What does vocabulary size $V$ trade off?

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Define BPE merge procedure as a greedy compression algorithm; compute one merge by hand on a toy corpus.
> 2. Derive embedding + unembedding parameter count $2Vd$ (or $Vd$ if tied) and show it as a fraction of $N$ for the 2B model.
> 3. Derive bytes-per-token $=\text{bytes}/T$ and fertility; show tokens needed to cover a fixed byte budget scales as $1/\text{bytes-per-token}$.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 1 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `assets/` tokenizer model; `input_pipeline/` preprocessing. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Larger $V$ always better (it raises softmax cost and embedding memory, and rare-token embeddings are under-trained).
- Comparing loss across tokenizers (see Ch 1.4).

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Fertility per language/domain on a held-out sample; bytes-per-token; fraction of tokens that are never or rarely seen in training (under-trained embeddings).

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict how train loss (nats/token) and BPB change if $V$ doubles at fixed corpus and model. Which changes, which should not?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
