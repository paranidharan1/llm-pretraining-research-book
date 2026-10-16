# Chapter 1.2: Autoregressive Factorization

> **Upstream Dependencies:** Chain rule of probability; 1.1 tokenization | **Downstream Effects:** 2.2 causal mask; 1.3 loss; input/target shift in the data pipeline | **MaxText Location:** Input/target shift in `input_pipeline/`; causal mask in `layers/attentions.py`

*Unit 1 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Why model a joint distribution over $T$ tokens as $T$ conditional next-token problems; why this gives a tractable exact likelihood.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$ from the chain rule.
> 2. Show log-likelihood $=\sum_t\ln P(x_t\mid x_{<t})$ and that one forward pass yields $T$ training signals when a causal mask is used.
> 3. Show that the shifted-by-one target alignment is exactly the factorization (inputs $x_{1:T-1}$, targets $x_{2:T}$).

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 1 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** Input/target shift in `input_pipeline/`; causal mask in `layers/attentions.py`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Off-by-one shift leaks the answer and gives an absurdly low loss.
- Padding or cross-document tokens counted in the loss (see Ch 4.2).

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Loss at the first few positions vs. later positions (per-position loss); sudden near-zero loss indicates leakage.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** If the mask accidentally allowed attention to $t+1$, what would the loss at convergence look like and why?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
