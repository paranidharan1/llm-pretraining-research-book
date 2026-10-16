# Chapter 2.3: SwiGLU Dense MLP

> **Upstream Dependencies:** Attention output (post residual + norm) | **Downstream Effects:** 2.5 residual stream; 2.4 (each expert is a SwiGLU MLP) | **MaxText Location:** `MaxText/layers/linears.py` (`MlpBlock`, `mlp_dim`)

*Unit 2 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Shazeer (2020) – GLU Variants](https://arxiv.org/abs/2002.05202)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** The MLP is where most parameters and per-token compute live; gating lets the layer modulate features multiplicatively.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Write $\text{SwiGLU}(x)=(\text{Swish}(xW_g)\odot xW_u)W_d$ with $\text{Swish}(u)=u\sigma(u)$.
> 2. Count parameters $3\,d\,d_{ff}$ and show the $2/3$ rule: $d_{ff}=\tfrac23\cdot4d$ matches a $8d^2$ ReLU FFN.
> 3. Derive FLOPs per token $6\,d\,d_{ff}$ (forward) and its share of $6N$.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 2 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/layers/linears.py` (`MlpBlock`, `mlp_dim`). Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Forgetting the $2/3$ width adjustment when comparing against GELU baselines.
- Dead gate units.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Activation RMS per layer; dead-neuron ratio (gate near 0 across a batch).

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** At $d=2048$, how many parameters does one SwiGLU block have if $d_{ff}=5504$? What fraction of a 2B model's parameters does this give across $L$ layers?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
