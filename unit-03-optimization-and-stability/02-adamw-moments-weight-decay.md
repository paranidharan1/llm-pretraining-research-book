# Chapter 3.2: AdamW: Moments and Weight Decay

> **Upstream Dependencies:** Gradients $\mathbf{g}_t$, FP32 master weights | **Downstream Effects:** Weight norm; update size; 3.3 schedule; 5.1 optimizer-state memory | **MaxText Location:** `MaxText/optimizers.py` (`adam_b1`, `adam_b2`, `adam_weight_decay`)

*Unit 3 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980)
- [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Per-parameter adaptive step sizes make training robust to gradient scale differences across layers.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive bias correction: with $m_0=0$, $\mathbb{E}[m_t]=(1-\beta_1^t)\mathbb{E}[g]$.
> 2. Show $|\Delta\theta|\lesssim\eta$ per parameter, and why updates are roughly sign-like.
> 3. Derive a steady-state weight norm for sign-like updates of per-coordinate size $\eta$ on $n$ parameters: balance update growth $\eta^2 n$ against decay shrinkage $2\eta\lambda\lVert\theta\rVert^2$ to get $\lVert\theta\rVert\approx\sqrt{\eta n/(2\lambda)}$, and state when the approximation holds.
> 4. Show why L2-in-gradient differs from decoupled decay under Adam.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 3 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/optimizers.py` (`adam_b1`, `adam_b2`, `adam_weight_decay`). Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Large $\beta_2$ makes $v_t$ slow to react to gradient growth, so spikes.
- Applying decay to norms/biases.
- BF16 optimizer states.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- `param_norm`; update-to-weight ratio; $\sqrt{\hat v_t}$ statistics; effective LR $\eta/\sqrt{\hat v}$.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict the effect on spike frequency of changing $\beta_2$ from $0.999$ to $0.95$ at fixed everything else, and what it costs in noise.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
