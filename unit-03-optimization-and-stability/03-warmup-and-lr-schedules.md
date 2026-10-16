# Chapter 3.3: Warmup and Learning-Rate Schedules

> **Upstream Dependencies:** Total token budget (4.4), batch size (4.3), Adam state (3.2) | **Downstream Effects:** Final validation loss; early stability | **MaxText Location:** `max_utils.py` `create_learning_rate_schedule`; `learning_rate`, `warmup_steps_fraction`

*Unit 3 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677)
- [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Early on, $v_t$ is poorly estimated and the loss surface is sharp; ramping LR avoids large mis-scaled steps. Decay lets the iterate settle.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Write linear warmup, cosine decay, and WSD formulas.
> 2. Show why a cosine horizon must equal total steps (area under the LR curve controls total 'movement').
> 3. Derive the linear-scaling heuristic and the $\sqrt{B}$ alternative for Adam; when each is justified.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 3 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `max_utils.py` `create_learning_rate_schedule`; `learning_rate`, `warmup_steps_fraction`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Resuming with a different horizon.
- Warmup specified in steps vs. fraction after changing batch size.
- Final LR too high or too low.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- LR vs. step; loss slope; grad norm during warmup.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** Predict final loss difference between a cosine schedule that ends at the planned step and one cut at 70% of its horizon.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
