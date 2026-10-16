# Chapter 2.4: MoE Routing, Capacity and Auxiliary Loss

> **Upstream Dependencies:** Router logits, `capacity_factor`, 2.3 expert MLP | **Downstream Effects:** 3.1 gradient spikes; 5.2 All-to-All; $N_{\text{active}}$ in 4.4 | **MaxText Location:** `MaxText/layers/moe.py` (`capacity_factor`, `load_balance_loss_weight`)

*Unit 2 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Fedus et al. (2021) – Switch](https://arxiv.org/abs/2101.03961)
- [Zoph et al. (2022) – ST-MoE](https://arxiv.org/abs/2202.08906)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Decouple parameter count from per-token compute: 5B total parameters but only 1.4B touched per token.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Define router $r=\text{softmax}(xW_r)$, top-$k$ selection, and output $y=\sum_{i\in\text{top-}k}\tilde r_i E_i(x)$.
> 2. Derive the Switch aux loss $\alpha E\sum_i f_iP_i$ and show its minimum $\alpha$ at uniform load ($f_i=P_i=1/E$).
> 3. Derive capacity $=\lceil\text{cf}\cdot kT_{\text{tok}}/E\rceil$ and the expected drop rate under imbalance.
> 4. Derive $N_{\text{active}}$ vs. $N_{\text{total}}$ for $E$ experts, top-$k$, shared parameters.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 2 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `MaxText/layers/moe.py` (`capacity_factor`, `load_balance_loss_weight`). Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Judging an MoE by total parameters.
- Aux-loss weight too small (collapse) or too large (hurts quality).
- Router in BF16.
- Eval-time dropping differs from train-time.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Per-expert load histogram; router entropy; router max logit; router Z-loss; dropped-token %.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** With $E=64$, $k=2$ and cf $=1.25$, what fraction of tokens is dropped if one expert receives $3\times$ the mean load?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
