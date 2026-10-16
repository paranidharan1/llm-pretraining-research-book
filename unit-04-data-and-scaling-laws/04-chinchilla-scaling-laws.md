# Chapter 4.4: Chinchilla and Compute-Optimal Scaling

> **Upstream Dependencies:** Architecture (dense vs. MoE), data quality | **Downstream Effects:** Token budget $D$, schedule horizon (3.3), pilot-run sizing | **MaxText Location:** Experiment planning and compute-budget arithmetic

*Unit 4 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361)
- [Hoffmann et al. (2022) – Chinchilla](https://arxiv.org/abs/2203.15556)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Given a fixed FLOP budget, how to split between model size and data to minimize loss.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive $C\approx6ND$: $2ND$ forward + $4ND$ backward.
> 2. Minimize $L=E+A/N^\alpha+B/D^\beta$ s.t. $6ND=C$ via Lagrange multipliers; derive $N_{\text{opt}}\propto C^{\beta/(\alpha+\beta)}$ and $D_{\text{opt}}\propto C^{\alpha/(\alpha+\beta)}$.
> 3. Explain IsoFLOP fits and why the $\approx20$ tokens/param rule follows from the fitted exponents.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 4 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** Experiment planning and compute-budget arithmetic. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Using total parameters for MoE in $C=6ND$ (should use $N_{\text{active}}$ for FLOPs).
- Treating compute-optimal as inference-optimal.
- Mismatched schedule horizon.

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- IsoFLOP minimum location; residual of $L(N,D)$ fit; loss vs. tokens at fixed $N$.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** With a 2B dense model, how many tokens does the 20:1 rule suggest, and what is the FLOP budget $C$? Repeat for $N_{\text{active}}=1.4\text{B}$.
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
