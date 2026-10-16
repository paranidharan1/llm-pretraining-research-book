# Chapter 1.3: Softmax, LogSumExp and Cross-Entropy

> **Upstream Dependencies:** Unembedding output (logits $\mathbf{z}_t$) | **Downstream Effects:** 3.1 backprop seed $\mathbf{p}-\mathbf{y}$; 3.4 BF16 stability; 1.4 | **MaxText Location:** `max_utils.py` cross-entropy helper; `config.z_loss`

*Unit 1 · [Unit README](README.md) · [Root Index](../README.md) · [Reading Library](../READING_LIBRARY.md)*

**Primary sources:**
- [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024)
- [Zoph et al. (2022) – ST-MoE](https://arxiv.org/abs/2202.08906)

---

## 1. First-Principles Intuition (Why This Exists)

> **Study guide:** Why exponentiate-and-normalize; why we never compute softmax then log; why large $\log Z$ is dangerous in low precision.

*To be filled in during the mentorship session.*

## 2. Formal Variable Definitions & Notation

> **Study guide:** List every symbol with shape and units (see [GLOSSARY.md](../GLOSSARY.md) for global notation).

| Symbol | Meaning | Shape / Units |
|---|---|---|
|  |  |  |

## 3. Line-by-Line Mathematical Derivation

> **Derivations to complete (no skipped algebraic steps):**
> 1. Derive the stable form $\ell=\text{LSE}(\mathbf{z})-z_y$ with $\text{LSE}=m+\ln\sum e^{z_j-m}$.
> 2. Derive $\partial\ell/\partial z_i=p_i-\mathbb{1}[i=y]$ step by step.
> 3. Derive the Z-loss gradient: $\partial(\text{LSE}^2)/\partial z_i=2\,\text{LSE}\cdot p_i$.
> 4. Show softmax is shift-invariant, so $\mathbf{z}$ is only identified up to a constant; why that lets $\log Z$ drift.

$$
\text{Begin derivation here: one algebraic step per line}
$$

## 4. Paper Lineage & Historical Tradeoffs

> **Study guide:** For each paper above, answer: (a) what problem existed before? (b) what exact mathematical/system intervention was introduced? (c) what tradeoff did it create? (See [Unit 1 README](README.md) for the lineage summary and [READING_LIBRARY.md](../READING_LIBRARY.md) for assumptions and limitations.)

| Paper | Problem before | Intervention | Tradeoff created |
|---|---|---|---|
|  |  |  |  |

## 5. MaxText / JAX / XLA Implementation Mapping

> **Starting pointer:** `max_utils.py` cross-entropy helper; `config.z_loss`. Pin the MaxText commit; map each symbol in §2 to a variable or config flag.

| Math symbol / equation | MaxText file : function / flag | Notes (dtype, sharding) |
|---|---|---|
|  |  |  |

## 6. Common Misconceptions & Failure Modes

- Computing $\log(\text{softmax}(z))$ naively.
- Taking the logsumexp in BF16 rather than FP32.
- Thinking Z-loss changes the argmax (it only controls scale).

*Add your own as they are encountered in runs.*

## 7. Measurable Diagnostic Signals in Training Logs

- Loss (nats/token); mean and max of $\text{LSE}(\mathbf{z}_t)$; max logit; Z-loss value.

| Metric | Healthy range | Failure signature | Where logged |
|---|---|---|---|
|  |  |  |  |

## 8. Controlled Thought Exercise & Ablation Prediction

> **Exercise:** With Z-loss weight $10^{-4}$ and typical $\text{LSE}\approx10$, what is the magnitude of the Z-loss term vs. the CE loss, and how does it change if $\text{LSE}$ doubles?
>
> Write your prediction *before* checking, using the format of the [experiment template](../templates/EXPERIMENT_LOG_TEMPLATE.md): single intervention, controlled invariants, expected metric change, falsification threshold.

**Prediction:**

**Result / correction:**

## 9. Personal Notes & Run Observations

*Run IDs, plots, surprises, open questions.*
