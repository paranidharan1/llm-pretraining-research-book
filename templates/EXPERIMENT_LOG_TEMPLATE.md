# Experiment Log Template

Copy this file to `experiments/EXP-YYYYMMDD-NN-short-title.md` for every run. **Fill sections 1–7 before launching.** A result without a pre-registered hypothesis and promotion rule is an anecdote. [← Back to Root Index](../README.md)

---

## 1. Experiment ID & Title

| Field | Value |
|---|---|
| **Experiment ID** | `EXP-YYYYMMDD-NN` |
| **Title** | |
| **Author / Date** | |
| **Model family** | ☐ 2B Dense ☐ 5B MoE (1.4B active) |
| **MaxText commit / JAX / libtpu versions** | |
| **Config file / command line** | |
| **Parent (baseline) experiment ID** | |

## 2. Falsifiable Mechanistic Hypothesis

State a mechanism *and* a prediction that could be wrong.

> **Template:** "Changing **[X]** will change **[metric]** by **[direction and approximate magnitude]** because **[mechanism, referencing a chapter, e.g. 3.2 update-to-weight ratio]**. If **[metric]** does not move by at least **[threshold]**, the hypothesis is rejected."

- **Hypothesis:**
- **Mechanism (cite chapter / paper):**
- **A-priori prediction (written before the run):**
- **What result would falsify it:**

## 3. Independent Variable (Single Intervention)

Exactly **one** change relative to the baseline.

| Baseline value | Treatment value | Config flag / code diff |
|---|---|---|
| | | |

> If you need to change two things, run two experiments (or a factorial design with all four cells).

## 4. Controlled Invariants

These must be **identical** between baseline and treatment. Tick or record each.

| Invariant | Value | Identical? |
|---|---|---|
| $N_{\text{total}}$ (total params) | | ☐ |
| $N_{\text{active}}$ (active params per token) | | ☐ |
| $B_{\text{tokens}}$ (global batch in tokens) | | ☐ |
| $T$ (sequence length) | | ☐ |
| $V$ (vocabulary size / tokenizer hash) | | ☐ |
| Data order / data seed | | ☐ |
| Init seed | | ☐ |
| Dataset & mixture version | | ☐ |
| Schedule horizon (total steps $S$, warmup steps) | | ☐ |
| Peak LR, $\beta_1$, $\beta_2$, $\epsilon$, weight decay, clip threshold | | ☐ |
| Precision (`dtype`, `weight_dtype`, `matmul_precision`) | | ☐ |
| Mesh / sharding / remat policy (should not alter math; affects MFU only) | | ☐ |

> **If a tokenizer or $V$ differs, compare in BPB only** ([Chapter 1.4](../unit-01-objective-and-loss/04-bpb-and-perplexity.md)).

## 5. Primary & Diagnostic Metrics

| Role | Metric | Logged every | Baseline | Treatment | Δ |
|---|---|---|---|---|---|
| **Primary** | Validation BPB (held-out set, fixed bytes) | eval interval | | | |
| Diagnostic | Train loss (nats/token) | step | | | |
| Diagnostic | MFU %, tokens/sec/chip | step | | | |
| Diagnostic | `grad_norm` (pre-clip) and clip fraction | step | | | |
| Diagnostic | `param_norm`, update-to-weight ratio | step | | | |
| Diagnostic (MoE) | Router max logit, router Z-loss, load-balance loss | step | | | |
| Diagnostic (MoE) | Per-expert load, dropped-token % | step | | | |
| Diagnostic | Max attention logit / attention entropy | interval | | | |
| Diagnostic | Peak HBM (GB), exposed-communication % | once | | | |

## 6. Hidden Confounders Checklist

☐ Different tokenizer or vocabulary (use BPB)
☐ Different data order, shard assignment, or dataset epoch boundary
☐ Different init seed (estimate seed-to-seed noise: run baseline at ≥2 seeds)
☐ LR schedule horizon not matched to step count ([3.3](../unit-03-optimization-and-stability/03-warmup-and-lr-schedules.md))
☐ Batch size changed without LR re-tuning ([4.3](../unit-04-data-and-scaling-laws/03-critical-batch-size-noise-scale.md))
☐ Different effective precision (e.g., a norm or router silently in BF16 vs FP32)
☐ Resume from checkpoint with changed optimizer state
☐ Hardware nondeterminism / preemption / different slice topology
☐ Eval set leakage from training data (dedup check, [4.1](../unit-04-data-and-scaling-laws/01-deduplication-and-filtering.md))
☐ Evaluation on a different number of bytes/tokens between runs
☐ Treatment changes parameter count or FLOPs per token (compare at matched compute *and* matched steps)
☐ Seed noise estimate: $\sigma_{\text{seed}}=$ ______ BPB (treatment effect must exceed $\sim2\sigma_{\text{seed}}$)

## 7. Compute Cost & Early-Stop Criteria

| Item | Value |
|---|---|
| TPU type & slice (e.g., v5p-64) | |
| Planned steps / tokens | |
| Est. step time (s) and MFU | |
| **Planned cost (chip-hours)** = chips × hours | |
| **Budget cap (chip-hours)** | |

**Early-stop (kill) criteria — fixed in advance:**

- ☐ Loss is `NaN`/`Inf`, or spikes > ______ nats above the running baseline and fails to recover within ______ steps
- ☐ `grad_norm` > ______ × baseline for ______ consecutive steps
- ☐ Router max logit > ______ or any expert load < ______ % for ______ steps (MoE collapse)
- ☐ Validation BPB worse than baseline by > ______ at step ______ (futility)
- ☐ MFU < ______ % (systems misconfiguration; fix before spending more)

## 8. Quantitative Promotion Rule

Decide **before** the run what counts as a win.

> **Promote** treatment to the new baseline **iff**: (a) validation BPB improves by ≥ ______ (≥ 2 × seed noise) at the final step *and* at ≥ ______ intermediate evals; **and** (b) no stability regression (`grad_norm` spikes, router collapse) in diagnostic metrics; **and** (c) MFU/tokens-per-second is within ______ % of baseline *or* the BPB gain justifies the cost; **and** (d) the effect is not explained by a ticked confounder in §6.
>
> **Reject** otherwise. **Inconclusive** if the effect lies within seed noise: re-run with a second seed rather than promoting.

## 9. Result (fill after the run)

| Outcome | ☐ Promote ☐ Reject ☐ Inconclusive |
|---|---|
| Hypothesis confirmed / falsified? | |
| Prediction vs. observation (§2 vs §5) | |
| Surprises, instability events, step numbers | |
| Links: W&B / TensorBoard / XProf trace / checkpoints | |
| Follow-up experiments | |
