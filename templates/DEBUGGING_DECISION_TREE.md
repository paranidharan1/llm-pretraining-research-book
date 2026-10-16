# Debugging Decision Tree

Diagnostic flowcharts for the four failure families of a 2B Dense / 5B MoE pretraining run on TPU: **loss spikes, NaNs/Infs, MoE collapse, and low MFU**. Each branch points to the chapter that explains the mechanism. The hypotheses are *ordered by prior likelihood and cost to check*, not guaranteed; always confirm with the logged metric before changing anything. [← Back to Root Index](../README.md)

> **Rule:** change one thing at a time, log it with the [experiment template](EXPERIMENT_LOG_TEMPLATE.md), and restart from the last healthy checkpoint rather than patching a corrupted optimizer state.

## 0. Triage

```mermaid
flowchart TD
    A[Something looks wrong] --> B{What do you see?}
    B -->|Loss jumps up then maybe recovers| S[Section 1: Loss spike]
    B -->|Loss or grad_norm is NaN / Inf| N[Section 2: NaN / Inf]
    B -->|Loss plateaus, experts unbalanced, dropped tokens| M[Section 3: MoE collapse]
    B -->|Loss fine but tokens/sec/chip or MFU is low| F[Section 4: Low MFU]
    B -->|Train loss much lower than val loss| D[Section 5: Data / leakage]
```

## 1. Loss Spike

```mermaid
flowchart TD
    S[Loss spike at step s] --> S1{grad_norm spiked BEFORE the loss?}
    S1 -->|No: loss spiked first| S2[Check data: bad batch, corrupted shard, repeated document - Ch 4.1, 4.2]
    S1 -->|Yes| S3{MoE model: router max logit or router Z-loss rising before spike?}
    S3 -->|Yes| S4[Router instability - go to Section 3, Ch 2.4]
    S3 -->|No| S5{Max attention logit growing for many steps?}
    S5 -->|Yes| S6[QK logit growth / attention entropy collapse: lower LR, QK-norm, check RoPE/softmax in FP32 - Ch 2.1, 2.2]
    S5 -->|No| S7{Does the spike come right after LR peak or warmup end?}
    S7 -->|Yes| S8[LR too high for batch size: lengthen warmup, lower peak LR - Ch 3.3, 4.3]
    S7 -->|No| S9{Adam beta2 large e.g. 0.999 and gradient scale changed suddenly?}
    S9 -->|Yes| S10[v_t lags gradient growth: lower beta2 e.g. 0.95, tighten clip - Ch 3.2, 3.1]
    S9 -->|No| S11[Check update-to-weight ratio, param_norm, and logit magnitude / Z-loss - Ch 1.3, 3.2, 3.4]
```

**Actions to try, in order:** (1) skip or re-shuffle the offending batch range and resume from the pre-spike checkpoint; (2) lower clip threshold; (3) lower $\beta_2$; (4) add/raise Z-loss ([1.3](../unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md)); (5) lower peak LR.

## 2. NaN / Inf

```mermaid
flowchart TD
    N[NaN or Inf appears] --> N1{First at step 0 or very early?}
    N1 -->|Yes| N2[Init scale, mask -inf handling, softmax of fully masked row, tokenizer produced out-of-range ID - Ch 2.2, 2.5, 1.1]
    N1 -->|No| N3{grad_norm huge for steps before NaN?}
    N3 -->|Yes| N4[Gradient explosion: clip, lower LR, check for logit growth - Ch 3.1]
    N3 -->|No| N5{Which tensor first goes non-finite? use jax debug_nans / per-layer stats}
    N5 -->|Logits or loss| N6[Compute LSE with max-subtraction in FP32; add Z-loss - Ch 1.3]
    N5 -->|Router logits / probs| N7[Router in FP32; add router Z-loss - Ch 2.4]
    N5 -->|Norm output| N8[epsilon too small, activations overflow BF16 range in residual stream - Ch 2.5, 3.4]
    N5 -->|Optimizer state| N9[Adam eps / v_t underflow, FP32 master weights missing - Ch 3.2, 3.4]
    N5 -->|Data| N10[Empty segment, divide-by-zero in loss normalization over masked tokens - Ch 4.2]
```

**Tip:** BF16 has FP32's exponent range, so overflow is rarer than in FP16, but BF16's ~3-digit precision causes *swamping* (small updates lost) and loss of accuracy in large reductions; check accumulations are FP32.

## 3. MoE Collapse (Router / Expert Imbalance)

```mermaid
flowchart TD
    M[Suspect MoE collapse] --> M1{Per-expert load: a few experts take most tokens?}
    M1 -->|Yes| M2{Load-balance aux loss weight too small, or aux loss not in the gradient path?}
    M2 -->|Yes| M3[Increase load_balance_loss_weight; verify aux loss is added to total loss - Ch 2.4]
    M2 -->|No| M4[Router logits grew large: add router Z-loss, FP32 router, lower router LR - Ch 2.4]
    M1 -->|No, balanced| M5{Dropped-token % high?}
    M5 -->|Yes| M6[Raise capacity_factor, or use a dropless implementation; check eval vs train mismatch - Ch 2.4]
    M5 -->|No| M7{Loss no better than a dense model at N_active?}
    M7 -->|Yes| M8[Experts not specializing: check router init, noise, expert capacity, data diversity - Ch 2.4, 4.1]
    M7 -->|No| M9[Healthy: monitor router entropy and expert load over time]
```

## 4. Low MFU / Low Throughput

```mermaid
flowchart TD
    F[MFU lower than expected] --> F1[Capture XProf trace for a few steady-state steps - Ch 5.2]
    F1 --> F2{Large gaps where MXU is idle?}
    F2 -->|Yes: waiting on collectives| F3{Which collective?}
    F3 -->|All-Gather / Reduce-Scatter| F4[FSDP gather not overlapped: tune fsdp axis size, per-device batch, collective-matmul flags - Ch 5.1, 5.2]
    F3 -->|All-to-All| F5[MoE dispatch dominates: rebalance expert vs fsdp axes, capacity, token layout - Ch 5.2, 2.4]
    F2 -->|No: MXU busy| F6{High recompute share?}
    F6 -->|Yes| F7[remat_policy too aggressive: save more, if HBM allows - Ch 5.3]
    F6 -->|No| F8{HBM near full forcing small per-device batch?}
    F8 -->|Yes| F9[Shard more, use remat, reduce activation memory - Ch 5.1, 5.3]
    F8 -->|No| F10[Host-bound input pipeline, small matmuls, unfused ops, non-multiple-of-128 dims - Ch 4.2, 5.3]
```

**Reminder:** MFU counts only useful model FLOPs ([PaLM](https://arxiv.org/abs/2204.02311)); remat raises hardware utilization but not MFU.

## 5. Train/Validation Gap or Suspiciously Good Loss

| Symptom | Likely cause | Check |
|---|---|---|
| Train loss ≪ val loss early | Duplicates between shards, memorization | [4.1](../unit-04-data-and-scaling-laws/01-deduplication-and-filtering.md) |
| Loss far below any plausible value | Target leakage: off-by-one shift or missing causal/segment mask | [1.2](../unit-01-objective-and-loss/02-autoregressive-factorization.md), [2.2](../unit-02-transformer-architecture/02-causal-attention-and-masking.md) |
| Run A "beats" run B but tokenizers differ | Per-token loss comparison | [1.4](../unit-01-objective-and-loss/04-bpb-and-perplexity.md) |
| Val BPB plateau while train falls | Overfitting, multi-epoch repetition | [4.4](../unit-04-data-and-scaling-laws/04-chinchilla-scaling-laws.md) |
