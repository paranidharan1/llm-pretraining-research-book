# Unit 3: Learning Dynamics and Stability

> **Book position:** Unit 3 of 5 · Previous: [Unit 2 – Transformer Engine](../unit-02-transformer-architecture/README.md) · Next: [Unit 4 – Data & Scaling Laws](../unit-04-data-and-scaling-laws/README.md)

## Unit Title & Ultimate Goal

**Ultimate Goal:** Diagnose why loss spikes happen, why BF16 overflows or swamps small updates, and how backpropagation Jacobians, gradient clipping, and AdamW states (`b1`, `b2`, `weight_decay`) govern weight trajectories.

**Why this matters for 2B Dense and 5B MoE runs.** A pretraining run is hundreds of thousands of optimizer steps on rented hardware; one unrecoverable divergence costs days of TPU time. The failure modes this unit explains:

- **Loss spikes.** Typically the Adam second-moment estimate $v_t$ lags behind a sudden gradient-magnitude increase (e.g., a router instability), producing an effective step far larger than intended.
- **Early divergence.** Too short a warmup means $v_t$ is miscalibrated while the loss landscape is sharpest.
- **BF16 swamping.** BF16 has only 8 bits of significand precision (7 stored), so a small update added to a large weight can round to zero; this is why master weights stay in FP32.
- **Silent decay mis-tuning.** Weight decay and learning rate jointly set the steady-state weight norm; changing one without the other changes the model's effective learning rate.

## How This Unit Connects to Previous & Next Units

**Upstream (from [Unit 2](../unit-02-transformer-architecture/README.md)).** The architecture defines the computational graph; this unit differentiates it. The residual stream's identity path (2.5) is what keeps $\partial \mathcal{L}/\partial x_0$ finite; the router (2.4) is what produces gradient outliers; the softmax/Z-loss gradient from [1.3](../unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) is the seed $\mathbf{p}-\mathbf{y}$ of the backward pass.

**Downstream (to [Unit 4](../unit-04-data-and-scaling-laws/README.md)).** The optimizer's behavior depends on batch size: the gradient noise scale (4.3) tells you how learning rate and batch size should co-scale, and the LR-schedule horizon must match the token budget chosen from scaling laws (4.4). Unit 5 consumes the optimizer state footprint: AdamW stores two FP32 moments per parameter, which, together with FP32 master weights, dominates per-chip HBM and motivates FSDP sharding.

## The Unit Dependency & Implementation Table

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **3.1 Backprop, Jacobians & Grad Clipping** | Applying the chain rule backward through the layers to compute $\mathbf{g} = \nabla_{\boldsymbol{\theta}} \mathcal{L}$, and rescaling $\mathbf{g}$ if $\lVert\mathbf{g}\rVert_2$ exceeds a threshold. | **Depends on:** forward activations & loss gradient.<br>**Affects:** protects weights from single bad data batches or router spikes. | Global L2 gradient norm (`grad_norm`) before & after clipping. | [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063) (clipping) | `train.py`: `jax.value_and_grad` & Optax `clip_by_global_norm`. |
| **3.2 AdamW (Adaptive Moments & Weight Decay)** | Maintaining a running average of gradient direction ($m_t$, controlled by $\beta_1$) and gradient squared magnitude ($v_t$, controlled by $\beta_2$), plus direct weight shrinkage ($\lambda$). | **Depends on:** gradients $\mathbf{g}_t$ and FP32 master weights.<br>**Affects:** step size per parameter and steady-state weight norm. | Parameter L2 norm (`param_norm`) & update-to-weight ratio. | [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980); [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) | `MaxText/optimizers.py` (`adam_b1`, `adam_b2`, `adam_weight_decay`). |
| **3.3 Warmup & LR Schedules (Cosine / WSD)** | Ramping learning rate $\eta_t$ linearly from $0$ (while Adam's $v_t$ calibrates) and decaying it smoothly to settle into a minimum. | **Depends on:** total token budget & batch size.<br>**Affects:** early divergence (if warmup is too short) & final validation loss. | Current learning rate $\eta_t$ vs. loss slope. | [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677); [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) | `max_utils.py`: `create_learning_rate_schedule`. |
| **3.4 Mixed Precision (BF16 vs FP32)** | Running matrix multiplies in 16-bit (bfloat16: 8-bit exponent, 7-bit mantissa) for TPU speed while keeping master weights & optimizer states in 32-bit (float32). | **Depends on:** TPU MXU hardware.<br>**Affects:** memory footprint, throughput, and rounding/underflow errors. | NaN/Inf occurrences, logit magnitude, weight-update swamping. | [Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) | `config.dtype="bfloat16"`, `config.weight_dtype="float32"`. |

> **Path caveat.** MaxText file/flag names drift between releases. Pin your MaxText commit in every experiment log.

## Primary Paper Lineage Breakdown

### Pascanu, Mikolov & Bengio (2013) – [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063)
- **Problem before:** Exploding gradients in deep/recurrent computation graphs caused catastrophic parameter jumps.
- **Intervention:** **Norm clipping**: $\mathbf{g}\leftarrow \mathbf{g}\cdot\min\!\left(1,\ \tau/\lVert\mathbf{g}\rVert_2\right)$; preserves direction, bounds step length.
- **Where in MaxText/JAX:** `optax.clip_by_global_norm` chained before the Adam transform (`gradient_clipping_threshold` in config).

### Kingma & Ba (2014) – [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
- **Problem before:** SGD with a single global learning rate handles badly scaled, sparse, or non-stationary gradients poorly.
- **Intervention:** Per-parameter step $\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ using bias-corrected EMAs of $g$ and $g^2$.
- **Where in MaxText/JAX:** `optimizers.py` → Optax `adamw` (`adam_b1`, `adam_b2`, `adam_eps`).

### Loshchilov & Hutter (2019) – [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- **Problem before:** L2 regularization added to the gradient gets rescaled by Adam's $1/\sqrt{v_t}$, so it does not behave like weight decay.
- **Intervention:** Decouple: $\theta\leftarrow\theta-\eta\left(\hat m/(\sqrt{\hat v}+\epsilon)+\lambda\theta\right)$.
- **Where in MaxText/JAX:** `adam_weight_decay` in `optimizers.py`.

### Goyal et al. (2017) – [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
- **Problem before:** Large-batch training diverged early or lost accuracy.
- **Intervention:** Linear LR scaling with batch size and **gradual linear warmup**.
- **Where in MaxText/JAX:** `create_learning_rate_schedule` (`learning_rate`, `warmup_steps_fraction`, `cosine_learning_rate_final_fraction`).

### Hoffmann et al. (2022) – [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) (schedule horizon)
- **Problem before:** Cosine schedules were often not matched to the actual number of training tokens, leaving loss on the table.
- **Intervention:** Showed that the cosine decay horizon must match the training token count (they note it should be $\approx$ the number of training steps).
- **Where in MaxText/JAX:** `steps` and the schedule horizon in config; this couples Unit 3 to Unit 4.4.

### Micikevicius et al. (2018) – [Mixed Precision Training](https://arxiv.org/abs/1710.03740)
- **Problem before:** FP32 training doubled memory and halved math throughput vs. 16-bit.
- **Intervention:** FP32 master weights, 16-bit forward/backward, and loss scaling for FP16. (BF16 shares FP32's exponent range, so loss scaling is generally unnecessary on TPU.)
- **Where in MaxText/JAX:** `dtype`, `weight_dtype`, and `matmul_precision` in config; the MXU accumulates in FP32.

## Chapter Navigation Links

| Chapter | File |
|---|---|
| 3.1 Backprop, Jacobians & Grad Clipping | [01-backprop-jacobians-grad-clipping.md](01-backprop-jacobians-grad-clipping.md) |
| 3.2 AdamW Moments & Weight Decay | [02-adamw-moments-weight-decay.md](02-adamw-moments-weight-decay.md) |
| 3.3 Warmup & LR Schedules | [03-warmup-and-lr-schedules.md](03-warmup-and-lr-schedules.md) |
| 3.4 Mixed Precision (BF16 / FP32) | [04-mixed-precision-bf16-fp32.md](04-mixed-precision-bf16-fp32.md) |

**Next:** [Unit 4 – Data & Scaling Laws →](../unit-04-data-and-scaling-laws/README.md)
