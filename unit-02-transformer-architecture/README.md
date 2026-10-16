# Unit 2: The Transformer Engine

> **Book position:** Unit 2 of 5 · Previous: [Unit 1 – Objective & Loss](../unit-01-objective-and-loss/README.md) · Next: [Unit 3 – Optimization & Stability](../unit-03-optimization-and-stability/README.md)

## Unit Title & Ultimate Goal

**Ultimate Goal:** Master the forward-pass equations of modern Dense and Sparse (MoE) Transformers, understand how every layer transforms representations, why MoE routers collapse or drop tokens, and how normalization placement keeps deep networks trainable.

**Why this matters for 2B Dense and 5B MoE runs.** The 2B dense model and the 5B-total / 1.4B-active MoE differ almost entirely in one block: the feed-forward layer. Everything else (RoPE, causal attention, pre-norm residual stream) is shared. Failure modes if you skip this unit:

- **Position bugs:** a wrong RoPE base or dimension pairing trains fine at $T=2\text{K}$ and silently fails at long context.
- **Mask leakage:** a missing causal or segment mask gives an unrealistically low loss (Unit 1's off-by-one failure, in attention form).
- **Router collapse:** without a load-balancing loss, a handful of experts receive most tokens; the "5B" model behaves like a much smaller dense model while still paying 5B of memory.
- **Token dropping:** a too-small `capacity_factor` drops tokens at the expert boundary, producing a hidden train/eval mismatch.
- **Residual blow-up:** wrong norm placement or initialization makes activations grow with depth and training diverges at step 0.

## How This Unit Connects to Previous & Next Units

**Upstream (from [Unit 1](../unit-01-objective-and-loss/README.md)).** Token IDs in a vocabulary of size $V$ and windows of length $T$ enter the embedding table. The final hidden state is multiplied by the unembedding matrix to produce the logits $\mathbf{z}_t$ that Chapter 1.3 turns into a loss. The causal mask in 2.2 is the architectural enforcement of the factorization in 1.2.

**Downstream (to [Unit 3](../unit-03-optimization-and-stability/README.md)).** The architecture *defines the computational graph that backpropagation traverses*. The residual stream (2.5) decides whether gradient magnitudes survive $L$ layers; the router (2.4) is the single most spike-prone component and is the reason Z-loss and gradient clipping matter. Layer shapes also define the sharding axes in [Unit 5](../unit-05-tpu-systems-and-distributed/README.md): `mlp_dim`, `num_experts`, and heads map directly to the `tensor` and `expert` mesh axes.

## The Unit Dependency & Implementation Table

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **2.1 RoPE (Rotary Position Embeddings)** | Rotating query ($Q$) and key ($K$) vectors by angles proportional to their token position $t$ so attention knows the distance between words. | **Depends on:** head dimension $d_k$ & base frequency $\theta$.<br>**Affects:** long-context extrapolation (8K → 128K). | Long-context retrieval accuracy & per-position validation loss. | [Su et al. (2021) – RoFormer](https://arxiv.org/abs/2104.09864) | `MaxText/layers/embeddings.py` (`RotaryEmbedding`). |
| **2.2 Scaled Dot-Product Attention & Masking** | Computing relevance scores between tokens via $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{Mask}\right)V$. | **Depends on:** RoPE, segment IDs, sequence length $T$.<br>**Affects:** compute scales quadratically ($O(T^2)$) with sequence length. | Attention entropy & max attention logit. | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) | `MaxText/layers/attentions.py` & Splash/FlashAttention kernels. |
| **2.3 SwiGLU MLP (Dense Feed-Forward)** | Non-linear feature transformation using a gated projection: $\left(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\right)W_{\text{down}}$. | **Depends on:** attention output.<br>**Affects:** holds $\sim 2/3$ of parameters in a 2B dense model. | Activation RMS norm & dead-neuron ratio. | [Shazeer (2020) – GLU Variants](https://arxiv.org/abs/2002.05202) | `MaxText/layers/linears.py` (`MlpBlock`, `mlp_dim`). |
| **2.4 MoE Routing, Capacity & Aux Loss** | Replacing one big MLP with $E$ smaller expert MLPs, using a router to send each token to the top-$k$ experts (e.g., 1.4B active out of 5B total). | **Depends on:** router logits, `capacity_factor`.<br>**Affects:** token dropping, expert imbalance, and BF16 router spikes. | Load-balance aux loss, router Z-loss, dropped-token %. | [Fedus et al. (2021) – Switch](https://arxiv.org/abs/2101.03961); [Zoph et al. (2022) – ST-MoE](https://arxiv.org/abs/2202.08906) | `MaxText/layers/moe.py` (`capacity_factor`, `load_balance_loss_weight`). |
| **2.5 Residual Stream & Pre-RMSNorm** | Keeping a clean additive highway $x_{l+1} = x_l + \text{Layer}(\text{RMSNorm}(x_l))$ so gradients flow straight from output to Layer 1. | **Depends on:** initialization scale & depth $L$.<br>**Affects:** prevents vanishing/exploding gradients at step 0. | Layerwise residual-stream variance / RMS norm. | [He et al. (2015)](https://arxiv.org/abs/1512.03385); [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467); [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) | `MaxText/layers/normalizations.py` (`RMSNorm`) & `models.py`. |

> **Path caveat.** MaxText file/flag names drift between releases. Pin your MaxText commit in every experiment log.

## Primary Paper Lineage Breakdown

### He et al. (2015) – [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Problem before:** Plain deep networks degraded with depth (not only from overfitting, but from optimization difficulty).
- **Intervention:** Identity skip connections $x_{l+1} = x_l + F(x_l)$, so the Jacobian $\partial x_{l+1}/\partial x_l = I + \partial F/\partial x_l$ always contains an identity path.
- **Where in MaxText/JAX:** The decoder block in `layers/models.py` / `layers/llama2.py`-style decoder layers adds each sublayer output to the residual stream.

### Vaswani et al. (2017) – [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- **Problem before:** Recurrent models were sequential in $T$ and struggled with long-range dependencies.
- **Intervention:** Scaled dot-product multi-head attention $\text{softmax}(QK^\top/\sqrt{d_k})V$; the $1/\sqrt{d_k}$ factor keeps logit variance $\approx 1$ so softmax does not saturate. *Tradeoff:* $O(T^2)$ time and memory.
- **Where in MaxText/JAX:** `layers/attentions.py`; on TPU the quadratic score matrix is avoided via fused Splash/Flash-style Pallas kernels.

### Zhang & Sennrich (2019) – [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)
- **Problem before:** LayerNorm's mean-centering adds compute and a reduction without clear benefit.
- **Intervention:** $\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$: re-scaling invariance without re-centering.
- **Where in MaxText/JAX:** `layers/normalizations.py`. Computed in FP32 for stability even when activations are BF16.

### Xiong et al. (2020) – [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745)
- **Problem before:** Post-LN Transformers needed a careful learning-rate warmup; gradients at initialization are large near the output.
- **Intervention:** Showed Pre-LN gives well-behaved gradients at initialization and can train without warmup (in their setting), establishing the Pre-Norm default. *Tradeoff:* Pre-LN can yield representation growth with depth.
- **Where in MaxText/JAX:** Norm placement in the decoder layer in `layers/models.py` and per-model decoder files.

### Shazeer (2020) – [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202)
- **Problem before:** The standard ReLU/GELU FFN $W_2\,\sigma(W_1x)$ was the default with little exploration of gating.
- **Intervention:** $\text{FFN}_{\text{SwiGLU}}(x)=(\text{Swish}(xW)\odot xV)W_2$. To keep parameter count equal to a ReLU FFN, the hidden width is reduced by a factor $2/3$.
- **Where in MaxText/JAX:** `layers/linears.py` (`MlpBlock`), with fused gate/up projections controlled by `mlp_dim` and `mlp_activations`.

### Su et al. (2021) – [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- **Problem before:** Absolute embeddings added to inputs do not encode *relative* distance directly; learned tables fail beyond the trained length.
- **Intervention:** Rotate each 2-D pair of $q,k$ coordinates by angle $t\theta_i$, making $\langle R_m q, R_n k\rangle$ depend only on $m-n$.
- **Where in MaxText/JAX:** `layers/embeddings.py` (`RotaryEmbedding`), applied to $Q$ and $K$ inside attention, not to the input embeddings.

### Fedus, Zoph & Shazeer (2021) – [Switch Transformers](https://arxiv.org/abs/2101.03961)
- **Problem before:** Mixture-of-Experts with top-$k>1$ routing was complex and unstable.
- **Intervention:** Top-1 routing, a capacity factor $C = \lceil \text{cf}\cdot T_{\text{tokens}}/E\rceil$, and the load-balancing loss $L_{\text{aux}}=\alpha E\sum_{i=1}^E f_iP_i$ ($f_i$ = fraction of tokens dispatched to expert $i$, $P_i$ = mean router probability).
- **Where in MaxText/JAX:** `layers/moe.py` (`capacity_factor`, `load_balance_loss_weight`).

### Zoph et al. (2022) – [ST-MoE](https://arxiv.org/abs/2202.08906)
- **Problem before:** Sparse models were unstable at scale and in low precision.
- **Intervention:** Router Z-loss, FP32 router computation, and a systematic study of top-2 routing and capacity.
- **Where in MaxText/JAX:** Router computation in `layers/moe.py`; keep the router matmul and softmax in FP32 (check your `dtype`/`weight_dtype` settings).

## Chapter Navigation Links

| Chapter | File |
|---|---|
| 2.1 RoPE Positional Embeddings | [01-rope-positional-embeddings.md](01-rope-positional-embeddings.md) |
| 2.2 Causal Attention & Masking | [02-causal-attention-and-masking.md](02-causal-attention-and-masking.md) |
| 2.3 SwiGLU Dense MLP | [03-swiglu-dense-mlp.md](03-swiglu-dense-mlp.md) |
| 2.4 MoE Routing, Capacity & Aux Loss | [04-moe-routing-capacity-auxloss.md](04-moe-routing-capacity-auxloss.md) |
| 2.5 Residual Stream & RMSNorm | [05-residual-stream-and-rmsnorm.md](05-residual-stream-and-rmsnorm.md) |

**Next:** [Unit 3 – Optimization & Stability →](../unit-03-optimization-and-stability/README.md)
