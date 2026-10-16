# Master Glossary

Equations, metrics, and system concepts used throughout the book. Entries link to the chapter where they are derived and to the paper (see [READING_LIBRARY.md](READING_LIBRARY.md)) where they originate. [← Back to Root Index](README.md)

## Notation

| Symbol | Meaning | Typical value / note |
|---|---|---|
| $V$ | Vocabulary size | 32K–256K |
| $T$ | Sequence (context window) length in tokens | 2K–8K for pretraining |
| $d$ or $d_{\text{model}}$ | Hidden / residual-stream width | |
| $d_k$ | Per-head key/query dimension | |
| $L$ | Number of transformer layers | |
| $E$, $k$ | Number of experts, experts activated per token | MoE |
| $N_{\text{total}}$, $N_{\text{active}}$ | Total and per-token-active parameters | 5B / 1.4B for the MoE |
| $D$ | Training tokens | |
| $C$ | Training compute in FLOPs | $C\approx 6\,N_{\text{active}}\,D$ |
| $B$, $B_{\text{tokens}}$ | Batch size in sequences / in tokens | $B_{\text{tokens}}=B\cdot T$ |
| $\mathbf{z}_t$ | Logits at position $t$ | $\in\mathbb{R}^V$ |
| $\mathbf{p}_t$, $\mathbf{y}_t$ | Predicted distribution, one-hot target | |
| $\mathbf{g}$ | Gradient $\nabla_{\boldsymbol\theta}\mathcal{L}$ | |
| $\eta_t$ | Learning rate at step $t$ | |

## A. Information Theory and Objective (Unit 1)

| Term | Definition / Equation | Chapter |
|---|---|---|
| **Entropy** | $H(p)=-\sum_x p(x)\ln p(x)$ | [1.2](unit-01-objective-and-loss/02-autoregressive-factorization.md) |
| **Cross-entropy** | $H(p,q)=-\sum_x p(x)\ln q(x)=H(p)+D_{\text{KL}}(p\Vert q)$ | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **Autoregressive factorization** | $P(\mathbf{x})=\prod_{t=1}^{T}P(x_t\mid x_{<t})$ | [1.2](unit-01-objective-and-loss/02-autoregressive-factorization.md) |
| **Softmax** | $p_i=\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **LogSumExp** | $\text{LSE}(\mathbf{z})=m+\ln\sum_j e^{z_j-m}$, with $m=\max_j z_j$ (numerically stable) | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **NLL / cross-entropy loss** | $\ell_t=-\ln p_{t,x_t}=\text{LSE}(\mathbf{z}_t)-z_{t,x_t}$ | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **Loss gradient w.r.t. logits** | $\partial\ell_t/\partial\mathbf{z}_t=\mathbf{p}_t-\mathbf{y}_t$ | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **Z-loss** | $\mathcal{L}_z=\lambda_z\,(\text{LSE}(\mathbf{z}_t))^2$; penalizes drift of the log-partition | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| **Fertility** | tokens per word (or per character) under a tokenizer | [1.1](unit-01-objective-and-loss/01-tokenization-and-fertility.md) |
| **Bytes-per-token** | $\text{UTF-8 bytes}/\text{tokens}$; the exchange rate between nats/token and BPB | [1.1](unit-01-objective-and-loss/01-tokenization-and-fertility.md) |
| **Bits-per-byte (BPB)** | $\text{BPB}=\dfrac{\text{Total nats}}{\ln 2\cdot\text{Bytes}}=\dfrac{\bar\ell}{\ln 2}\cdot\dfrac{\text{tokens}}{\text{bytes}}$ | [1.4](unit-01-objective-and-loss/04-bpb-and-perplexity.md) |
| **Perplexity** | $\text{PPL}=e^{\bar\ell}$ (tokenizer-dependent; effective branching factor) | [1.4](unit-01-objective-and-loss/04-bpb-and-perplexity.md) |

## B. Architecture (Unit 2)

| Term | Definition / Equation | Chapter |
|---|---|---|
| **RoPE** | Rotate each coordinate pair $(q_{2i},q_{2i+1})$ by angle $t\,\theta_i$, $\theta_i=\theta^{-2i/d_k}$ (base $\theta$, often $10^4$ or larger) | [2.1](unit-02-transformer-architecture/01-rope-positional-embeddings.md) |
| **Scaled dot-product attention** | $\text{softmax}\!\left(\dfrac{QK^\top}{\sqrt{d_k}}+M\right)V$, with $M_{ij}=-\infty$ for $j>i$ | [2.2](unit-02-transformer-architecture/02-causal-attention-and-masking.md) |
| **Segment mask** | additional mask $M_{ij}=-\infty$ when positions $i,j$ lie in different packed documents | [2.2](unit-02-transformer-architecture/02-causal-attention-and-masking.md), [4.2](unit-04-data-and-scaling-laws/02-sequence-packing-and-masking.md) |
| **SwiGLU** | $\big(\text{Swish}(xW_{\text{gate}})\odot xW_{\text{up}}\big)W_{\text{down}}$, $\text{Swish}(u)=u\,\sigma(u)$ | [2.3](unit-02-transformer-architecture/03-swiglu-dense-mlp.md) |
| **Router** | $\mathbf{r}=\text{softmax}(xW_r)\in\mathbb{R}^E$; select top-$k$ experts, combine outputs weighted by $r_i$ | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| **Capacity** | per-expert token limit $\lceil \text{cf}\cdot kT_{\text{tok}}/E\rceil$; tokens beyond it are dropped | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| **Load-balance aux loss** | $\mathcal{L}_{\text{aux}}=\alpha E\sum_{i=1}^{E}f_iP_i$ ($f_i$ token fraction, $P_i$ mean router prob) | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| **Router Z-loss** | $\mathcal{L}_{z}=\dfrac{1}{B}\sum_{b}\Big(\ln\sum_{i}e^{r_{b,i}}\Big)^2$ over router logits | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| **RMSNorm** | $\dfrac{x}{\sqrt{\frac1d\sum_i x_i^2+\epsilon}}\odot g$ | [2.5](unit-02-transformer-architecture/05-residual-stream-and-rmsnorm.md) |
| **Pre-Norm residual block** | $x_{l+1}=x_l+F_l(\text{RMSNorm}(x_l))$ | [2.5](unit-02-transformer-architecture/05-residual-stream-and-rmsnorm.md) |

## C. Optimization and Stability (Unit 3)

| Term | Definition / Equation | Chapter |
|---|---|---|
| **Global grad norm** | $\lVert\mathbf{g}\rVert_2=\sqrt{\sum_{\text{all params}}g_i^2}$ | [3.1](unit-03-optimization-and-stability/01-backprop-jacobians-grad-clipping.md) |
| **Gradient clipping** | $\mathbf{g}\leftarrow\mathbf{g}\cdot\min\!\big(1,\tau/\lVert\mathbf{g}\rVert_2\big)$ | [3.1](unit-03-optimization-and-stability/01-backprop-jacobians-grad-clipping.md) |
| **Adam moments** | $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$; $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$ | [3.2](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) |
| **Bias correction** | $\hat m_t=m_t/(1-\beta_1^t)$, $\hat v_t=v_t/(1-\beta_2^t)$ | [3.2](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) |
| **AdamW update** | $\theta_{t+1}=\theta_t-\eta_t\left(\dfrac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}+\lambda\theta_t\right)$ | [3.2](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) |
| **Update-to-weight ratio** | $\lVert\Delta\theta\rVert/\lVert\theta\rVert$ per tensor; healthy values are commonly cited around $10^{-3}$ | [3.2](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) |
| **Warmup** | $\eta_t=\eta_{\max}\,t/t_w$ for $t\le t_w$ | [3.3](unit-03-optimization-and-stability/03-warmup-and-lr-schedules.md) |
| **Cosine decay** | $\eta_t=\eta_{\min}+\tfrac12(\eta_{\max}-\eta_{\min})\big(1+\cos(\pi\,\tfrac{t-t_w}{t_{\text{end}}-t_w})\big)$ | [3.3](unit-03-optimization-and-stability/03-warmup-and-lr-schedules.md) |
| **WSD schedule** | Warmup–Stable–Decay: constant LR plateau, then a short decay phase | [3.3](unit-03-optimization-and-stability/03-warmup-and-lr-schedules.md) |
| **BF16** | 1 sign, 8 exponent, 7 mantissa bits; FP32's dynamic range, ~3 decimal digits of precision | [3.4](unit-03-optimization-and-stability/04-mixed-precision-bf16-fp32.md) |
| **FP32** | 1 sign, 8 exponent, 23 mantissa bits | [3.4](unit-03-optimization-and-stability/04-mixed-precision-bf16-fp32.md) |
| **Machine epsilon** | $2^{-7}\approx 7.8\times10^{-3}$ (BF16) vs. $2^{-23}\approx 1.2\times10^{-7}$ (FP32) | [3.4](unit-03-optimization-and-stability/04-mixed-precision-bf16-fp32.md) |
| **Update swamping** | $w+\Delta$ rounds back to $w$ when $\lvert\Delta\rvert<\tfrac12\epsilon_{\text{mach}}\lvert w\rvert$ | [3.4](unit-03-optimization-and-stability/04-mixed-precision-bf16-fp32.md) |

## D. Data and Scaling (Unit 4)

| Term | Definition / Equation | Chapter |
|---|---|---|
| **MinHash / LSH** | Estimates Jaccard similarity of shingle sets; bands of signatures hash near-duplicates to the same bucket | [4.1](unit-04-data-and-scaling-laws/01-deduplication-and-filtering.md) |
| **Suffix-array dedup** | Finds repeated substrings of at least a minimum length across the corpus | [4.1](unit-04-data-and-scaling-laws/01-deduplication-and-filtering.md) |
| **Sequence packing** | Concatenate documents (with EOS) into windows of length $T$, tracking `segmentation_ids` and per-segment positions | [4.2](unit-04-data-and-scaling-laws/02-sequence-packing-and-masking.md) |
| **Gradient noise scale** | $B_{\text{noise}}=\text{tr}(\Sigma)/\lVert G\rVert^2$ | [4.3](unit-04-data-and-scaling-laws/03-critical-batch-size-noise-scale.md) |
| **Critical batch size** | $B_{\text{crit}}\approx B_{\text{noise}}$; beyond it, extra batch yields diminishing step reduction | [4.3](unit-04-data-and-scaling-laws/03-critical-batch-size-noise-scale.md) |
| **Compute accounting** | $C\approx 6ND$ ($2ND$ forward, $4ND$ backward) | [4.4](unit-04-data-and-scaling-laws/04-chinchilla-scaling-laws.md) |
| **Chinchilla parametric loss** | $L(N,D)=E+\dfrac{A}{N^{\alpha}}+\dfrac{B}{D^{\beta}}$ | [4.4](unit-04-data-and-scaling-laws/04-chinchilla-scaling-laws.md) |
| **IsoFLOP curve** | Final loss vs. $N$ at fixed $C$; its minimum gives $N_{\text{opt}}(C)$ | [4.4](unit-04-data-and-scaling-laws/04-chinchilla-scaling-laws.md) |

## E. TPU Systems (Unit 5)

| Term | Definition | Chapter |
|---|---|---|
| **HBM** | High-bandwidth memory on each TPU chip; holds weights, optimizer state, activations | [5.1](unit-05-tpu-systems-and-distributed/01-sharding-mesh-fsdp-tensor-expert.md) |
| **MXU** | Matrix multiply unit; the systolic array that delivers peak FLOP/s | [5.3](unit-05-tpu-systems-and-distributed/03-remat-checkpointing-and-mfu.md) |
| **ICI / DCN** | Inter-chip interconnect within a slice / data-center network between slices | [5.1](unit-05-tpu-systems-and-distributed/01-sharding-mesh-fsdp-tensor-expert.md) |
| **Mesh** | Named logical grid of devices (axes e.g. `data`, `fsdp`, `tensor`, `expert`) | [5.1](unit-05-tpu-systems-and-distributed/01-sharding-mesh-fsdp-tensor-expert.md) |
| **FSDP** | Fully-sharded data parallel (ZeRO-3): shard params/grads/optimizer state, gather just-in-time | [5.1](unit-05-tpu-systems-and-distributed/01-sharding-mesh-fsdp-tensor-expert.md) |
| **All-Gather** | Each device receives the concatenation of all shards | [5.2](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) |
| **Reduce-Scatter** | Sum across devices, then each keeps one shard of the result | [5.2](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) |
| **All-Reduce** | Sum across devices; every device gets the full result (= Reduce-Scatter + All-Gather) | [5.2](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) |
| **All-to-All** | Each device sends a distinct slice to every other device; used for MoE dispatch/combine | [5.2](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) |
| **Exposed communication** | Collective time not overlapped with compute; shows as gaps in the XProf trace | [5.2](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) |
| **Rematerialization** | Discard activations in the forward pass and recompute them in backward | [5.3](unit-05-tpu-systems-and-distributed/03-remat-checkpointing-and-mfu.md) |
| **MFU** | $\text{MFU}=\dfrac{\text{model FLOP/s achieved}}{\text{peak FLOP/s}}$, counting only useful model FLOPs (not recompute) | [5.3](unit-05-tpu-systems-and-distributed/03-remat-checkpointing-and-mfu.md) |

## F. Diagnostic Metrics (cross-cutting)

| Metric | What it tells you | See |
|---|---|---|
| `loss` (nats/token) | Raw objective; compare only under the same tokenizer | [1.3](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) |
| Validation BPB | Tokenizer-independent progress | [1.4](unit-01-objective-and-loss/04-bpb-and-perplexity.md) |
| `grad_norm` (pre/post clip) | Instability onset; how often clipping triggers | [3.1](unit-03-optimization-and-stability/01-backprop-jacobians-grad-clipping.md) |
| `param_norm` | Weight-decay equilibrium, drift | [3.2](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) |
| Max attention logit | QK growth preceding attention-entropy collapse | [2.2](unit-02-transformer-architecture/02-causal-attention-and-masking.md) |
| Router max logit, Z-loss | Router instability under BF16 | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| Expert load / dropped-token % | MoE collapse and capacity tuning | [2.4](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) |
| MFU, tokens/sec/chip | Systems efficiency | [5.3](unit-05-tpu-systems-and-distributed/03-remat-checkpointing-and-mfu.md) |
