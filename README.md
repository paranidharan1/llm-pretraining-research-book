# LLM Pretraining Research Book

**A first-principles research manual for 2B Dense and 5B MoE (1.4B active) pretraining on TPUs with MaxText / JAX / XLA.**

## Ultimate Goal

This repository is a textbook, not a study plan. Its purpose is to move you from **operating** distributed training runs (changing YAML flags in MaxText and waiting for loss curves) to **conducting original pretraining science**:

- deriving the math behind every component,
- predicting ablation outcomes *a priori*,
- diagnosing BF16 and MoE instabilities from gradients and spectral norms,
- and modifying JAX/XLA kernels when a flag is not enough.

Every unit feeds the next, every claim is tied to a primary paper (see [READING_LIBRARY.md](READING_LIBRARY.md)), and every equation is mapped to a concrete place in the MaxText/JAX/XLA stack.

## The Five Core Pillars

1. **Exact mathematical derivations.** No skipped algebraic steps. Each chapter has a *Line-by-Line Mathematical Derivation* section.
2. **Historical and research-paper lineage.** For each idea: what problem existed before, what intervention solved it, and what tradeoff it created.
3. **Flag-to-mechanism mapping in MaxText / JAX / XLA.** A config flag is never a black box; each is traced to the line of math it controls.
4. **Unconfounded experimental design and falsifiable hypothesis testing.** One intervention per experiment, controlled invariants, and a quantitative promotion rule ([template](templates/EXPERIMENT_LOG_TEMPLATE.md)).
5. **Concrete focus on 2B Dense and 5B MoE (1.4B active) architectures on TPU clusters.** Examples, budgets, and exercises use these two models.

## How to Read This Book

Read the units in order; each unit's README states what it consumes from the previous unit and what it hands to the next. Inside a unit, read the README first, then the chapters. Fill in each chapter's sections 3, 8 and 9 yourself (derivation, prediction, notes) during study sessions.

## Master Concept Dependency Graph

```text
UNIT 1: THE OBJECTIVE (What the model is trying to do)
  [1.1 Raw Text & Tokenization] ──(determines)──> [1.2 Sequence Length T & Vocab V]
                                                        │
                                                 (feeds into)
                                                        ▼
  [1.4 Evaluation Metrics (BPB / PPL)] <──(measures)── [1.3 Cross-Entropy / NLL Loss]
                                                        │
UNIT 2: THE ENGINE (How the model processes tokens)     │ (computed from logits of)
                                                        ▼
  [2.1 Embeddings & RoPE Positions] ──(feeds)──> [2.2 Causal Attention] ──(routes to)──> [2.3 Dense MLP vs MoE Router]
                                                                                                │
                                                                                         (stabilized by)
                                                                                                ▼
UNIT 3: THE LEARNING DYNAMICS (How weights update safely)                              [2.5 Residual Stream & RMSNorm]
                                                                                                │
  [3.4 BF16 / FP32 Precision] <──(constrains)── [3.1 Backprop & Gradients] <──(flows through)───┘
                                                        │
                                                 (updates weights via)
                                                        ▼
  [3.3 LR Schedules & Clipping] ──(controls)──> [3.2 AdamW Optimizer States]
                                                        │
UNIT 4: DATA & SCALING LAWS (What to feed & how long)   │ (scaled across batches via)
                                                        ▼
  [4.1 Filtering & Deduplication] ──(shapes)──> [4.2 Packing & Mixtures] ──(governed by)──> [4.3 Scaling Laws & Batch Size]
                                                                                                │
UNIT 5: TPU SYSTEMS (How it runs fast on hardware)                                       (mapped onto)
                                                                                                ▼
  [5.1 Sharding (FSDP / Tensor / Expert)] ──(communicates via)──> [5.2 XLA Collectives] ──(measured by)──> [5.3 MFU & Step Time]
```

> **Note on numbering.** The graph's node labels are conceptual groupings. The chapter numbers match the files: 2.3 and 2.4 are separate chapters (Dense SwiGLU MLP and MoE routing); 2.5 is the residual stream and RMSNorm; and in Unit 4, 4.3 (critical batch size) and 4.4 (Chinchilla scaling laws) together form the graph's "Scaling Laws & Batch Size" node.

## Master Table of Contents

### Front matter and back matter

| Document | Purpose |
|---|---|
| [READING_LIBRARY.md](READING_LIBRARY.md) | Annotated bibliography of the 20 primary pretraining papers (plus the MaxText repo and supplementary citations), with direct links |
| [GLOSSARY.md](GLOSSARY.md) | Master glossary of equations, metrics, and system concepts |
| [templates/EXPERIMENT_LOG_TEMPLATE.md](templates/EXPERIMENT_LOG_TEMPLATE.md) | Rigorous ablation and hypothesis template |
| [templates/DEBUGGING_DECISION_TREE.md](templates/DEBUGGING_DECISION_TREE.md) | Diagnostic flowchart for spikes, NaNs, MoE collapse and low MFU |

### Units and chapters

| Unit | Chapters |
|---|---|
| **[Unit 1 – The Objective and the Loss](unit-01-objective-and-loss/README.md)** | [1.1 Tokenization & Fertility](unit-01-objective-and-loss/01-tokenization-and-fertility.md) · [1.2 Autoregressive Factorization](unit-01-objective-and-loss/02-autoregressive-factorization.md) · [1.3 Softmax, LogSumExp & Cross-Entropy](unit-01-objective-and-loss/03-softmax-logsumexp-crossentropy.md) · [1.4 BPB & Perplexity](unit-01-objective-and-loss/04-bpb-and-perplexity.md) |
| **[Unit 2 – The Transformer Engine](unit-02-transformer-architecture/README.md)** | [2.1 RoPE](unit-02-transformer-architecture/01-rope-positional-embeddings.md) · [2.2 Causal Attention & Masking](unit-02-transformer-architecture/02-causal-attention-and-masking.md) · [2.3 SwiGLU Dense MLP](unit-02-transformer-architecture/03-swiglu-dense-mlp.md) · [2.4 MoE Routing, Capacity & Aux Loss](unit-02-transformer-architecture/04-moe-routing-capacity-auxloss.md) · [2.5 Residual Stream & RMSNorm](unit-02-transformer-architecture/05-residual-stream-and-rmsnorm.md) |
| **[Unit 3 – Learning Dynamics and Stability](unit-03-optimization-and-stability/README.md)** | [3.1 Backprop, Jacobians & Grad Clipping](unit-03-optimization-and-stability/01-backprop-jacobians-grad-clipping.md) · [3.2 AdamW Moments & Weight Decay](unit-03-optimization-and-stability/02-adamw-moments-weight-decay.md) · [3.3 Warmup & LR Schedules](unit-03-optimization-and-stability/03-warmup-and-lr-schedules.md) · [3.4 Mixed Precision BF16/FP32](unit-03-optimization-and-stability/04-mixed-precision-bf16-fp32.md) |
| **[Unit 4 – Data and Scaling Laws](unit-04-data-and-scaling-laws/README.md)** | [4.1 Deduplication & Filtering](unit-04-data-and-scaling-laws/01-deduplication-and-filtering.md) · [4.2 Sequence Packing & Masking](unit-04-data-and-scaling-laws/02-sequence-packing-and-masking.md) · [4.3 Critical Batch Size & Noise Scale](unit-04-data-and-scaling-laws/03-critical-batch-size-noise-scale.md) · [4.4 Chinchilla Scaling Laws](unit-04-data-and-scaling-laws/04-chinchilla-scaling-laws.md) |
| **[Unit 5 – TPU Systems and Distributed Training](unit-05-tpu-systems-and-distributed/README.md)** | [5.1 Sharding: Mesh, FSDP, Tensor, Expert](unit-05-tpu-systems-and-distributed/01-sharding-mesh-fsdp-tensor-expert.md) · [5.2 XLA Collectives](unit-05-tpu-systems-and-distributed/02-xla-collectives-communication.md) · [5.3 Remat, Checkpointing & MFU](unit-05-tpu-systems-and-distributed/03-remat-checkpointing-and-mfu.md) |

## Unit Dependency Tables ("Page 1" Reference)

Each table is repeated verbatim in its unit's README, alongside the paper-lineage breakdown.

### Unit 1: The Objective and the Loss

**Ultimate Goal:** Understand what the pretraining loss curve actually measures in information theory, why changing a tokenizer invalidates raw cross-entropy loss comparisons, and how raw logits become a numerically stable scalar loss in BF16/FP32.

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **1.1 Tokenization & Fertility** | Splitting raw UTF-8 strings into integer IDs from a vocabulary $\mathcal{V}$ of size $V$. | **Depends on:** raw corpus statistics.<br>**Affects:** sequence length $T$, embedding memory, and per-token loss. | Fertility (tokens per word) & bytes-per-token. | [Sennrich et al. (2016) – BPE](https://arxiv.org/abs/1508.07909) | `assets/` tokenizer model & `input_pipeline/` preprocessing. |
| **1.2 Autoregressive Factorization** | Breaking the probability of a full text sequence into step-by-step next-token predictions: $P(\mathbf{x}) = \prod_{t=1}^T P(x_t \mid x_{<t})$. | **Depends on:** chain rule of probability.<br>**Affects:** requires causal masking so token $t$ cannot peek at $t+1$. | Joint sequence log-probability $\sum_t \log P(x_t \mid x_{<t})$. | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024) | Shifting inputs vs. targets by 1 position in `input_pipeline/`. |
| **1.3 Softmax & LogSumExp (Cross-Entropy)** | Converting raw model output scores (logits $\mathbf{z}_t$) into probabilities and penalizing the model for assigning low probability to the true next token $x_t$. | **Depends on:** unembedding matrix output.<br>**Affects:** produces the initial gradient $\mathbf{p}_t - \mathbf{y}_t$ that starts backpropagation. | Training loss (nats/token) & Z-loss (logit drift penalty). | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024); [ST-MoE (2022)](https://arxiv.org/abs/2202.08906) (Z-loss) | `max_utils.py`: `cross_entropy_with_logits` & `config.z_loss`. |
| **1.4 Bits-Per-Byte (BPB) & Perplexity** | Converting raw nats/token loss into a tokenizer-independent compression score (BPB) or branching factor (perplexity). | **Depends on:** cross-entropy loss & UTF-8 byte count.<br>**Affects:** how you judge whether Ablation A actually beat Ablation B. | $\text{BPB} = \dfrac{\text{Total Nats}}{\ln(2) \cdot \text{Bytes}}$; $\text{PPL} = e^{\text{Loss}}$. | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024) | Evaluation loop in `train.py` / downstream eval scripts. |

### Unit 2: The Transformer Engine

**Ultimate Goal:** Master the forward-pass equations of modern Dense and Sparse (MoE) Transformers, understand how every layer transforms representations, why MoE routers collapse or drop tokens, and how normalization placement keeps deep networks trainable.

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **2.1 RoPE (Rotary Position Embeddings)** | Rotating query ($Q$) and key ($K$) vectors by angles proportional to their token position $t$ so attention knows the distance between words. | **Depends on:** head dimension $d_k$ & base frequency $\theta$.<br>**Affects:** long-context extrapolation (8K → 128K). | Long-context retrieval accuracy & per-position validation loss. | [Su et al. (2021) – RoFormer](https://arxiv.org/abs/2104.09864) | `MaxText/layers/embeddings.py` (`RotaryEmbedding`). |
| **2.2 Scaled Dot-Product Attention & Masking** | Computing relevance scores between tokens via $\text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + \text{Mask}\right)V$. | **Depends on:** RoPE, segment IDs, sequence length $T$.<br>**Affects:** compute scales quadratically ($O(T^2)$) with sequence length. | Attention entropy & max attention logit. | [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) | `MaxText/layers/attentions.py` & Splash/FlashAttention kernels. |
| **2.3 SwiGLU MLP (Dense Feed-Forward)** | Non-linear feature transformation using a gated projection: $\left(\text{Swish}(xW_{\text{gate}}) \odot xW_{\text{up}}\right)W_{\text{down}}$. | **Depends on:** attention output.<br>**Affects:** holds $\sim 2/3$ of parameters in a 2B dense model. | Activation RMS norm & dead-neuron ratio. | [Shazeer (2020) – GLU Variants](https://arxiv.org/abs/2002.05202) | `MaxText/layers/linears.py` (`MlpBlock`, `mlp_dim`). |
| **2.4 MoE Routing, Capacity & Aux Loss** | Replacing one big MLP with $E$ smaller expert MLPs, using a router to send each token to the top-$k$ experts (e.g., 1.4B active out of 5B total). | **Depends on:** router logits, `capacity_factor`.<br>**Affects:** token dropping, expert imbalance, and BF16 router spikes. | Load-balance aux loss, router Z-loss, dropped-token %. | [Fedus et al. (2021) – Switch](https://arxiv.org/abs/2101.03961); [Zoph et al. (2022) – ST-MoE](https://arxiv.org/abs/2202.08906) | `MaxText/layers/moe.py` (`capacity_factor`, `load_balance_loss_weight`). |
| **2.5 Residual Stream & Pre-RMSNorm** | Keeping a clean additive highway $x_{l+1} = x_l + \text{Layer}(\text{RMSNorm}(x_l))$ so gradients flow straight from output to Layer 1. | **Depends on:** initialization scale & depth $L$.<br>**Affects:** prevents vanishing/exploding gradients at step 0. | Layerwise residual-stream variance / RMS norm. | [He et al. (2015)](https://arxiv.org/abs/1512.03385); [Zhang & Sennrich (2019)](https://arxiv.org/abs/1910.07467); [Xiong et al. (2020)](https://arxiv.org/abs/2002.04745) | `MaxText/layers/normalizations.py` (`RMSNorm`) & `models.py`. |

### Unit 3: Learning Dynamics and Stability

**Ultimate Goal:** Diagnose why loss spikes happen, why BF16 overflows or swamps small updates, and how backpropagation Jacobians, gradient clipping, and AdamW states (`b1`, `b2`, `weight_decay`) govern weight trajectories.

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **3.1 Backprop, Jacobians & Grad Clipping** | Applying the chain rule backward through the layers to compute $\mathbf{g} = \nabla_{\boldsymbol{\theta}} \mathcal{L}$, and rescaling $\mathbf{g}$ if $\lVert\mathbf{g}\rVert_2$ exceeds a threshold. | **Depends on:** forward activations & loss gradient.<br>**Affects:** protects weights from single bad data batches or router spikes. | Global L2 gradient norm (`grad_norm`) before & after clipping. | [Pascanu et al. (2013)](https://arxiv.org/abs/1211.5063) (clipping) | `train.py`: `jax.value_and_grad` & Optax `clip_by_global_norm`. |
| **3.2 AdamW (Adaptive Moments & Weight Decay)** | Maintaining a running average of gradient direction ($m_t$, controlled by $\beta_1$) and gradient squared magnitude ($v_t$, controlled by $\beta_2$), plus direct weight shrinkage ($\lambda$). | **Depends on:** gradients $\mathbf{g}_t$ and FP32 master weights.<br>**Affects:** step size per parameter and steady-state weight norm. | Parameter L2 norm (`param_norm`) & update-to-weight ratio. | [Kingma & Ba (2014)](https://arxiv.org/abs/1412.6980); [Loshchilov & Hutter (2019)](https://arxiv.org/abs/1711.05101) | `MaxText/optimizers.py` (`adam_b1`, `adam_b2`, `adam_weight_decay`). |
| **3.3 Warmup & LR Schedules (Cosine / WSD)** | Ramping learning rate $\eta_t$ linearly from $0$ (while Adam's $v_t$ calibrates) and decaying it smoothly to settle into a minimum. | **Depends on:** total token budget & batch size.<br>**Affects:** early divergence (if warmup is too short) & final validation loss. | Current learning rate $\eta_t$ vs. loss slope. | [Goyal et al. (2017)](https://arxiv.org/abs/1706.02677); [Hoffmann et al. (2022)](https://arxiv.org/abs/2203.15556) | `max_utils.py`: `create_learning_rate_schedule`. |
| **3.4 Mixed Precision (BF16 vs FP32)** | Running matrix multiplies in 16-bit (bfloat16: 8-bit exponent, 7-bit mantissa) for TPU speed while keeping master weights & optimizer states in 32-bit (float32). | **Depends on:** TPU MXU hardware.<br>**Affects:** memory footprint, throughput, and rounding/underflow errors. | NaN/Inf occurrences, logit magnitude, weight-update swamping. | [Micikevicius et al. (2018)](https://arxiv.org/abs/1710.03740) | `config.dtype="bfloat16"`, `config.weight_dtype="float32"`. |

### Unit 4: Data and Scaling Laws

**Ultimate Goal:** Treat data curation, deduplication, sequence packing, and mixture weighting as rigorous mathematical optimization problems, and connect them to Chinchilla scaling laws and critical batch size theory.

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **4.1 Deduplication & Quality Filtering** | Removing exact/near-duplicate documents (MinHash LSH / suffix arrays) and scoring text quality via heuristics or classifiers. | **Depends on:** raw web dumps.<br>**Affects:** prevents memorization spikes, lowers validation BPB, increases data efficiency. | Retention rate %, duplicate cluster size, held-out eval loss. | [Lee et al. (2022)](https://arxiv.org/abs/2107.06499); [FineWeb (2024)](https://arxiv.org/abs/2406.17557) | Offline data-curation pipeline before Grain/TFRecord creation. |
| **4.2 Sequence Packing & Document Masking** | Concatenating multiple documents into a fixed window $T$ with `segmentation_ids` so there is zero wasted padding compute and no cross-document attention leakage. | **Depends on:** document length distribution.<br>**Affects:** effective tokens per batch and long-context boundary behavior. | Padding waste % (non-zero segmentation ratio). | [Raffel et al. (2020) – T5](https://arxiv.org/abs/1910.10683) | `input_pipeline/_grain_data_processing.py` (packing). |
| **4.3 Critical Batch Size & Gradient Noise Scale** | The threshold batch size $B_{\text{crit}}$ where doubling batch size stops cutting required training steps in half because gradient noise is already averaged out. | **Depends on:** current loss level & dataset diversity.<br>**Affects:** optimal global batch size (`per_device_batch_size` × chips) and LR scaling. | Gradient noise scale $B_{\text{noise}} = \text{tr}(\Sigma) / \lVert\mathbf{g}\rVert^2$. | [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) | Global batch token calculation in MaxText config. |
| **4.4 Compute-Optimal Scaling ($C \approx 6ND$)** | Mathematical law governing how to trade off active parameters ($N$) vs. training tokens ($D$) for a fixed TPU FLOP budget ($C$). | **Depends on:** architecture (Dense vs MoE) & data quality.<br>**Affects:** how big your pilot ablations should be before a full run. | IsoFLOP curves & power-law loss fit $L(N, D)$. | [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361); [Hoffmann et al. (2022) – Chinchilla](https://arxiv.org/abs/2203.15556) | Experiment planning & compute-budget calculation. |

### Unit 5: TPU Systems and Distributed Training

**Ultimate Goal:** Translate mathematical operations into physical TPU meshes, understand when XLA inserts All-Gather, Reduce-Scatter, or All-to-All collectives, and maximize Model FLOPs Utilization (MFU).

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **5.1 Sharding Mesh (FSDP, Tensor, Expert)** | Splitting weights, optimizer states, and activations across TPU axes (`data`, `fsdp`, `tensor`, `expert`) so a 5B+ model fits in HBM. | **Depends on:** TPU slice topology (e.g., v5e/v5p ICI links).<br>**Affects:** HBM memory per chip and communication overhead. | Peak HBM usage (GB) & ICI link utilization. | [Rajbhandari et al. (2020) – ZeRO](https://arxiv.org/abs/1910.02054); [GSPMD (2021)](https://arxiv.org/abs/2105.04663) | `MaxText/layers/quantizations.py` & `sharding.py` (`ici_fsdp_parallelism`, etc.). |
| **5.2 Collectives (All-Gather, Reduce-Scatter, All-to-All)** | The exact network operations XLA inserts to gather weights before a layer (All-Gather), sum/shard gradients (Reduce-Scatter), or route tokens to MoE experts (All-to-All). | **Depends on:** sharding annotations.<br>**Affects:** whether your TPU MXUs sit idle waiting for network transfers. | % step time spent in exposed communication (XProf). | [GShard (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668) | XLA compiler GSPMD pass & TensorBoard XProf trace. |
| **5.3 Rematerialization (Activation Checkpointing) & MFU** | Discarding intermediate forward activations and recomputing them during backprop to save HBM, trading extra FLOPs for larger batch sizes and higher Model FLOPs Utilization (MFU). | **Depends on:** HBM capacity & `remat_policy`.<br>**Affects:** step time, max sequence length $T$, and hardware efficiency. | MFU % (achieved TFLOP/s ÷ peak TPU TFLOP/s) & tokens/sec/chip. | [Chen et al. (2016)](https://arxiv.org/abs/1604.06174); [PaLM (2022)](https://arxiv.org/abs/2204.02311) (MFU) | `config.remat_policy` & `max_utils.py` TFLOP calculator. |

## Repository Conventions

- **Math:** GitHub-flavored Markdown with `$...$` inline and `$$...$$` display LaTeX.
- **Units:** loss in nats/token unless stated; BPB in bits/byte; $\ln$ is the natural log, $\log_2$ is explicit.
- **MaxText paths:** pointers into [MaxText](https://github.com/AI-Hypercomputer/maxtext), which changes quickly. Pin a commit hash in each experiment log and verify paths against it.
- **Citations:** every paper is listed with arXiv/DOI links in [READING_LIBRARY.md](READING_LIBRARY.md); within unit READMEs each is linked at first mention.
