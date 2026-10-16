# Unit 1: The Objective and the Loss

> **Book position:** Unit 1 of 5 · Previous: [Root Index](../README.md) · Next: [Unit 2 – The Transformer Engine](../unit-02-transformer-architecture/README.md)

## Unit Title & Ultimate Goal

**Ultimate Goal:** Understand what the pretraining loss curve actually measures in information theory, why changing a tokenizer invalidates raw cross-entropy loss comparisons, and how raw logits become a numerically stable scalar loss in BF16/FP32.

**Why this matters for 2B Dense and 5B MoE runs.** Every later decision in this book (architecture, optimizer, data mixture, parallelism) is judged by one number: validation loss. If you do not know exactly what that number is, a "win" in an ablation can be an artifact. Concretely:

- **Tokenizer confound.** Loss is in *nats per token*. A tokenizer with a larger vocabulary packs more bytes into each token, so each token is harder to predict and per-token loss goes *up* even when the model compresses text *better*. Comparing two runs with different tokenizers by raw loss is meaningless; bits-per-byte (BPB) is the fix.
- **Numerical failure.** The loss is computed from logits $\mathbf{z}_t \in \mathbb{R}^V$ with $V$ often $\ge 32\text{K}$. A naive $\log(\text{softmax}(\mathbf{z}))$ overflows or underflows in BF16/FP32. The log-sum-exp formulation, and the auxiliary Z-loss that keeps $\log Z$ bounded, are what keep a 5B MoE from emitting `NaN` at step 40K.
- **Wrong target alignment.** An off-by-one in the input/target shift lets the model copy its input and report a near-zero loss that means nothing.

## How This Unit Connects to Previous & Next Units

**Upstream (inputs).** This is the first unit; its only input is a raw text corpus. Everything here is defined *before* a single parameter exists: the tokenizer fixes the vocabulary $V$ and the realized sequence length $T$ for a given amount of text, and the autoregressive factorization fixes what "predicting text" means.

**Downstream (outputs).** Unit 1 produces three artifacts that the rest of the book consumes:

1. **$V$ and $T$** feed [Unit 2](../unit-02-transformer-architecture/README.md): $V$ sets the embedding/unembedding matrix size, $T$ sets the $O(T^2)$ attention cost and the causal mask.
2. **The loss gradient $\mathbf{p}_t - \mathbf{y}_t$** is the seed of backpropagation in [Unit 3](../unit-03-optimization-and-stability/README.md); the Z-loss term and BF16 behavior of the logits determine whether that seed is stable.
3. **BPB** is the common currency used by [Unit 4](../unit-04-data-and-scaling-laws/README.md) (scaling-law fits, data-quality ablations) and by the experiment template in [`templates/EXPERIMENT_LOG_TEMPLATE.md`](../templates/EXPERIMENT_LOG_TEMPLATE.md).

## The Unit Dependency & Implementation Table

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **1.1 Tokenization & Fertility** | Splitting raw UTF-8 strings into integer IDs from a vocabulary $\mathcal{V}$ of size $V$. | **Depends on:** raw corpus statistics.<br>**Affects:** sequence length $T$, embedding memory, and per-token loss. | Fertility (tokens per word) & bytes-per-token. | [Sennrich et al. (2016) – BPE](https://arxiv.org/abs/1508.07909) | `assets/` tokenizer model & `input_pipeline/` preprocessing. |
| **1.2 Autoregressive Factorization** | Breaking the probability of a full text sequence into step-by-step next-token predictions: $P(\mathbf{x}) = \prod_{t=1}^T P(x_t \mid x_{<t})$. | **Depends on:** chain rule of probability.<br>**Affects:** requires causal masking so token $t$ cannot peek at $t+1$. | Joint sequence log-probability $\sum_t \log P(x_t \mid x_{<t})$. | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024) | Shifting inputs vs. targets by 1 position in `input_pipeline/`. |
| **1.3 Softmax & LogSumExp (Cross-Entropy)** | Converting raw model output scores (logits $\mathbf{z}_t$) into probabilities and penalizing the model for assigning low probability to the true next token $x_t$. | **Depends on:** unembedding matrix output.<br>**Affects:** produces the initial gradient $\mathbf{p}_t - \mathbf{y}_t$ that starts backpropagation. | Training loss (nats/token) & Z-loss (logit drift penalty). | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024); [ST-MoE (2022)](https://arxiv.org/abs/2202.08906) (Z-loss) | `max_utils.py`: `cross_entropy_with_logits` & `config.z_loss`. |
| **1.4 Bits-Per-Byte (BPB) & Perplexity** | Converting raw nats/token loss into a tokenizer-independent compression score (BPB) or branching factor (perplexity). | **Depends on:** cross-entropy loss & UTF-8 byte count.<br>**Affects:** how you judge whether Ablation A actually beat Ablation B. | $\text{BPB} = \dfrac{\text{Total Nats}}{\ln(2) \cdot \text{Bytes}}$; $\text{PPL} = e^{\text{Loss}}$. | [Shannon (1948)](https://ieeexplore.ieee.org/document/6773024) | Evaluation loop in `train.py` / downstream eval scripts. |

> **Path caveat.** MaxText file/flag names drift between releases. Treat the "Spot in MaxText" column as a pointer, and pin the exact commit of [MaxText](https://github.com/AI-Hypercomputer/maxtext) you are running in each experiment log.

## Primary Paper Lineage Breakdown

### Shannon (1948) – [A Mathematical Theory of Communication](https://ieeexplore.ieee.org/document/6773024)
- **Problem before:** No quantitative notion of "information" or of how well a source can be compressed; "predictability of language" had no unit.
- **Intervention:** Defined entropy $H(X) = -\sum_x p(x)\log p(x)$ and the cross-entropy / coding-cost view: the expected code length under model $q$ for data from $p$ is $H(p,q) = H(p) + D_{\text{KL}}(p\Vert q)$. Minimizing cross-entropy *is* maximizing likelihood *is* minimizing compression cost.
- **Where in MaxText/JAX:** The training loss in `train.py` (via the cross-entropy helper) is the empirical $H(p,q)$ in nats; the BPB conversion is an eval-side post-processing step you may need to add yourself.

### Sennrich, Haddow & Birch (2016) – [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- **Problem before:** Word-level vocabularies were closed and huge; rare/unseen words became `<UNK>`. Character-level models made sequences far too long.
- **Intervention:** Byte-Pair Encoding: iteratively merge the most frequent adjacent symbol pair, yielding an open-vocabulary subword segmentation with a tunable size $V$.
- **Where in MaxText/JAX:** Tokenizer assets are loaded in the `input_pipeline/` (SentencePiece/tiktoken/HF tokenizers depending on config); tokenization is done on host CPU, not on TPU. *Tradeoff created:* the vocabulary choice silently changes $T$, embedding memory $V\cdot d$, and the meaning of "loss per token".

### Zoph et al. (2022) – [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) (Z-loss)
- **Problem before:** Large sparse models trained in low precision showed sudden loss spikes and divergence; large-magnitude logits amplify BF16 rounding error in the softmax.
- **Intervention:** The **router Z-loss** $L_z = \frac{1}{B}\sum_i \left(\log\sum_j e^{z_{ij}}\right)^2$ penalizes large log-partition values. The same idea (penalizing $\log Z$ of the *output* softmax) is used in many dense LMs, e.g. PaLM's auxiliary loss.
- **Where in MaxText/JAX:** `config.z_loss` style options on the output logits, and the router Z-loss in `layers/moe.py` (see Unit 2). Chapter [1.3](03-softmax-logsumexp-crossentropy.md) derives the gradient of the Z-loss.

## Chapter Navigation Links

| Chapter | File |
|---|---|
| 1.1 Tokenization & Fertility | [01-tokenization-and-fertility.md](01-tokenization-and-fertility.md) |
| 1.2 Autoregressive Factorization | [02-autoregressive-factorization.md](02-autoregressive-factorization.md) |
| 1.3 Softmax, LogSumExp & Cross-Entropy | [03-softmax-logsumexp-crossentropy.md](03-softmax-logsumexp-crossentropy.md) |
| 1.4 Bits-Per-Byte & Perplexity | [04-bpb-and-perplexity.md](04-bpb-and-perplexity.md) |

**Next:** [Unit 2 – The Transformer Engine →](../unit-02-transformer-architecture/README.md)
