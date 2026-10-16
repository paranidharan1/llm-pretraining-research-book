# Reading Library: Annotated Bibliography

Twenty primary papers in the order they enter the book, followed by the MaxText repository and supplementary citations used in unit tables. Links go to the arXiv abstract page (or the publisher/IEEE page for Shannon). [← Back to Root Index](README.md)

**Unit key:** U1 Objective & Loss · U2 Transformer Engine · U3 Optimization & Stability · U4 Data & Scaling Laws · U5 TPU Systems.

## A. The 20 Foundational Papers

| # | Paper Title & Link | Authors & Year | Core Contribution & Why It Matters | Assumptions & Limitations | Connection to Modern Pretraining | Unit Mapping |
|---|---|---|---|---|---|---|
| 1 | [A Mathematical Theory of Communication](https://ieeexplore.ieee.org/document/6773024) | Claude E. Shannon, 1948 | Defines entropy and cross-entropy; shows optimal code length is $-\log_2 p$. Establishes that the LM loss is a compression cost. | Assumes a stationary, well-defined source distribution; says nothing about learnability. | The pretraining loss is empirical cross-entropy; BPB is Shannon's bits-per-symbol with bytes as symbols. | U1 (1.2, 1.3, 1.4) |
| 2 | [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) | Sennrich, Haddow, Birch, 2016 | Adapts byte-pair encoding to build open-vocabulary subword vocabularies. | Greedy, frequency-driven merges; segmentation not linguistically motivated; vocabulary fixed after training. | Basis of nearly all LLM tokenizers (byte-level BPE variants). Vocabulary size trades $T$ vs. $V\cdot d$ embedding cost. | U1 (1.1) |
| 3 | [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | He, Zhang, Ren, Sun, 2015 | Identity skip connections make very deep networks trainable; the Jacobian always contains $I$. | Original setting is CNN image classification with BatchNorm; says little about transformer norm placement. | The residual stream is the central data structure of every transformer layer stack. | U2 (2.5) |
| 4 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | Vaswani et al., 2017 | Introduces the Transformer: multi-head scaled dot-product attention with positional encodings. | $O(T^2)$ compute and memory; original used Post-LN, sinusoidal absolute positions, ReLU FFN. | Core architecture; modern LLMs replace position encoding (RoPE), norm (Pre-RMSNorm) and FFN (SwiGLU) while keeping attention. | U2 (2.2) |
| 5 | [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) | Zhang & Sennrich, 2019 | Drops mean-centering from LayerNorm; keeps re-scaling invariance with less compute. | Evidence of equal quality is empirical and task-dependent. | Default norm in Llama-style and MaxText decoder models. | U2 (2.5) |
| 6 | [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) | Xiong et al., 2020 | Mean-field analysis: Post-LN has large gradients near the output at init; Pre-LN is well-behaved and removes the need for warmup in their setting. | Analysis at initialization; Pre-LN may have weaker final quality or growing residual norm with depth. | Justifies Pre-Norm as the default for stable large-scale training. | U2 (2.5), U3 |
| 7 | [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) | Noam Shazeer, 2020 | Gated FFN variants (notably SwiGLU/GeGLU) improve perplexity at matched parameters. | Empirical, no theoretical explanation (the paper itself attributes success "to divine benevolence"); small-scale experiments. | SwiGLU is the default dense FFN; each MoE expert is a SwiGLU MLP. | U2 (2.3, 2.4) |
| 8 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | Su et al., 2021 | Encodes position by rotating $Q,K$ so attention scores depend on relative offset. | Extrapolation beyond trained length still degrades without scaling tricks (position interpolation, base changes). | Standard in Llama/Gemma/Qwen-class models; long-context extension recipes modify RoPE's base/frequencies. | U2 (2.1) |
| 9 | [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | Kingma & Ba, 2014 | Adaptive per-parameter step sizes from EMAs of gradient and its square, with bias correction. | Original convergence proof has a known flaw (later work, e.g. AMSGrad); sensitive to $\epsilon$, $\beta_2$ at scale. | Optimizer for essentially all LLM pretraining; $\beta_2\approx0.95$ is common at large batch to tame spikes. | U3 (3.2) |
| 10 | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | Loshchilov & Hutter, 2019 | Separates weight decay from the gradient-based update so it is not rescaled by $1/\sqrt{v}$. | Optimal decay interacts with LR and schedule; the paper's normalized-decay heuristic is rarely used at LLM scale. | AdamW is the de facto LLM optimizer. | U3 (3.2) |
| 11 | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Micikevicius et al., 2018 | FP32 master weights with FP16 compute and loss scaling. | Focused on FP16 (needs loss scaling); BF16's wider range changes the failure modes. | TPU recipe: BF16 matmuls, FP32 master weights, FP32 optimizer state, selective FP32 ops (norms, softmax, router). | U3 (3.4) |
| 12 | [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162) | McCandlish et al., 2018 | Gradient noise scale predicts critical batch size and the data/step tradeoff. | Noise-scale estimates are noisy; $B_{\text{crit}}$ drifts upward as loss falls. | Guides global batch size and LR co-scaling; larger $B$ is only useful up to $B_{\text{crit}}$. | U4 (4.3) |
| 13 | [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | Kaplan et al., 2020 | Power laws of loss in $N$, $D$, $C$; introduces the $C\approx 6ND$ accounting. | Fixed cosine horizon biased the fits; excluded embedding params from $N$; favored model size over data. | Basis of compute planning; its flaws motivated Chinchilla. | U4 (4.4) |
| 14 | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Hoffmann et al., 2022 | Equal scaling of $N$ and $D$ (about 20 tokens/param); parametric fit $L(N,D)$; schedule must match token horizon. | Dense models, single-epoch data; later work (e.g., over-training for inference cost) departs from "compute-optimal". | Sets the token budget for the 2B dense run; MoE active-vs-total accounting is an open question. | U3 (3.3), U4 (4.4) |
| 15 | [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499) | Lee et al., 2022 | Suffix-array exact and MinHash near-duplicate removal; reduces memorization and train/test overlap. | Thresholds are corpus-specific; dedup can remove legitimately repeated useful text. | Standard stage in every serious data pipeline. | U4 (4.1) |
| 16 | [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) | Penedo et al., 2023 | Shows heavily filtered and deduplicated web data alone can match curated corpora. | Filter recipe is heuristic and English-centric; only part of the dataset was released. | Evidence that web data quality, not source diversity, is the main lever. | U4 (4.1) |
| 17 | [The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale](https://arxiv.org/abs/2406.17557) | Penedo et al., 2024 | Open, ablation-backed web filtering/dedup recipes; FineWeb-Edu classifier-filtered subset. | Per-snapshot dedup chosen empirically; small-model ablations may not transfer to every scale. | Practical open data source and a template for data ablations. | U4 (4.1) |
| 18 | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | Fedus, Zoph, Shazeer, 2021 | Top-1 routing, capacity factor, and the load-balancing auxiliary loss. | Token dropping at capacity; training instability in low precision; top-1 routing is a choice, not a law. | Foundation of modern MoE routing; defines aux-loss and capacity vocabulary. | U2 (2.4) |
| 19 | [ST-MoE: Designing Stable and Transferable Sparse Expert Models](https://arxiv.org/abs/2202.08906) | Zoph et al., 2022 | Router Z-loss, FP32 routing, and stability/transfer analysis for sparse models. | Studied mostly in encoder-decoder, fine-tuning transfer; decoder-only pretraining dynamics differ. | Source of the Z-loss now used in both MoE routers and output logits. | U1 (1.3), U2 (2.4), U3 |
| 20 | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) | Rae et al., 2021 | Large-scale training study with MassiveText quality filtering, dedup, and ablations of optimizer/precision choices. | Pre-Chinchilla compute split (under-trained for its size); results at 280B scale. | Reference for data-quality filtering and the practical record of loss spikes, clipping and precision at scale. | U3, U4 (4.1) |

## B. Software Reference

| # | Resource & Link | Maintainer | Why It Matters | Caveats | Unit Mapping |
|---|---|---|---|---|---|
| S1 | [MaxText (GitHub)](https://github.com/AI-Hypercomputer/maxtext) | Google AI Hypercomputer | Open-source JAX LLM training reference with config-driven sharding, MoE and MFU reporting; the codebase every "Spot in MaxText" pointer targets. | File and flag names change between releases; always pin a commit. | All units |

## C. Supplementary Citations Used in Unit Tables

These are referenced in the unit dependency tables but are not among the 20 foundational papers.

| Paper & Link | Authors & Year | Role in This Book | Unit |
|---|---|---|---|
| [On the difficulty of training recurrent neural networks](https://arxiv.org/abs/1211.5063) | Pascanu, Mikolov, Bengio, 2013 | Gradient norm clipping | U3 (3.1) |
| [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) | Goyal et al., 2017 | Linear warmup and LR scaling | U3 (3.3) |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)](https://arxiv.org/abs/1910.10683) | Raffel et al., 2020 | Example packing, C4 filtering | U4 (4.2) |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | Rajbhandari et al., 2020 | State/gradient/parameter sharding (FSDP) | U5 (5.1) |
| [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663) | Xu et al., 2021 | XLA sharding propagation | U5 (5.1) |
| [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668) | Lepikhin et al., 2020 | Expert parallelism, All-to-All dispatch | U5 (5.2) |
| [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174) | Chen et al., 2016 | Activation checkpointing | U5 (5.3) |
| [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311) | Chowdhery et al., 2022 | Definition of MFU; z-loss on output logits | U1, U5 (5.3) |

## D. Suggested Reading Order

1. **Unit 1:** #1 → #2 → #19 (Z-loss section only).
2. **Unit 2:** #3 → #4 → #5 → #6 → #7 → #8 → #18 → #19.
3. **Unit 3:** Pascanu → #9 → #10 → Goyal → #11.
4. **Unit 4:** #15 → #16 → #17 → T5 (packing) → #12 → #13 → #14 → #20.
5. **Unit 5:** ZeRO → GSPMD → GShard → Chen → PaLM (MFU section) → [MaxText](https://github.com/AI-Hypercomputer/maxtext) docs.
