# Unit 4: Data and Scaling Laws

> **Book position:** Unit 4 of 5 · Previous: [Unit 3 – Optimization & Stability](../unit-03-optimization-and-stability/README.md) · Next: [Unit 5 – TPU Systems](../unit-05-tpu-systems-and-distributed/README.md)

## Unit Title & Ultimate Goal

**Ultimate Goal:** Treat data curation, deduplication, sequence packing, and mixture weighting as rigorous mathematical optimization problems, and connect them to Chinchilla scaling laws and critical batch size theory.

**Why this matters for 2B Dense and 5B MoE runs.** Under a fixed TPU budget $C$, the model is only one of the levers; the other two are *which tokens* and *how many*. Failure modes:

- **Duplicated data** inflates apparent progress (training loss falls faster than held-out loss) and promotes memorization.
- **Padding waste** burns FLOPs on tokens that carry no gradient signal, directly lowering effective MFU.
- **Cross-document attention** (packing without segment masks) teaches the model to attend to unrelated text.
- **Wrong batch size:** far below $B_{\text{crit}}$ wastes wall-clock; far above it wastes tokens.
- **Wrong $N$/$D$ split:** a 2B model trained on far fewer tokens than the compute-optimal count, or an MoE budgeted by total instead of active parameters, wastes the run.

## How This Unit Connects to Previous & Next Units

**Upstream (from [Unit 3](../unit-03-optimization-and-stability/README.md)).** The gradient and its noise determine how many tokens you can usefully average per step (4.3); the LR-schedule horizon requires the token budget $D$ chosen here (4.4). BPB from [1.4](../unit-01-objective-and-loss/04-bpb-and-perplexity.md) is the metric on which all data ablations are decided.

**Downstream (to [Unit 5](../unit-05-tpu-systems-and-distributed/README.md)).** The chosen global batch in tokens, $B_{\text{tokens}} = \text{per\_device\_batch\_size}\times\text{chips}\times T$, is what Unit 5 must fit into HBM, and the budget $C\approx 6ND$ is what MFU converts into wall-clock and chip-hours.

## The Unit Dependency & Implementation Table

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **4.1 Deduplication & Quality Filtering** | Removing exact/near-duplicate documents (MinHash LSH / suffix arrays) and scoring text quality via heuristics or classifiers. | **Depends on:** raw web dumps.<br>**Affects:** prevents memorization spikes, lowers validation BPB, increases data efficiency. | Retention rate %, duplicate cluster size, held-out eval loss. | [Lee et al. (2022)](https://arxiv.org/abs/2107.06499); [FineWeb (2024)](https://arxiv.org/abs/2406.17557) | Offline data-curation pipeline before Grain/TFRecord creation. |
| **4.2 Sequence Packing & Document Masking** | Concatenating multiple documents into a fixed window $T$ with `segmentation_ids` so there is zero wasted padding compute and no cross-document attention leakage. | **Depends on:** document length distribution.<br>**Affects:** effective tokens per batch and long-context boundary behavior. | Padding waste % (non-zero segmentation ratio). | [Raffel et al. (2020) – T5](https://arxiv.org/abs/1910.10683) | `input_pipeline/_grain_data_processing.py` (packing). |
| **4.3 Critical Batch Size & Gradient Noise Scale** | The threshold batch size $B_{\text{crit}}$ where doubling batch size stops cutting required training steps in half because gradient noise is already averaged out. | **Depends on:** current loss level & dataset diversity.<br>**Affects:** optimal global batch size (`per_device_batch_size` × chips) and LR scaling. | Gradient noise scale $B_{\text{noise}} = \text{tr}(\Sigma) / \lVert\mathbf{g}\rVert^2$. | [McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162) | Global batch token calculation in MaxText config. |
| **4.4 Compute-Optimal Scaling ($C \approx 6ND$)** | Mathematical law governing how to trade off active parameters ($N$) vs. training tokens ($D$) for a fixed TPU FLOP budget ($C$). | **Depends on:** architecture (Dense vs MoE) & data quality.<br>**Affects:** how big your pilot ablations should be before a full run. | IsoFLOP curves & power-law loss fit $L(N, D)$. | [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361); [Hoffmann et al. (2022) – Chinchilla](https://arxiv.org/abs/2203.15556) | Experiment planning & compute-budget calculation. |

> **Path caveat.** MaxText file/flag names drift between releases. Pin your MaxText commit in every experiment log.

## Primary Paper Lineage Breakdown

### Lee et al. (2022) – [Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)
- **Problem before:** Web-scale corpora contained many exact and near-duplicate documents and long repeated substrings; models memorized and evaluation sets leaked.
- **Intervention:** ExactSubstr (suffix-array) and NearDup (MinHash) deduplication; showed lower memorization and equal-or-better perplexity with fewer tokens.
- **Where in MaxText/JAX:** Not in MaxText; it is an offline pipeline that runs before Grain/TFRecord creation.

### Penedo et al. (2024) – [The FineWeb Datasets](https://arxiv.org/abs/2406.17557)
- **Problem before:** Open web-scale pretraining corpora lacked documented, ablation-backed filtering recipes.
- **Intervention:** Documented per-snapshot MinHash deduplication, heuristic filters, and the finding that *more aggressive global dedup is not always better*; released FineWeb-Edu (classifier-filtered).
- **Where in MaxText/JAX:** Data source for Grain input pipelines; informs your own filtering ablations.

### Raffel et al. (2020) – [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- **Problem before:** Fixed-length batches of variable-length documents required heavy padding.
- **Intervention:** Example packing, plus the cleaned C4 corpus and its heuristic filters.
- **Where in MaxText/JAX:** `input_pipeline/_grain_data_processing.py` (packing producing `inputs_segmentation` and `inputs_position`), consumed by the attention mask in `layers/attentions.py`.

### McCandlish et al. (2018) – [An Empirical Model of Large-Batch Training](https://arxiv.org/abs/1812.06162)
- **Problem before:** No principled way to choose batch size; "bigger is faster" saturates at an unknown point.
- **Intervention:** The gradient noise scale $B_{\text{noise}}=\text{tr}(\Sigma)/\lVert G\rVert^2$ predicts $B_{\text{crit}}$ and the steps/data tradeoff $\left(\frac{S}{S_{\min}}-1\right)\left(\frac{E}{E_{\min}}-1\right)=1$.
- **Where in MaxText/JAX:** Not built in; you compute it from per-microbatch gradient norms. The global batch is set via `per_device_batch_size`, number of chips, and `max_target_length`.

### Kaplan et al. (2020) – [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- **Problem before:** No predictive relationship between loss, model size, data, and compute.
- **Intervention:** Power laws $L(N)\propto N^{-\alpha_N}$, $L(D)\propto D^{-\alpha_D}$, $L(C)\propto C^{-\alpha_C}$, and $C\approx 6ND$; concluded that larger models should be favored for fixed compute.
- **Where in MaxText/JAX:** Planning layer only (spreadsheets/notebooks), not framework code.

### Hoffmann et al. (2022) – [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)
- **Problem before:** Kaplan et al.'s prescription under-trained models (partly due to a fixed schedule horizon).
- **Intervention:** Three fitting approaches showing $N_{\text{opt}}$ and $D_{\text{opt}}$ should scale roughly equally with $C$ (about 20 tokens per parameter); parametric fit $L(N,D)=E+A/N^\alpha+B/D^\beta$.
- **Where in MaxText/JAX:** Planning layer: choose `steps` $\times$ batch tokens to hit $D_{\text{opt}}$. For MoE, the correct $N$ (active vs. total) is an open question you will test (Chapter 4.4).

## Chapter Navigation Links

| Chapter | File |
|---|---|
| 4.1 Deduplication & Filtering | [01-deduplication-and-filtering.md](01-deduplication-and-filtering.md) |
| 4.2 Sequence Packing & Masking | [02-sequence-packing-and-masking.md](02-sequence-packing-and-masking.md) |
| 4.3 Critical Batch Size & Noise Scale | [03-critical-batch-size-noise-scale.md](03-critical-batch-size-noise-scale.md) |
| 4.4 Chinchilla Scaling Laws | [04-chinchilla-scaling-laws.md](04-chinchilla-scaling-laws.md) |

**Next:** [Unit 5 – TPU Systems & Distributed Training →](../unit-05-tpu-systems-and-distributed/README.md)
