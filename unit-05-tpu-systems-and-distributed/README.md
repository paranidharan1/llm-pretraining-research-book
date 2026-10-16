# Unit 5: TPU Systems and Distributed Training

> **Book position:** Unit 5 of 5 · Previous: [Unit 4 – Data & Scaling Laws](../unit-04-data-and-scaling-laws/README.md) · Back to: [Root Index](../README.md)

## Unit Title & Ultimate Goal

**Ultimate Goal:** Translate mathematical operations into physical TPU meshes, understand when XLA inserts All-Gather, Reduce-Scatter, or All-to-All collectives, and maximize Model FLOPs Utilization (MFU).

**Why this matters for 2B Dense and 5B MoE runs.** The math of Units 1–4 fixes *how many FLOPs* a run needs ($C\approx 6ND$); this unit determines *how fast* those FLOPs execute and *whether they fit at all*. Failure modes:

- **OOM at the wrong moment.** A 5B-total MoE holds FP32 weights (20 GB) plus two FP32 Adam moments (40 GB) plus gradients before any activation is stored; unsharded, this overflows a single chip's HBM.
- **Exposed communication.** If an All-Gather or All-to-All is not overlapped with compute, MXUs idle and MFU collapses, with no change in loss.
- **MoE All-to-All bottleneck.** Expert parallelism routes tokens across chips; a poorly chosen `expert` axis turns the router into a network benchmark.
- **Remat misconfiguration.** Too little remat means a small batch (poor MFU); too much wastes up to roughly one extra forward pass of FLOPs.

## How This Unit Connects to Previous & Next Units

**Upstream (from [Units 2–4](../unit-04-data-and-scaling-laws/README.md)).** Layer shapes (Unit 2), optimizer-state sizes (Unit 3), and the global batch/token budget (Unit 4) are the *inputs* to the sharding problem: they determine tensor sizes, the number of bytes per parameter, and the total compute to be scheduled.

**Downstream.** This is the last unit. Its outputs feed back into the experiment loop: measured MFU and tokens/sec/chip turn a FLOP budget into chip-hours, which sets how many ablations fit (see [`templates/EXPERIMENT_LOG_TEMPLATE.md`](../templates/EXPERIMENT_LOG_TEMPLATE.md)); and exposed-communication profiles drive the MFU branch of [`templates/DEBUGGING_DECISION_TREE.md`](../templates/DEBUGGING_DECISION_TREE.md).

## The Unit Dependency & Implementation Table

| Topic / Node | Core Definition (What it is) | Depends On → Affects | Measured By (Metrics) | Origin Paper (Where it came from) | Spot in MaxText / XLA Stack |
|---|---|---|---|---|---|
| **5.1 Sharding Mesh (FSDP, Tensor, Expert)** | Splitting weights, optimizer states, and activations across TPU axes (`data`, `fsdp`, `tensor`, `expert`) so a 5B+ model fits in HBM. | **Depends on:** TPU slice topology (e.g., v5e/v5p ICI links).<br>**Affects:** HBM memory per chip and communication overhead. | Peak HBM usage (GB) & ICI link utilization. | [Rajbhandari et al. (2020) – ZeRO](https://arxiv.org/abs/1910.02054); [GSPMD (2021)](https://arxiv.org/abs/2105.04663) | `MaxText/layers/quantizations.py` & `sharding.py` (`ici_fsdp_parallelism`, etc.). |
| **5.2 Collectives (All-Gather, Reduce-Scatter, All-to-All)** | The exact network operations XLA inserts to gather weights before a layer (All-Gather), sum/shard gradients (Reduce-Scatter), or route tokens to MoE experts (All-to-All). | **Depends on:** sharding annotations.<br>**Affects:** whether your TPU MXUs sit idle waiting for network transfers. | % step time spent in exposed communication (XProf). | [GShard (Lepikhin et al., 2020)](https://arxiv.org/abs/2006.16668) | XLA compiler GSPMD pass & TensorBoard XProf trace. |
| **5.3 Rematerialization (Activation Checkpointing) & MFU** | Discarding intermediate forward activations and recomputing them during backprop to save HBM, trading extra FLOPs for larger batch sizes and higher Model FLOPs Utilization (MFU). | **Depends on:** HBM capacity & `remat_policy`.<br>**Affects:** step time, max sequence length $T$, and hardware efficiency. | MFU % (achieved TFLOP/s ÷ peak TPU TFLOP/s) & tokens/sec/chip. | [Chen et al. (2016)](https://arxiv.org/abs/1604.06174); [PaLM (2022)](https://arxiv.org/abs/2204.02311) (MFU) | `config.remat_policy` & `max_utils.py` TFLOP calculator. |

> **Path caveat.** MaxText file/flag names drift between releases (for example, sharding rules live in config `logical_axis_rules` and `mesh_axes`). Pin your MaxText commit in every experiment log.

## Primary Paper Lineage Breakdown

### Rajbhandari et al. (2020) – [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **Problem before:** Data parallelism replicated parameters, gradients, and optimizer states on every device (16 bytes/parameter under mixed-precision Adam), which capped model size.
- **Intervention:** Partition optimizer states (stage 1), gradients (stage 2), and parameters (stage 3) across data-parallel ranks, gathering parameters just-in-time. This is what "FSDP" implements.
- **Where in MaxText/JAX:** The `fsdp` mesh axis (`ici_fsdp_parallelism`, `dcn_fsdp_parallelism`); parameters are sharded and XLA inserts All-Gather before use and Reduce-Scatter on gradients.

### Xu et al. (2021) – [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/abs/2105.04663)
- **Problem before:** Parallelism was hand-written per model (explicit sends/receives).
- **Intervention:** Annotate a few tensors with shardings; the XLA compiler propagates shardings and inserts the collectives automatically.
- **Where in MaxText/JAX:** `jax.sharding.Mesh`, `NamedSharding`/`PartitionSpec`, and `with_sharding_constraint`; MaxText maps *logical axes* (`embed`, `mlp`, `heads`) to *mesh axes* via `logical_axis_rules`.

### Lepikhin et al. (2020) – [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding](https://arxiv.org/abs/2006.16668)
- **Problem before:** Sparse MoE did not scale across many accelerators with simple code.
- **Intervention:** Expert-parallel MoE with dispatch/combine via All-to-All, expressed through sharding annotations.
- **Where in MaxText/JAX:** The `expert` mesh axis (`ici_expert_parallelism`) and the dispatch/combine einsums or ragged-all-to-all path in `layers/moe.py`.

### Chen et al. (2016) – [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- **Problem before:** Activation memory scaled as $O(L)$ and limited batch size and depth.
- **Intervention:** Checkpoint $\sqrt{L}$ activations and recompute the rest in the backward pass: $O(\sqrt{L})$ memory for about one extra forward pass.
- **Where in MaxText/JAX:** `jax.checkpoint` (`jax.remat`) with the `remat_policy` config choosing which intermediates to save (`full`, `minimal`, `save_dot_with_no_batch_dims`, etc.).

### Chowdhery et al. (2022) – [PaLM: Scaling Language Modeling with Pathways](https://arxiv.org/abs/2204.02311)
- **Problem before:** Hardware utilization was reported inconsistently (often counting recomputation as useful work).
- **Intervention:** Defined **Model FLOPs Utilization (MFU)** as observed throughput divided by theoretical peak, counting only the model FLOPs of the forward and backward pass (not rematerialization), and demonstrated 46.2% MFU on TPU v4 at 540B scale.
- **Where in MaxText/JAX:** MaxText logs TFLOP/s per device from its own FLOP calculator (`max_utils.py`); you divide by the chip's peak to get MFU.

## Chapter Navigation Links

| Chapter | File |
|---|---|
| 5.1 Sharding: Mesh, FSDP, Tensor, Expert | [01-sharding-mesh-fsdp-tensor-expert.md](01-sharding-mesh-fsdp-tensor-expert.md) |
| 5.2 XLA Collectives & Communication | [02-xla-collectives-communication.md](02-xla-collectives-communication.md) |
| 5.3 Remat, Checkpointing & MFU | [03-remat-checkpointing-and-mfu.md](03-remat-checkpointing-and-mfu.md) |

**Back to:** [Root Index](../README.md)
