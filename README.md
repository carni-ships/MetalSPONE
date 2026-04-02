# MetalSPONE 🥄✨

**SP1 Proof Generation Optimization for Apple Silicon — 15x faster at production soundness, 56x faster at dev config, on an 18GB M3 Pro**

This is a fork of [SP1 v4.0.0](https://github.com/succinctlabs/sp1) with systematic **proof generation** optimizations developed over 23 sessions, reducing the Fibonacci benchmark from **50.5 seconds to 3.4 seconds at production FRI config** (FRI\_QUERIES=100, ~100 bits soundness) and **0.9 seconds at dev config** (FRI\_QUERIES=1).

> **Note:** All timings in this project refer to **proof generation** (the computationally expensive step), not proof verification. Verification is near-instant: ~1-10ms off-chain, ~150K gas on-chain (Groth16 on Ethereum). See [Proof Generation vs Verification](#proof-generation-vs-verification) for details.

A detailed research report is available in [`docs/SP1_Optimization_Report.md`](docs/SP1_Optimization_Report.md) (also in [PDF](docs/SP1_Optimization_Report.pdf) and [LaTeX](docs/SP1_Optimization_Report.tex)).

## Results

All metrics below are for **proof generation** (the prover). Verification costs are unchanged.

| Metric | Original SP1 | MetalSPONE (prod) | MetalSPONE (dev) |
|--------|-------------|-------------------|------------------|
| Proof generation time | 50.5s | **3.4s** (15x) | **0.9s** (56x) |
| FRI soundness | ~100 bits | ~100 bits | ~4 bits |
| Peak memory (prover) | ~10 GB | ~7 GB | ~7 GB |
| Compress proofs generated | 9 | 1 | 1 |
| Circuit padding waste | 140-150% | <5% | <2% |
| Prover startup time | 85s | <1s | <1s |

| FRI Config | Soundness | Median Time | Speedup |
|------------|-----------|-------------|---------|
| `FRI_QUERIES=100, LOG_BLOWUP=1` | ~100 bits | **3.4s** | 15x |
| `FRI_QUERIES=50, LOG_BLOWUP=1` | ~66 bits | **2.6s** | 19x |
| `FRI_QUERIES=33, LOG_BLOWUP=2` | ~82 bits | **1.7s** | 30x |
| `FRI_QUERIES=10, LOG_BLOWUP=3` | ~46 bits | **1.1s** | 46x |
| `FRI_QUERIES=1, LOG_BLOWUP=4` | ~4 bits | **0.9s** | 56x |

*Benchmarked on Apple M3 Pro (12 CPU cores, 18 GPU cores, 18GB RAM). All configs: SHARD\_BATCH\_SIZE=3, VERIFY\_VK=false, METAL\_DFT=1.*

## Proof Generation vs Verification

ZK proof systems have two distinct operations with vastly different computational costs:

| | Proof Generation (Prover) | Proof Verification (Verifier) |
|---|---|---|
| **What it does** | Generates a cryptographic proof that a computation was executed correctly | Checks that a proof is valid |
| **Computational cost** | O(n) — proportional to computation size | O(log n) — logarithmic, near-instant |
| **Typical time** | Seconds to minutes | Milliseconds |
| **This project** | **50.5s → 0.8s (63x faster)** | Unchanged (~1-10ms off-chain) |
| **On-chain (Ethereum)** | N/A (done off-chain) | ~150K gas (Groth16) / ~350K gas (PLONK) |
| **Hardware needs** | Multi-core CPU + GPU, GBs of RAM | Minimal — runs on any device |

**This project optimizes proof generation only.** Verification is inherently cheap and was already fast enough — there is no meaningful optimization to be done there.

## Production Readiness

All FRI configs run successfully on 18GB, including full production soundness:

| Config | Soundness | Proof Gen Time | Use Case |
|--------|-----------|---------------|----------|
| `FRI_QUERIES=1, LOG_BLOWUP=4` | ~4 bits | **0.9s** | Dev / benchmarking only |
| `FRI_QUERIES=10, LOG_BLOWUP=3` | ~46 bits | **1.1s** | Light security |
| `FRI_QUERIES=33, LOG_BLOWUP=2` | ~82 bits | **1.7s** | Moderate security |
| `FRI_QUERIES=50, LOG_BLOWUP=1` | ~66 bits | **2.6s** | Good security |
| `FRI_QUERIES=100, LOG_BLOWUP=1` | ~100+ bits | **3.4s** | **Full production** |

All core optimizations — GPU acceleration, shape tuning, batching, LDE caching, batch inversion, lazy program construction — apply at every FRI config level.

### Production Throughput

On a single M3 Pro at production FRI config (3.4s per proof):

| Use Case | Throughput Needed | Single M3 Pro | Notes |
|----------|------------------|---------------|-------|
| On-chain attestation (periodic) | 1 proof/min | **Sufficient** | ~17 proofs/min capacity |
| Bridge (per-block, 12s slots) | 5 proofs/min | **Sufficient** | ~3.5x headroom |
| High-throughput rollup | 10+ proofs/min | **Sufficient** | Up to ~17 proofs/min |
| Real-time proving | Sub-second | Borderline | 0.9s at dev config, 1.7s at q=33 |

### Scaling Strategies

Proof generation is embarrassingly parallel — each proof is independent:

- **Horizontal scaling:** Run N prover instances on N machines for Nx throughput. All per-prover optimizations multiply with the number of instances.
- **Bigger hardware:** M3 Max/Ultra (more GPU cores, more RAM) or server GPUs (NVIDIA A100) would reduce per-proof time further.
- **Proof aggregation:** Batch multiple transactions into a single proof to amortize the fixed proving overhead.

A typical production deployment would run multiple prover instances behind a job queue, with each instance benefiting from the full set of MetalSPONE optimizations.

## What Changed

### Metal GPU Acceleration (Proof Generation)
- **MetalMmcs** — GPU-accelerated Poseidon2 Merkle tree hashing for all polynomial commitments
- **MetalDft** — GPU NTT/DFT via custom Metal kernels (DIF radix-4/8, column-tiled, fused bitrev+scale+zero\_pad)
- GPU constraint evaluation path (behind `METAL_CONSTRAINTS=1`, beneficial for larger circuits)

### FRI / PCS Open Optimizations (Proof Generation)
- Eliminated redundant LDE recomputation in both query phase and open phase
- `open_sequential_cached_split()` — preprocessed round as read-only reference, avoiding 3.5s clone
- Parallel Lagrange interpolation and inverse denominator computation
- Precomputed `scaled_inv_denoms` and `beta_powers` in permutation trace generation

### Shape Tuning (Proof Generation)
- **4 custom core shapes** with tight chip heights (Global=15, non-critical chips at 2^4-2^12)
- **10 recursion shapes** including a micro shape (BatchFRI=12, MemoryVar=14) for FRI\_QUERIES<=1
- Flexible preprocessed chip heights [10..22] (Program chip: 2^14, down from 2^19)

### Constraint and Trace Optimizations (Proof Generation)
- Cross-row batch inversion (4096-block) for permutation trace generation (-15.6% CPU)
- Precomputed `local_sends`/`local_receives` in Chip struct (-9% CPU)
- O(N^2) to O(N) prefix/suffix products in constraint evaluation
- Disabled debug backtraces in recursion compiler (2.7x speedup)

### Compress Tree (Proof Generation)
- `first_layer_batch_size=2`, `REDUCE_BATCH_SIZE=3` — reduces 9 compress proofs to 1
- PK cache in compress workers
- **Lazy program construction** — disabled eager precomputation of all shape combinations (11^3 = 1331 programs), which was causing OOM on 18GB at production FRI configs. Programs now built on-demand with negligible overhead (~50ms per unique shape).

### FRI Parameter Overrides
- `FRI_QUERIES` env var (default 100) — controls number of FRI query openings
- `LOG_BLOWUP` env var (default 1) — controls LDE blowup factor
- `POW_BITS` env var (default 16) — controls proof-of-work difficulty

### Build
- Fat LTO, `panic=abort`, `codegen-units=1`, `opt-level=3`, `-C target-cpu=native`
- jemalloc allocator (via tikv-jemallocator)

## Proof Generation Timeline

```
Baseline .......... 50.5s  (1x)
+ Metal GPU ....... 31.6s  (1.6x)   Sessions 1-5
+ Compress batch .. 11.0s  (4.6x)   Sessions 5-8
+ Backtrace fix ...  6.5s  (7.8x)   Sessions 9-14
+ Shape tuning ....  1.49s (33.9x)  Sessions 16-18
+ Preprocessed ....  0.9s  (56.1x)  Session 19
+ Micro shape .....  0.8s  (63.1x)  Session 20
  Final assessment .  0.8s  (floor)  Session 21
+ Prod config fix .  3.2s  (q=100)  Session 23
```

## Proof Generation Breakdown

### Dev config (FRI_QUERIES=1, 0.9s median)
```
Total: ~0.9s
+-- prove_core: ~0.5s (56%)
|   +-- trace gen:       40ms
|   +-- perm_trace:      20ms
|   +-- quotient:       100ms
|   +-- pcs_open:       150ms
+-- compress: ~0.35s (39%)
|   +-- setup:           40ms
|   +-- commit:          50ms
|   +-- open:           165ms
+-- overhead: ~0.05s (5%)
```

### Production config (FRI_QUERIES=100, 3.4s median)
```
Total: ~3.4s
+-- prove_core: ~0.5s (15%)
+-- compress: ~2.7s (79%)
|   +-- setup:          220ms
|   +-- commit:         350ms
|   +-- open:          1800ms
|       +-- perm_trace:   150ms
|       +-- perm_commit:  300ms
|       +-- quotient:     380ms
|       +-- q_commit:     180ms
|       +-- pcs_open:     770ms
+-- overhead: ~0.2s (6%)
```

## Quick Start

```bash
# Build with release optimizations
cargo build --release -p sp1-prover

# Run at production FRI config (q=100, ~100 bits soundness, ~3.2s)
SHARD_BATCH_SIZE=3 VERIFY_VK=false FRI_QUERIES=100 LOG_BLOWUP=1 METAL_DFT=1 \
  cargo test --release -p sp1-prover -- bench_compress --nocapture

# Run at dev config (q=1, fastest, ~0.9s)
SHARD_BATCH_SIZE=3 VERIFY_VK=false FRI_QUERIES=1 LOG_BLOWUP=4 METAL_DFT=1 \
  cargo test --release -p sp1-prover -- bench_compress --nocapture
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FRI_QUERIES` | 100 | Number of FRI query openings (1 = fastest, ~4 bits soundness) |
| `LOG_BLOWUP` | 1 | LDE blowup factor (4 is optimal at FRI\_QUERIES=1) |
| `POW_BITS` | 16 | Proof-of-work difficulty bits |
| `SHARD_BATCH_SIZE` | 1 | Core trace batch size |
| `VERIFY_VK` | true | Verify recursion VK map (false skips 85s startup) |
| `METAL_DFT` | 0 | Enable Metal GPU DFT (macOS only) |
| `METAL_CONSTRAINTS` | 0 | Enable Metal GPU constraint evaluation |
| `SP1_PROGRAM_CACHE` | false | Eagerly precompute all compress programs (uses more RAM) |

## Project Structure

Key modified files from upstream SP1:

```
crates/stark/src/bb31_poseidon2.rs     # FRI config with env var overrides
crates/stark/src/prover.rs             # GPU constraint eval, phase timing
crates/stark/src/permutation.rs        # Cross-row batch inversion
crates/stark/src/chip.rs               # Precomputed interactions
crates/core/machine/src/shape/         # Custom core shapes (small_shapes.json, mod.rs)
crates/recursion/core/src/shape.rs     # 10 recursion shapes including micro
crates/p3-fri/src/two_adic_pcs.rs      # Cached LDEs, parallel interp, cached_split
crates/prover/src/lib.rs               # Compress batching, PK cache, bench_compress
docs/                                  # Optimization report (md, pdf, latex)
```

## Based On

- [SP1](https://github.com/succinctlabs/sp1) v4.0.0 by Succinct Labs
- [Plonky3](https://github.com/Plonky3/Plonky3) — the underlying STARK proving toolkit
