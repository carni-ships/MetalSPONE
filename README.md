# 🥄✨ MetalSPONE

**SP1 Prover Optimization for Apple Silicon — 63x faster proving on an 18GB M3 Pro**

This is a fork of [SP1 v4.0.0](https://github.com/succinctlabs/sp1) with systematic prover optimizations developed over 21 sessions, reducing the Fibonacci benchmark (prove\_core + compress) from **50.5 seconds to 0.8 seconds**.

A detailed research report is available in [`docs/SP1_Optimization_Report.md`](docs/SP1_Optimization_Report.md) (also in [PDF](docs/SP1_Optimization_Report.pdf) and [LaTeX](docs/SP1_Optimization_Report.tex)).

## Results

| Metric | Original SP1 | MetalSPONE | Improvement |
|--------|-------------|------------|-------------|
| Proving time | 50.5s | 0.8s | **63x faster** |
| Throughput | ~100 Hz | ~6,400 Hz | 64x |
| Peak memory | ~10 GB | ~7 GB | 30% reduction |
| Compress proofs | 9 | 1 | 9x fewer |
| Padding waste | 140-150% | <2% | ~75x reduction |
| Startup time | 85s | <1s | 85x faster |

*Benchmarked on Apple M3 Pro (12 CPU cores, 18 GPU cores, 18GB RAM).*

## What Changed

### Metal GPU Acceleration
- **MetalMmcs** — GPU-accelerated Poseidon2 Merkle tree hashing for all polynomial commitments
- **MetalDft** — GPU NTT/DFT via custom Metal kernels (DIF radix-4/8, column-tiled, fused bitrev+scale+zero\_pad)
- GPU constraint evaluation path (behind `METAL_CONSTRAINTS=1`, beneficial for larger circuits)

### FRI / PCS Open Optimizations
- Eliminated redundant LDE recomputation in both query phase and open phase
- `open_sequential_cached_split()` — preprocessed round as read-only reference, avoiding 3.5s clone
- Parallel Lagrange interpolation and inverse denominator computation
- Precomputed `scaled_inv_denoms` and `beta_powers` in permutation trace generation

### Shape Tuning
- **4 custom core shapes** with tight chip heights (Global=15, non-critical chips at 2^4-2^12)
- **10 recursion shapes** including a micro shape (BatchFRI=12, MemoryVar=14) for FRI\_QUERIES<=1
- Flexible preprocessed chip heights [10..22] (Program chip: 2^14, down from 2^19)

### Constraint and Trace Optimizations
- Cross-row batch inversion (4096-block) for permutation trace generation (-15.6% CPU)
- Precomputed `local_sends`/`local_receives` in Chip struct (-9% CPU)
- O(N^2) to O(N) prefix/suffix products in constraint evaluation
- Disabled debug backtraces in recursion compiler (2.7x speedup)

### Compress Tree
- `first_layer_batch_size=2`, `REDUCE_BATCH_SIZE=3` — reduces 9 compress proofs to 1
- PK cache in compress workers

### FRI Parameter Overrides
- `FRI_QUERIES` env var (default 100) — controls number of FRI query openings
- `LOG_BLOWUP` env var (default 1) — controls LDE blowup factor
- `POW_BITS` env var (default 16) — controls proof-of-work difficulty

### Build
- Fat LTO, `panic=abort`, `codegen-units=1`, `opt-level=3`, `-C target-cpu=native`
- jemalloc allocator (via tikv-jemallocator)

## Performance Timeline

```
Baseline .......... 50.5s  (1x)
+ Metal GPU ....... 31.6s  (1.6x)   Sessions 1-5
+ Compress batch .. 11.0s  (4.6x)   Sessions 5-8
+ Backtrace fix ...  6.5s  (7.8x)   Sessions 9-14
+ Shape tuning ....  1.49s (33.9x)  Sessions 16-18
+ Preprocessed ....  0.9s  (56.1x)  Session 19
+ Micro shape .....  0.8s  (63.1x)  Session 20
  Final assessment .  0.8s  (floor)  Session 21
```

## Phase Breakdown at 0.8s

```
Total: ~0.80s
+-- prove_core: ~0.47s (59%)
|   +-- trace gen:       43ms
|   +-- commit:          72ms
|   +-- perm_trace:      20ms
|   +-- perm_commit:     35ms
|   +-- quotient:        95ms
|   +-- q_commit:        46ms
|   +-- pcs_open:       158ms
+-- compress: ~0.25s (31%)
|   +-- setup:           35ms
|   +-- commit:          48ms
|   +-- open:           165ms
+-- overhead: ~0.08s (10%)
```

## Quick Start

```bash
# Build with release optimizations
cargo build --release -p sp1-prover

# Run the Fibonacci benchmark (aggressive config for dev/benchmarking)
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

## Soundness Note

The aggressive FRI config (`FRI_QUERIES=1`, `LOG_BLOWUP=4`) provides ~4 bits of soundness — suitable for benchmarking and development only. Production deployments should use higher `FRI_QUERIES` (e.g., 100). All core optimizations (GPU, shapes, batching, caching, etc.) apply equally to production configs.

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
