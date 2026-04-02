#!/bin/bash
# Quick A/B test: RAYON_NUM_THREADS=6 vs 12
# Runs full proof, captures per-shard times from stderr.
# Kill with Ctrl-C after ~20 shards to compare.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Swap status ==="
sysctl vm.swapusage

COMMON_ENV="SP1_DEV=true FRI_QUERIES=21 SHARD_SIZE=1048576 SHARD_BATCH_SIZE=1 MALLOC_NANO_ZONE=0 RUST_LOG=info,p3_fri=off"

THREADS=${1:-6}
echo ""
echo "=== Running with RAYON_NUM_THREADS=$THREADS ==="
echo "Press Ctrl-C after ~20 shards complete for quick comparison."
echo ""

env $COMMON_ENV RAYON_NUM_THREADS=$THREADS \
    cargo run --release -- --prove 2>&1 | tee "/tmp/sp1_rayon_${THREADS}.log"
