#!/usr/bin/env bash
# Profile one batched GPU solve with Nsight Systems (Linux, GNU sed).
#
#   bash profile.sh [CASE.SIF] [batch size] [sync]      defaults: CRE-B.SIF 16
#
# First run: instruments the tree with the NVTX range macro from
# _sync_annotate.jl (every column-0 `function` in src/ except src/models/, and
# in ext/MadIPMCUDAExt/), adds `using NVTX` to those modules and NVTX to both
# projects, and marks the tree with .isannotated. Later runs reuse it. Undo:
#   rm .isannotated && git checkout -- src ext Project.toml benchmark/Project.toml
# The whole process is captured (a cudaProfilerApi capture range gets closed
# early by a profiler stop from inside CUDA.jl's kernel launch path); the
# NVTX ranges "warmup" and "batch <case> bs=<b>" (with "init" and "solve"
# inside) mark the two solves in the timeline.
# Output: benchmark/profiles/<case>-bs<batch>[-sync].nsys-rep plus a short
# summary of the NVTX ranges and CUDA kernels on stdout.
set -euo pipefail
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$BENCH_DIR/.." && pwd)"
FLAG="$REPO_DIR/.isannotated"

if [ ! -f "$FLAG" ]; then
    echo "annotating $REPO_DIR"
    cp "$BENCH_DIR/_sync_annotate.jl" "$REPO_DIR/src/_sync_annotate.jl"
    # functions named *_kernel! are device code launched with @cuda (or their
    # launchers) and must stay untouched: NVTX calls do not compile for the GPU
    find "$REPO_DIR/src" -name '*.jl' -not -path '*/models/*' -not -name '_sync_annotate.jl' \
        -exec sed -i '/^function [^(]*_kernel!(/! s/^function /MadIPM.@sync_annotate function /' {} +
    find "$REPO_DIR/ext/MadIPMCUDAExt" -name '*.jl' \
        -exec sed -i '/^function [^(]*_kernel!(/! s/^function /MadIPM.@sync_annotate function /' {} +
    # header insertions are guarded so a run that failed before the flag was
    # written cannot double them
    grep -q '_sync_annotate' "$REPO_DIR/src/MadIPM.jl" || sed -i '/^module MadIPM$/a\
using NVTX\
include("_sync_annotate.jl")' "$REPO_DIR/src/MadIPM.jl"
    grep -q '^using NVTX' "$REPO_DIR/ext/MadIPMCUDAExt/MadIPMCUDAExt.jl" || sed -i '/^import MadNLP$/a\
using NVTX' "$REPO_DIR/ext/MadIPMCUDAExt/MadIPMCUDAExt.jl"
    julia --project="$REPO_DIR" -e 'using Pkg; Pkg.add("NVTX")'
    # resolve: the benchmark manifest must pick up NVTX as a new dependency of MadIPM
    julia --project="$BENCH_DIR" -e 'using Pkg; Pkg.add("NVTX"); Pkg.resolve(); Pkg.precompile()'
    touch "$FLAG"
    echo "annotated"
fi

CASE=${1:-CRE-B.SIF}
BS=${2:-16}
SYNC=${3:-}
mkdir -p "$BENCH_DIR/profiles"
OUT="$BENCH_DIR/profiles/${CASE%.SIF}-bs${BS}${SYNC:+-sync}"

cd "$BENCH_DIR"
nsys profile \
    --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --force-overwrite=true -o "$OUT" \
    julia --project="$BENCH_DIR" "$BENCH_DIR/nsys_batch.jl" "$CASE" "$BS" $SYNC

echo "report: $OUT.nsys-rep"
nsys stats --report nvtx_sum,cuda_gpu_kern_sum --format table --force-export=true "$OUT.nsys-rep" 2>/dev/null \
    | grep -v "^$" | head -70
