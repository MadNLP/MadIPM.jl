#!/usr/bin/env bash
# Set up a fresh GPU box (Ubuntu 22.04 + CUDA 12 image, run as root) for the
# benchmarks: Julia 1.12 via juliaup, the MadIPM checkout, the benchmark
# environment, then a GPU / data / HSL check.
#
#   setup_pod.sh [archive.tar.gz]      default: ~/MadIPM.tar.gz
#
# Make the archive from the branch to benchmark and copy it over:
#   git archive --format=tar.gz --prefix=MadIPM/ -o MadIPM.tar.gz <branch>
#   scp MadIPM.tar.gz benchmark/setup_pod.sh benchmark/run_pod.sh root@<ip>:
#   ssh root@<ip> bash setup_pod.sh
set -euo pipefail

archive=${1:-$HOME/MadIPM.tar.gz}

export DEBIAN_FRONTEND=noninteractive
SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
$SUDO apt-get update -qq && $SUDO apt-get install -y -qq git curl tar > /dev/null

if [ ! -x "$HOME/.juliaup/bin/julia" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12 > /dev/null
fi
export PATH="$HOME/.juliaup/bin:$PATH"
julia --version

rm -rf "$HOME/MadIPM"
tar xzf "$archive" -C "$HOME"
cd "$HOME/MadIPM/benchmark"

# MadIPM comes from the checkout through [sources] path = ".."
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 2>&1 | tail -3

julia --project=. -e '
using CUDA; @assert CUDA.functional() "CUDA not functional"; println("GPU: ", CUDA.name(CUDA.device()))
using QPSReader; println("netlib: ", fetch_netlib())
using HSL; println("libHSL functional: ", LIBHSL_isfunctional(), " (--cpu-solver=auto falls back to MUMPS otherwise)")'
echo "SETUP DONE"
