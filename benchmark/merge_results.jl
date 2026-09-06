# Join the CPU-side and GPU-side CSVs of a benchmark run with `--side=cpu` and
# `--side=gpu` into the combined layout: rows matched by case name, the CPU
# blocks taken from the cpu file and the GPU blocks from the gpu file.
# Applies to the per-batch-size layout of 1-scalability, 2-performance and
# the DC-OPF benchmark job (name, nvar, ncon, nnzj, then 10 columns per batch
# size: 6 CPU, 4 GPU).
#
#   julia merge_results.jl results/2-benchmark-netlib
#   -> reads results/2-benchmark-netlib-{cpu,gpu}.csv, writes results/2-benchmark-netlib.csv
using DelimitedFiles

function merge_results(stem)
    cpu = readdlm("$stem-cpu.csv")
    gpu = readdlm("$stem-gpu.csv")
    size(cpu, 2) == size(gpu, 2) || error("column layouts differ: $(size(cpu, 2)) vs $(size(gpu, 2))")
    shift = 3
    (size(cpu, 2) - 1 - shift) % 10 == 0 || error("not a per-batch-size layout")
    nb = (size(cpu, 2) - 1 - shift) ÷ 10
    out = copy(gpu)
    for (k, case) in enumerate(cpu[:, 1])
        j = findfirst(==(case), gpu[:, 1])
        j === nothing && error("$case is in the cpu file but not in the gpu file")
        out[j, 2:1+shift] .= cpu[k, 2:1+shift]
        for l in 1:nb
            cols = 1 + shift + 10*(l-1) .+ (1:6)
            out[j, cols] .= cpu[k, cols]
        end
    end
    missing_cases = setdiff(gpu[:, 1], cpu[:, 1])
    isempty(missing_cases) || @warn "cases in the gpu file without cpu results (kept as -1)" missing_cases
    writedlm("$stem.csv", out)
    println("wrote $stem.csv ($(size(out, 1)) rows)")
end

merge_results(ARGS[1])
