# One batched GPU solve of a netlib instance under the profiler, for Nsight
# Systems. Run through benchmark/profile.sh:
#   bash profile.sh [CASE.SIF] [batch size] [sync]
# The instance goes through the benchmark pipeline (presolve, Ruiz scaling,
# standard form, cost-perturbed batch), one warmup solve compiles everything
# (NVTX range "warmup"), then a fresh solver construction and solve run under
# the NVTX range "batch <case> bs=<b>" with "init" and "solve" inside. The
# whole process is traced; jump to that range in the timeline.
# With `sync`, every annotated function synchronizes the GPU on entry and
# exit, so its range measures its own GPU work.
include("common.jl")
using NVTX

case = get(ARGS, 1, "CRE-B.SIF")
batch = parse(Int, get(ARGS, 2, "16"))
sync = "sync" in ARGS

options = (
    print_level = MadNLP.ERROR,
    tol = 1e-6,
    max_iter = 300,
    regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
    cudss_pivot_epsilon = 1e-8,
)

println("GPU: ", CUDA.name(CUDA.device()), "  case: ", case, "  batch: ", batch, "  sync: ", sync)
qp = qps_model(readqps(joinpath(fetch_netlib(), case)))
qps = build_qps(qp, batch)
gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps))

NVTX.enable_gc_hooks()
println("warmup")
NVTX.@range "warmup" timed_gpu_solve(gpu_bnlp; options...)

if sync
    isdefined(MadIPM, :_PROFILE_SYNC_HOOK) ||
        error("sync needs the annotated tree; run through profile.sh")
    MadIPM._PROFILE_SYNC_HOOK[] = CUDA.synchronize
end
GC.gc(true); GC.gc(true)

println("profiled run")
stats = nothing
NVTX.@range "batch $case bs=$batch" begin
    solver = NVTX.@range "init" MadIPM.UniformBatchMPCSolver(
        gpu_bnlp;
        uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.LDL,
        options...
    )
    global stats = NVTX.@range "solve" MadIPM.solve!(solver)
    CUDA.synchronize()
end
println("converged ", count(==(MadNLP.SOLVE_SUCCEEDED), stats.status), "/", batch,
        ", mean iterations ", sum(stats.iter) / batch)
