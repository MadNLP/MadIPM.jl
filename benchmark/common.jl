
using DelimitedFiles
using Printf
using MadIPM, MadNLP
using MadNLPHSL
using NLPModels
using MadNLPGPU, CUDA, KernelAbstractions
using Random, Distributions, SparseArrays, Memoize
using Statistics
using SparseMatricesCOO: SparseMatrixCOO
using QPSReader
using Adapt

import MadIPM.Models: LPData, QPData, ScalarModel, LinearModel, QuadraticModel,
    ObjRHSBatchQuadraticModel, operator_sparse_matrix
const BQMS = MadIPM.Models.Scaling
const BQMP = MadIPM.Models.Presolve

function refresh_memory()
    CUDA.reclaim()
    GC.gc(true)
    CUDA.reclaim()
    return
end

# QPSReader sets `objsense = :notset` for files without an explicit OBJSENSE;
# treat that as minimize per the LP convention, only flip on explicit `:max`.
_minimize(qps::QPSData) = (qps.objsense != :max)
_name(qps::QPSData) = qps.name === nothing ? "QPSData" : qps.name

function qps_model(qps::QPSData)
    nvar, ncon = length(qps.lvar), length(qps.lcon)
    A = SparseMatrixCOO(ncon, nvar, qps.arows, qps.acols, qps.avals)
    H = SparseMatrixCOO(nvar, nvar, qps.qrows, qps.qcols, qps.qvals)
    data = QPData(A, qps.c, H;
        lvar = qps.lvar, uvar = qps.uvar,
        lcon = qps.lcon, ucon = qps.ucon,
        c0   = qps.c0)
    return QuadraticModel(data; minimize = _minimize(qps), name = _name(qps))
end

function scale_qp(qp::ScalarModel)
    _, scaling = BQMS.ruiz_equilibration(operator_sparse_matrix(qp.data.A))
    return BQMS.scale_model(qp, scaling.row, scaling.col)
end

_A_coo(qp::ScalarModel) = operator_sparse_matrix(qp.data.A)
_Q_coo(qp::QuadraticModel) = operator_sparse_matrix(qp.data.Q)
_Q_coo(qp::LinearModel{T}) where {T} =
    SparseMatrixCOO(qp.meta.nvar, qp.meta.nvar, Int[], Int[], T[])

function _shared_matrix_qp(base::ScalarModel, c, lcon, ucon, lvar, uvar)
    data = QPData(_A_coo(base), c, _Q_coo(base);
        lcon = lcon, ucon = ucon,
        lvar = lvar, uvar = uvar,
        c0 = base.data.c0[1])
    return QuadraticModel(data;
        x0 = copy(base.meta.x0), minimize = base.meta.minimize,
        name = base.meta.name)
end

function build_qps(base_qp, batch_size; shift_c=true, shift_b=false)
    base_pqp, pstatus = MadIPM.presolve_qp(base_qp)
    @assert pstatus == BQMP.PRESOLVE_UNCHANGED || pstatus == BQMP.PRESOLVE_REDUCED "Presolve declared the instance unsolvable ($pstatus)"

    base_sqp = MadIPM.standard_form_qp(scale_qp(base_pqp))

    n = base_sqp.meta.nvar
    m = base_sqp.meta.ncon

    c_base = copy(base_sqp.data.c)
    lcon0 = copy(base_sqp.meta.lcon)
    ucon0 = copy(base_sqp.meta.ucon)

    return [begin
        rng = Xoshiro(i)

        c_noise = shift_c ? rand(rng, Uniform(0.99, 1.01), n) : 1.0
        c_new = c_base .* c_noise

        shift = shift_b ? randn(rng, m) .* (0.02 .* max.(abs.(lcon0), 1.0)) : 0.0
        lcon_new = lcon0 .+ shift
        ucon_new = ucon0 .+ shift

        _shared_matrix_qp(
            base_sqp, c_new, lcon_new, ucon_new,
            copy(base_sqp.meta.lvar), copy(base_sqp.meta.uvar),
        )
    end for i in 1:batch_size]
end

to_gpu(bnlp) = Adapt.adapt(CuArray, bnlp)

#=
    Timing

CPU and GPU report the same two phases, both read from a wall clock:
  init  — solver construction (KKT allocation, symbolic analysis)
  solve — `solve!`: scaling, initial point, and the IPM iterations
The solvers' internal counters are not comparable with each other (the scalar
solver starts its clock after the initial point, the batch solver in its
constructor), so they are not used. GPU work is synchronized before the clock
is read. Presolve, scaling and the host-to-device transfer happen once per
instance, outside both timers.
=#

function timed_cpu_solve(qp; linear_solver, options...)
    t_init = @elapsed solver = MadIPM.MPCSolver(qp; linear_solver=linear_solver, options...)
    t_solve = @elapsed stats = MadIPM.solve!(solver)
    return solver, stats, t_init, t_solve
end

function timed_gpu_solve(gpu_bnlp; options...)
    t_init = @elapsed begin
        solver = MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
            cudss_algorithm = MadNLP.LDL,
            options...
        )
        CUDA.synchronize()
    end
    t_solve = @elapsed begin
        stats = MadIPM.solve!(solver)
        CUDA.synchronize()
    end
    return solver, stats, t_init, t_solve
end

# Sequential CPU baseline: solve the instances one after the other, each with
# its own solver, keeping per-instance wall times. Summing over the first b
# instances then gives the sequential cost of exactly the problems a batch of
# size b holds.
function timed_cpu_sequential(qps; linear_solver, options...)
    n = length(qps)
    stats = Vector{Any}(undef, n)
    t_init = zeros(n)
    t_solve = zeros(n)
    for (i, qp) in enumerate(qps)
        _, stats[i], t_init[i], t_solve[i] =
            timed_cpu_solve(qp; linear_solver=linear_solver, options...)
    end
    return stats, t_init, t_solve
end

# Per-batch summaries. GPU: converged count, mean iterations, init time, solve
# time of the batch solved at once. CPU, over instances 1:b: converged count,
# mean iterations, summed init and solve times (one CPU solving them one after
# the other), then the max init and solve times (b CPUs solving one instance
# each with perfect scaling, i.e. the wall time is set by the slowest instance).
function cpu_summary(stats, t_init, t_solve, b)
    converged = count(s -> s.status == MadNLP.SOLVE_SUCCEEDED, stats[1:b])
    return (
        converged, sum(s.iter for s in stats[1:b]) / b,
        sum(t_init[1:b]), sum(t_solve[1:b]),
        maximum(t_init[1:b]), maximum(t_solve[1:b]),
    )
end

function gpu_summary(stats, t_init, t_solve)
    converged = count(isequal(MadNLP.SOLVE_SUCCEEDED), stats.status)
    return (converged, sum(stats.iter) / length(stats.iter), t_init, t_solve)
end

# Compile both code paths on the instance layout the benchmarks use: the CPU
# baseline solves the same presolved, scaled, standard-form problem as the
# batch, with the linear solver the benchmark will use.
function _warmup(qp; linear_solver=Ma57Solver)
    qps = build_qps(qp, 2)
    warmup_options = (
        print_level=MadNLP.ERROR,
        max_iter=1,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
    )
    timed_cpu_solve(qps[1]; linear_solver=linear_solver, warmup_options...)
    gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps))
    timed_gpu_solve(gpu_bnlp; warmup_options...)
    return
end

