
using DelimitedFiles
using Printf
using MadIPM, MadNLP
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

function _warmup(qp)
    # Warmup CPU
    cpu_solver = MadIPM.MPCSolver(
        qp;
        print_level=MadNLP.ERROR,
        max_iter=1,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
        linear_solver=Ma57Solver,
    )
    MadIPM.solve!(cpu_solver)

    # Warmup GPU
    qps = build_qps(qp, 2)
    cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
    gpu_bnlp = to_gpu(cpu_bnlp)

    gpu_solver = MadIPM.UniformBatchMPCSolver(
        gpu_bnlp;
        print_level=MadNLP.ERROR,
        max_iter=1,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
        uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.LDL,
    )
    stats = MadIPM.solve!(gpu_solver)
    return
end

