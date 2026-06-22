
using MadIPM, MadNLP
using MadNLPHSL
using QuadraticModels, NLPModels
using BatchQuadraticModels: ObjRHSBatchQuadraticModel
using MadNLPGPU, CUDA, CUDSS, KernelAbstractions
using Random, Distributions, SparseArrays, Memoize
using Base.Threads, Polyester
using Printf
using SparseMatricesCOO
using QPSReader
using HSL

CUDA.device!(1)

function refresh_memory()
    CUDA.reclaim()
    GC.gc(true)
    CUDA.reclaim()
    return
end


const MadIPMCUDAExt = Base.get_extension(MadIPM, :MadIPMCUDAExt)
function SparseMatricesCOO.SparseMatrixCOO(A::MadIPMCUDAExt.MadIPMOperator)
    return SparseMatricesCOO.SparseMatrixCOO(A.A)
end

function _scale_coo!(A, Dr, Dc)
    k = 1
    for (i, j) in zip(A.rows, A.cols)
        A.vals[k] = A.vals[k] / (Dr[i] * Dc[j])
        k += 1
    end
end

function scale_qp(qp::QuadraticModel)
    A = qp.data.A
    m, n = size(A)

    if !LIBHSL_isfunctional()
        return qp
    end

    A_csc = sparse(A.rows, A.cols, A.vals, m, n)
    Dr, Dc = HSL.mc77(A_csc, 0)

    Hs = copy(qp.data.H)
    As = copy(qp.data.A)
    _scale_coo!(Hs, Dc, Dc)
    _scale_coo!(As, Dr, Dc)

    data = QuadraticModels.QPData(
        qp.data.c0,
        qp.data.c ./ Dc,
        # qp.data.v,
        Hs,
        As,
    )

    return QuadraticModel(
        NLPModelMeta(
            qp.meta.nvar;
            ncon=qp.meta.ncon,
            lvar=qp.meta.lvar .* Dc,
            uvar=qp.meta.uvar .* Dc,
            lcon=qp.meta.lcon ./ Dr,
            ucon=qp.meta.ucon ./ Dr,
            x0=qp.meta.x0 .* Dc,
            y0=qp.meta.y0 ./ Dr,
            nnzj=qp.meta.nnzj,
            lin_nnzj=qp.meta.nnzj,
            lin=qp.meta.lin,
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        Counters(),
        data,
    )
end

function build_qps(base_qp, batch_size; T = Float64, shift_c=true, shift_b=false, shift_A=false)
    if T != eltype(base_qp.data.c)
        base_qp = convert(QuadraticModel{T, Vector{T}}, base_qp)
    end

    base_pqp, flag = MadIPM.presolve_qp(base_qp)
    @assert flag "Presolve failed for $case"

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

        QuadraticModel(
            c_new,
            base_sqp.data.H;
            A = base_sqp.data.A,
            lcon = lcon_new,
            ucon = ucon_new,
            lvar = copy(base_sqp.meta.lvar),
            uvar = copy(base_sqp.meta.uvar),
            x0 = copy(base_sqp.meta.x0),
            c0 = base_sqp.data.c0,
        )
    end for i in 1:batch_size]
end

