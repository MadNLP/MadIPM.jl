module MadIPMCUDAExt

using LinearAlgebra
using SparseArrays
using NLPModels
using QuadraticModels
using CUDA
using CUDA.CUSPARSE
using KernelAbstractions
import QuadraticModels: SparseMatrixCOO
import MadIPM

include("cuda_wrapper.jl")
include("operators.jl")

@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj, Ax)
    i = @index(Global, Linear)
    for c in Ap[i]:Ap[i+1]-1
        rows[c] = i
        cols[c] = Aj[c]
    end
end

function fill_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows, cols)
    @assert length(cols) == length(rows)
    if length(cols) > 0
        backend = CUDABackend()
        _fill_sparse_structure!(backend)(
            rows, cols,
            A.rowPtr, A.colVal, A.nzVal; ndrange=size(A, 1),
        )
    end
end

function NLPModels.obj(qp::QuadraticModel{T, S, M1}, x::AbstractVector) where {T, S, M1 <: MadIPMOperator}
  NLPModels.increment!(qp, :neval_obj)
  mul!(qp.data.v, qp.data.H, x)
  return qp.data.c0 + dot(qp.data.c, x) + dot(qp.data.v, x) / 2
end

function NLPModels.grad!(qp::QuadraticModel{T, S, M1}, x::AbstractVector, g::AbstractVector) where {T, S, M1 <: MadIPMOperator}
  NLPModels.increment!(qp, :neval_grad)
  mul!(g, qp.data.H, x)
  g .+= qp.data.c
  return g
end

function NLPModels.hess_structure!(
    qp::QuadraticModel{T, S, M1},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, S, M1 <: MadIPMOperator{T, <: CUSPARSE.CuSparseMatrixCSR}}
    fill_structure!(qp.data.H.A, rows, cols)
    return rows, cols
end

function NLPModels.hess_coord!(
    qp::QuadraticModel{T, S, M1},
    x::AbstractVector{T},
    vals::AbstractVector{T};
    obj_weight::Real = one(eltype(x)),
) where {T, S, M1 <: MadIPMOperator{T, <: CUSPARSE.CuSparseMatrixCSR}}
    NLPModels.increment!(qp, :neval_hess)
    vals .= obj_weight .* qp.data.H.A.nzVal
    return vals
end

function NLPModels.jac_lin_coord!(
    qp::QuadraticModel{T, S, M1, M2},
    x::AbstractVector,
    vals::AbstractVector,
) where {T, S, M1, M2 <: MadIPMOperator{T, <: CUSPARSE.CuSparseMatrixCSR}}
    @lencheck qp.meta.nvar x
    @lencheck qp.meta.lin_nnzj vals
    NLPModels.increment!(qp, :neval_jac_lin)
    vals .= qp.data.A.A.nzVal
    return vals
end

function NLPModels.jac_lin_structure!(
    qp::QuadraticModel{T, S, M1, M2},
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: MadIPMOperator{T, <: CUSPARSE.CuSparseMatrixCSR}}
    @lencheck qp.meta.lin_nnzj rows cols
    fill_structure!(qp.data.A.A, rows, cols)
    return rows, cols
end

#=
    CuSparseMatrixCOO
=#

function CUSPARSE.CuSparseMatrixCOO(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    return CUSPARSE.CuSparseMatrixCOO{Tv, Ti}(
        CuVector(A.rows),
        CuVector(A.cols),
        CuVector(A.vals),
        size(A),
        nnz(A),
    )
end

#=
    CuSparseMatrixCSR
=#

function CUSPARSE.CuSparseMatrixCSR(A::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    m, n = size(A)
    Ap, Ai, Ax = MadIPM.coo_to_csr(m, n, A.rows, A.cols, A.vals)
    return CUSPARSE.CuSparseMatrixCSR{Tv, Ti}(
        CuVector(Ap),
        CuVector(Ai),
        CuVector(Ax),
        size(A),
    )
end

#=
    Pass QuadraticModel to the GPU
=#

function Base.convert(::Type{QuadraticModel{T, S}}, qp::QuadraticModel{T}) where {T, S<:CuArray}
    H = MadIPMOperator(CuSparseMatrixCSR(qp.data.H), symmetric=true)
    A = MadIPMOperator(CuSparseMatrixCSR(qp.data.A), symmetric=false)

    return QuadraticModel(
        S(qp.data.c),
        H;
        A=A,
        lcon=S(qp.meta.lcon),
        ucon=S(qp.meta.ucon),
        lvar=S(qp.meta.lvar),
        uvar=S(qp.meta.uvar),
        c0=qp.data.c0,
        x0=S(qp.meta.x0),
    )
end

end

