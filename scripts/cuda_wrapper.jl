using CUDA
using CUDA.CUSPARSE
using MadNLPGPU
using KernelAbstractions
using SparseArrays
using LinearAlgebra
import LinearAlgebra: BlasFloat

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        # TODO: do we need Atomix?
        dest[to_map[k]] += src[k]
    end
end

function MadNLP.transfer!(
    dest::CUSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    fill!(nonzeros(dest), zero(Tv))
    if length(map) > 0
        _transfer_to_map!(CUDABackend())(nonzeros(dest), map, src.V; ndrange=length(map))
        KernelAbstractions.synchronize(CUDABackend())
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

mutable struct MadIPMOperator{T,M,M2} <: AbstractMatrix{T}
    type::Type{T}
    m::Int
    n::Int
    A::M
    mat::M2
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
end

Base.eltype(A::MadIPMOperator{T}) where T = T
Base.size(A::MadIPMOperator) = (A.m, A.n)
SparseArrays.nnz(A::MadIPMOperator) = nnz(A.A)

for (SparseMatrixType, BlasType) in ((:(CuSparseMatrixCSR{T}), :BlasFloat),
                                     (:(CuSparseMatrixCSC{T}), :BlasFloat),
                                     (:(CuSparseMatrixCOO{T}), :BlasFloat))
    @eval begin
        function MadIPMOperator(A::$SparseMatrixType; transa::Char='N', symmetric::Bool=false) where T <: $BlasType
            m, n = size(A)
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            bool = symmetric && (nnz(A) > 0)
            mat = bool ? tril(A, -1) + A' : A
            descA = CUSPARSE.CuSparseMatrixDescriptor(mat, 'O')
            descX = CUSPARSE.CuDenseVectorDescriptor(T, n)
            descY = CUSPARSE.CuDenseVectorDescriptor(T, m)
            algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
            buffer_size = Ref{Csize_t}()
            CUSPARSE.cusparseSpMV_bufferSize(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
            buffer = CuVector{UInt8}(undef, buffer_size[])
            if CUSPARSE.version() ≥ v"12.3"
                CUSPARSE.cusparseSpMV_preprocess(CUSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer)
            end
            M = typeof(A)
            M2 = typeof(mat)
            return MadIPMOperator{T,M,M2}(T, m, n, A, mat, transa, descA, buffer)
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::MadIPMOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    alpha = Ref{T}(one(T))
    beta = Ref{T}(zero(T))
    CUSPARSE.cusparseSpMV(CUSPARSE.handle(), A.transa, alpha, A.descA, descX, beta, descY, T, algo, A.buffer)
end
