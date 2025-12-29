mutable struct MadIPMOperator{T,M,M2} <: AbstractMatrix{T}
    type::Type{T}
    m::Int
    n::Int
    A::M
    mat::M2
    transa::Char
    descA::CUSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
    alpha::Base.RefValue{T}
    beta::Base.RefValue{T}
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
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            return MadIPMOperator{T,M,M2}(T, m, n, A, mat, transa, descA, buffer, alpha, beta)
        end
    end
end

function LinearAlgebra.mul!(y::CuVector{T}, A::MadIPMOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    descY = CUSPARSE.CuDenseVectorDescriptor(y)
    descX = CUSPARSE.CuDenseVectorDescriptor(x)
    algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    CUSPARSE.cusparseSpMV(CUSPARSE.handle(), A.transa, A.alpha, A.descA, descX, A.beta, descY, T, algo, A.buffer)
end
