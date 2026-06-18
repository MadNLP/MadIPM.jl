mutable struct MadIPMOperator{T,M,M2} <: AbstractMatrix{T}
    type::Type{T}
    m::Int
    n::Int
    A::M
    mat::M2
    transa::Char
    descA::cuSPARSE.CuSparseMatrixDescriptor
    buffer::CuVector{UInt8}
    spmm_buffer::CuVector{UInt8}
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
        function MadIPMOperator(A::$SparseMatrixType; transa::Char='N', symmetric::Bool=false, spmm_ncols::Int=0) where T <: $BlasType
            m, n = size(A)
            op_in = transa == 'N' ? n : m
            op_out = transa == 'N' ? m : n
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            bool = symmetric && (nnz(A) > 0)
            mat = bool ? tril(A, -1) + A' : A
            descA = cuSPARSE.CuSparseMatrixDescriptor(mat, 'O')
            descX = cuSPARSE.CuDenseVectorDescriptor(T, op_in)
            descY = cuSPARSE.CuDenseVectorDescriptor(T, op_out)
            algo = cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
            buffer_size = Ref{Csize_t}()
            cuSPARSE.cusparseSpMV_bufferSize(cuSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer_size)
            buffer = CuVector{UInt8}(undef, buffer_size[])
            if cuSPARSE.version() ≥ v"12.3"
                cuSPARSE.cusparseSpMV_preprocess(cuSPARSE.handle(), transa, alpha, descA, descX, beta, descY, T, algo, buffer)
            end
            M = typeof(A)
            M2 = typeof(mat)
            alpha = Ref{T}(one(T))
            beta = Ref{T}(zero(T))
            spmm_buffer = if spmm_ncols > 0
                descB = cuSPARSE.CuDenseMatrixDescriptor(T, n, spmm_ncols)
                descC = cuSPARSE.CuDenseMatrixDescriptor(T, m, spmm_ncols)
                spmm_buf_size = Ref{Csize_t}()
                spmm_algo = cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT
                cuSPARSE.cusparseSpMM_bufferSize(cuSPARSE.handle(), transa, 'N', alpha, descA, descB, beta, descC, T, spmm_algo, spmm_buf_size)
                buf = CuVector{UInt8}(undef, spmm_buf_size[])
                if cuSPARSE.version() ≥ v"12.3"
                    cuSPARSE.cusparseSpMM_preprocess(cuSPARSE.handle(), transa, 'N', alpha, descA, descB, beta, descC, T, spmm_algo, buf)
                end
                buf
            else
                CuVector{UInt8}(undef, 0)
            end
            return MadIPMOperator{T,M,M2}(T, op_out, op_in, A, mat, transa, descA, buffer, spmm_buffer, alpha, beta)
        end
    end
end

function LinearAlgebra.mul!(Y::CuMatrix{T}, A::MadIPMOperator{T}, X::CuMatrix{T}) where T <: BlasFloat
    (size(Y, 1) != A.m) && throw(DimensionMismatch("size(Y,1) != A.m"))
    (size(X, 1) != A.n) && throw(DimensionMismatch("size(X,1) != A.n"))
    descX = cuSPARSE.CuDenseMatrixDescriptor(X)
    descY = cuSPARSE.CuDenseMatrixDescriptor(Y)
    cuSPARSE.cusparseSpMM(
        cuSPARSE.handle(), A.transa, 'N',
        A.alpha, A.descA, descX, A.beta, descY,
        T, cuSPARSE.CUSPARSE_SPMM_ALG_DEFAULT, A.spmm_buffer,
    )
end

function LinearAlgebra.mul!(y::CuVector{T}, A::MadIPMOperator{T}, x::CuVector{T}) where T <: BlasFloat
    (length(y) != A.m) && throw(DimensionMismatch("length(y) != A.m"))
    (length(x) != A.n) && throw(DimensionMismatch("length(x) != A.n"))
    descY = cuSPARSE.CuDenseVectorDescriptor(y)
    descX = cuSPARSE.CuDenseVectorDescriptor(x)
    algo = cuSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    cuSPARSE.cusparseSpMV(cuSPARSE.handle(), A.transa, A.alpha, A.descA, descX, A.beta, descY, T, algo, A.buffer)
end

function LinearAlgebra.mul!(Y::CuMatrix{T}, A::MadIPMOperator{T}, X::CuMatrix{T}, α::Number, β::Number) where T <: BlasFloat
    A.alpha[] = T(α)
    A.beta[] = T(β)
    mul!(Y, A, X)
    A.alpha[] = one(T)
    A.beta[] = zero(T)
    return Y
end
