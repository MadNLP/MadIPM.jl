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
        backend = CUDABackend()
        _transfer_to_map!(backend)(nonzeros(dest), map, src.V; ndrange=length(map))
        KernelAbstractions.synchronize(backend)
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:CUSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzVal, 0.0)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
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

function MadIPM.coo_to_csr(
    n_rows,
    n_cols,
    Ai::CuVector{Ti},
    Aj::CuVector{Ti},
    Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt=:csr)
    return (B.rowPtr, B.colVal, B.nzVal)
end

# Should be backported in MadNLPGPU.jl in the future
@kernel function _scale_augmented_system_coo_kernel!(dest_V, @Const(src_I), @Const(src_J), @Const(src_V), @Const(scaling), @Const(n), @Const(m))
    k = @index(Global, Linear)
    i = src_I[k]
    j = src_J[k]

    # Primal regularization pr_diag
    if k <= n
        dest_V[k] = src_V[k]
    # Hessian block
    elseif i <= n && j <= n
        dest_V[k] = src_V[k] * scaling[i] * scaling[j]
    # Jacobian block
    elseif n + 1 <= i <= n + m && j <= n
        dest_V[k] = src_V[k] * scaling[j]
    # Dual regularization du_diag
    elseif (n + 1 <= i <= n + m) && (n + 1 <= j <= n + m)
        dest_V[k] = src_V[k]
    end
    nothing
end

function MadNLP._build_scale_augmented_system_coo!(dest, src, scaling::CuArray, n, m)
    backend = CUDABackend()
    kernel! = _scale_augmented_system_coo_kernel!(backend)
    N = nnz(src)
    kernel!(dest.V, src.I, src.J, src.V, scaling, n, m; ndrange = N)
    KernelAbstractions.synchronize(backend)
end

@kernel function assemble_normal_system_kernel!(@Const(n_rows), @Const(n_cols), @Const(Jtp), @Const(Jtj), @Const(Jtx),
                                                @Const(Cp), @Const(Cj), Cx, @Const(Dx), @Const(Tv))
    i = @index(Global, Linear)

    for c in Cp[i]:Cp[i+1]-1
        j = Cj[c]
        acc = zero(Tv)

        p1 = Jtp[i]
        p2 = Jtp[j]
        p1_end = Jtp[i+1] - 1
        p2_end = Jtp[j+1] - 1

        while p1 <= p1_end && p2 <= p2_end
            k1 = Jtj[p1]
            k2 = Jtj[p2]

            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1
                p2 += 1
            elseif k1 < k2
                p1 += 1
            else
                p2 += 1
            end
        end

        Cx[c] = acc
    end
    nothing
end

function MadIPM.assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::CuArray{Ti},
    Jtj::CuArray{Ti},
    Jtx::CuArray{Tv},
    Cp::CuArray{Ti},
    Cj::CuArray{Ti},
    Cx::CuArray{Tv},
    Dx::CuArray{Tv},
) where {Ti, Tv}
    backend = CUDABackend()
    kernel! = assemble_normal_system_kernel!(backend)
    kernel!(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows)
    KernelAbstractions.synchronize(backend)
end

@kernel function count_normal_nnz!(Cp, @Const(Jtp), @Const(Jtj), @Const(n_rows), @Const(n_cols))
    i = @index(Global, Linear)

    # thread-local binary buffer
    xb = @localmem UInt8 n_cols
    for k = 1:n_cols
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    count = 0
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                count += 1
                break
            end
        end
    end

    Cp[i+1] = count
    nothing
end

@kernel function fill_normal_indices!(Cj, @Const(Cp), @Const(Jtp), @Const(Jtj), @Const(n_rows), @Const(n_cols))
    i = @index(Global, Linear)

    xb = @localmem UInt8 n_cols
    for k = 1:n_cols
        xb[k] = 0
    end

    for c = Jtp[i]:Jtp[i+1]-1
        j = Jtj[c]
        xb[j] = 1
    end

    pos = Cp[i]
    for j = i:n_rows
        for c = Jtp[j]:Jtp[j+1]-1
            k = Jtj[c]
            if xb[k] == 1
                Cj[pos] = j
                pos += 1
                break
            end
        end
    end
    nothing
end

function MadIPM.build_normal_system(
    n_rows,
    n_cols,
    Jtp::CuVector{Ti},
    Jtj::CuVector{Ti},
) where {Ti}
    backend = CUDABackend()
    Cp = CUDA.ones(Ti, n_rows + 1)
    kernel1! = count_normal_nnz!(backend)
    kernel1!(Cp, Jtp, Jtj, n_rows, n_cols; ndrange = n_rows)
    KernelAbstractions.synchronize(backend)

    Cp = cumsum(Cp)
    nnz_JtJ = CUDA.@allowscalar (Cp[end] - 1)
    Cj = CUDA.zeros(Ti, nnz_JtJ)

    kernel2! = fill_normal_indices!(backend)
    kernel2!(Cj, Cp, Jtp, Jtj, n_rows, n_cols; ndrange = n_rows)
    KernelAbstractions.synchronize(backend)
    return (Cp, Cj)
end

MadIPM.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC
MadIPM._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPM._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPM._nzval(A::CuSparseMatrixCSC) = A.nzVal

