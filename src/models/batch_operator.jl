abstract type BatchSparseOperator end

struct HostBatchSparseOperator{MT,VI<:AbstractVector{Int}} <: BatchSparseOperator
    nzvals::MT
    rows::VI
    cols::VI
    rowptr::VI
    nz_idx::VI
    val_idx::VI
end

_sparse_structure(A::BatchSparseOperator) = (A.rows, A.cols)
_sparse_values(A::BatchSparseOperator) = A.nzvals
_copy_sparse_structure!(
    A::BatchSparseOperator,
    rows::AbstractVector,
    cols::AbstractVector,
) = (copyto!(rows, A.rows); copyto!(cols, A.cols); (rows, cols))

"""
    sync_batch_operator!(op)

Refresh the operator's internal value storage after the source values it was
built over have been mutated in place. No-op on CPU, where the operator reads
the shared storage through an index indirection; the CUDA operator gathers the
values into its block-diagonal CSR buffer.
"""
sync_batch_operator!(op::BatchSparseOperator) = op
sync_batch_operator!(::Nothing) = nothing

function _coo_to_csr(indices::AbstractVector{Int}, n::Int)
    nnz = length(indices)
    rowptr = zeros(Int, n + 1)
    for i = 1:nnz
        rowptr[indices[i]+1] += 1
    end
    rowptr[1] = 1
    for r = 1:n
        rowptr[r+1] += rowptr[r]
    end
    colidx = Vector{Int}(undef, nnz)
    pos = copy(rowptr[1:n])
    for i = 1:nnz
        r = indices[i]
        colidx[pos[r]] = i
        pos[r] += 1
    end
    return rowptr, colidx
end

function _symmetric_scatter_ops(
    rows::AbstractVector{Int},
    cols::AbstractVector{Int},
    nnz::Int,
)
    off_diag = findall(rows .!= cols)
    scatter_rows = vcat(rows, cols[off_diag])
    nz_idx = vcat(collect(1:nnz), off_diag)
    gather_cols = vcat(cols, rows[off_diag])
    return scatter_rows, nz_idx, gather_cols
end

_build_op(nzvals::Matrix, rows, cols, rowptr, nz_map, val_map, colidx) =
    HostBatchSparseOperator(nzvals, rows, cols, rowptr, nz_map[colidx], val_map[colidx])

function _jacobian_op(qp_ref, nzvals)
    rows, cols = _sparse_structure(qp_ref.data.A)
    rowptr, colidx = _coo_to_csr(rows, qp_ref.meta.ncon)
    return _build_op(nzvals, rows, cols, rowptr, collect(1:qp_ref.meta.nnzj), cols, colidx)
end

function _hessian_op(qp_ref, nzvals)
    rows, cols = _sparse_structure(qp_ref.data.Q)
    sym_rows, sym_nz, sym_cols = _symmetric_scatter_ops(rows, cols, qp_ref.meta.nnzh)
    rowptr, colidx = _coo_to_csr(sym_rows, qp_ref.meta.nvar)
    return _build_op(nzvals, rows, cols, rowptr, sym_nz, sym_cols, colidx)
end

batch_spmv!(
    out::AbstractMatrix{T},
    op::BatchSparseOperator,
    B::AbstractMatrix,
    alpha::T = one(T),
    beta::T = zero(T);
    val_offset::Int = 0,
) where {T} = _batch_spmv_impl!(out, op, B, alpha, beta, val_offset)

LinearAlgebra.mul!(
    Y::AbstractMatrix{T},
    op::BatchSparseOperator,
    X::AbstractMatrix{T},
    α::Number,
    β::Number,
) where {T} = batch_spmv!(Y, op, X, T(α), T(β))
LinearAlgebra.mul!(
    Y::AbstractMatrix{T},
    op::BatchSparseOperator,
    X::AbstractMatrix{T},
) where {T} = batch_spmv!(Y, op, X)

function _batch_spmv_impl!(
    out::AbstractMatrix{T},
    op::HostBatchSparseOperator,
    B::AbstractMatrix,
    alpha::T,
    beta::T,
    val_offset::Int,
) where {T}
    nout = length(op.rowptr) - 1
    beta_zero = iszero(beta)
    @inbounds for r = 1:nout, j = 1:size(out, 2)
        acc = zero(T)
        for k = op.rowptr[r]:(op.rowptr[r+1]-1)
            acc += op.nzvals[op.nz_idx[k], j] * B[op.val_idx[k]+val_offset, j]
        end
        out[r, j] = beta_zero ? alpha * acc : alpha * acc + beta * out[r, j]
    end
    return out
end

batch_mapreduce!(f, op, neutral, out::AbstractMatrix, srcs::AbstractMatrix...) =
    (out .= mapreduce(f, op, srcs...; dims = 1, init = neutral))

batch_maximum!(out::AbstractMatrix{T}, src::AbstractMatrix{T}) where {T} =
    batch_mapreduce!(identity, max, typemin(T), out, src)
