function _stack_columns(MT, src, getter = identity)
    @assert !isempty(src) "Need at least one column"
    out = MT(undef, length(getter(first(src))), length(src))
    for (j, item) in enumerate(src)
        copyto!(view(out, :, j), getter(item))
    end
    return out
end

function _all_equal(qps, getter)
    length(qps) < 2 && return true
    ref = getter(qps[1])
    for j = 2:length(qps)
        x = getter(qps[j])
        x === ref && continue
        x == ref || return false
    end
    return true
end

_copy_sparse_structure!(A::SparseMatrixCOO, rows::AbstractVector, cols::AbstractVector) =
    (rows .= A.rows; cols .= A.cols; (rows, cols))
function _copy_sparse_structure!(
    A::SparseMatrixCSC,
    rows::AbstractVector,
    cols::AbstractVector,
)
    @inbounds for j in axes(A, 2), k = A.colptr[j]:(A.colptr[j+1]-1)
        rows[k] = A.rowval[k];
        cols[k] = j
    end
    return rows, cols
end

_copy_sparse_values!(A::SparseMatrixCOO, vals::AbstractVector) = (vals .= A.vals; vals)
_copy_sparse_values!(A::SparseMatrixCSC, vals::AbstractVector) =
    (copyto!(vals, nonzeros(A)); vals)
_copy_sparse_values!(vals::AbstractMatrix, src) = (vals .= _sparse_values(src); vals)

function _sparse_structure(A::Union{SparseMatrixCOO,SparseMatrixCSC})
    rows = Vector{Int}(undef, nnz(A));
    cols = similar(rows)
    return _copy_sparse_structure!(A, rows, cols)
end

_sparse_values(A::SparseMatrixCOO) = A.vals
_sparse_values(A::SparseMatrixCSC) = nonzeros(A)
