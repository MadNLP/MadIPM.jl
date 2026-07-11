@kernel function _fill_sparse_structure!(rows, cols, Ap, Aj)
  i = @index(Global, Linear)
  for c in Ap[i]:Ap[i + 1] - 1
    rows[c] = i; cols[c] = Aj[c]
  end
end

function _copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCSR, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows)
  length(cols) > 0 && _fill_sparse_structure!(CUDABackend())(rows, cols, A.rowPtr, A.colVal; ndrange = size(A, 1))
  return rows, cols
end

function _copy_sparse_structure!(A::CUSPARSE.CuSparseMatrixCOO, rows::CuVector, cols::CuVector)
  @assert length(cols) == length(rows) == nnz(A)
  copyto!(rows, A.rowInd); copyto!(cols, A.colInd)
  return rows, cols
end

function _copy_sparse_values!(A::Union{CUSPARSE.CuSparseMatrixCSR, CUSPARSE.CuSparseMatrixCSC, CUSPARSE.CuSparseMatrixCOO},
                              vals::CuVector)
  @assert length(vals) == nnz(A)
  copyto!(vals, A.nzVal)
  return vals
end

_copy_sparse_structure!(A::CuSparseOperator, rows::CuVector, cols::CuVector) =
  _copy_sparse_structure!(operator_sparse_matrix(A), rows, cols)
_copy_sparse_values!(A::CuSparseOperator, vals::CuVector) =
  _copy_sparse_values!(operator_sparse_matrix(A), vals)

_sparse_values(A::_CuSparseMatrix) = A.nzVal
_sparse_values(A::CuSparseOperator) = _sparse_values(operator_sparse_matrix(A))

_mul_jt!(jtv, A::CuSparseOperator{T}, v) where {T} = mul!(jtv, transpose(A), v)

function _adapt_data_op(_to, op::SparseOperator; symmetric::Bool)
  scalar = operator_sparse_matrix(op)
  if symmetric
    source = _to_cu_csr(scalar)
    full   = _to_cu_csr(_expand_symmetric_matrix(scalar))
    return _cu_sparse_operator(source, full)
  else
    csr = _to_cu_csr(scalar)
    return _cu_sparse_operator(csr, csr)
  end
end

_adapt_data_q(to, ::Nothing) = nothing
_adapt_data_q(to, Q::SparseOperator) = _adapt_data_op(to, Q; symmetric = true)

function Adapt.adapt_structure(to, d::ModelData{T, VT, W, MQ, <:SparseOperator}) where {T, VT, W, MQ}
  A = _adapt_data_op(to, d.A; symmetric = false)
  Q = _adapt_data_q(to, d.Q)
  c = Adapt.adapt(to, d.c)
  v = Adapt.adapt(to, d._v)
  # `T` is not inferable from the fields, so pass the parameters explicitly.
  return ModelData{T, typeof(c), typeof(v), typeof(Q), typeof(A)}(
    A, Q,
    Adapt.adapt(to, d.lcon), Adapt.adapt(to, d.ucon),
    Adapt.adapt(to, d.lvar), Adapt.adapt(to, d.uvar),
    c, Adapt.adapt(to, d.c0), v,
  )
end

Adapt.adapt_structure(to, m::ScalarModel{T, VT, W, MQ, <:SparseOperator}) where {T, VT, W, MQ} =
  ScalarModel(Adapt.adapt(to, m.data);
    x0 = Adapt.adapt(to, m.meta.x0), y0 = Adapt.adapt(to, m.meta.y0),
    minimize = m.meta.minimize, name = m.meta.name)
