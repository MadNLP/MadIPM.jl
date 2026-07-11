function Adapt.adapt_structure(::Type{<:CuArray}, bm::BatchModel{T, <:Matrix}) where {T}
  nbatch = bm.meta.nbatch
  return BatchModel(
    _adapt_batch_meta(CuArray, bm.meta),
    Adapt.adapt(CuArray, bm.c_batch),
    Adapt.adapt(CuArray, bm.c0_batch),
    _adapt_op_cuda(bm.A, nbatch; symmetric = false),
    _adapt_op_cuda(bm.Q, nbatch; symmetric = true),
    Adapt.adapt(CuArray, bm._HX),
  )
end

_adapt_op_cuda(::Nothing, nbatch; symmetric) = nothing

function _adapt_op_cuda(op::AbstractSparseOperator, nbatch; symmetric)
  scalar = operator_sparse_matrix(op)
  if symmetric
    source = _to_cu_csr(scalar)
    full   = _to_cu_csr(_expand_symmetric_matrix(scalar))
    return _cu_sparse_operator(source, full; spmm_ncols = nbatch)
  else
    csr = _to_cu_csr(scalar)
    return _cu_sparse_operator(csr, csr; spmm_ncols = nbatch)
  end
end

_adapt_op_cuda(op::BatchSparseOperator, nbatch; symmetric) = Adapt.adapt_structure(CuArray, op)
