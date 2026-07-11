mutable struct CuBatchSpMVLayout{T, Ti}
  A::CUSPARSE.CuSparseMatrixCSR{T, Ti}
  descA::CUSPARSE.CuSparseMatrixDescriptor
  descX::CUSPARSE.CuDenseVectorDescriptor
  xlen::Int
  buffer::CuVector{UInt8}
end

mutable struct CuBatchSparseOperator{T, Ti, MT <: CuMatrix{T}, VI} <: BatchSparseOperator
  nzvals::MT                     # source values (nsrc × nbatch); may alias external storage
  rows::VI                       # COO structure of one instance (structure queries)
  cols::VI
  m::Int                         # output rows per instance
  nbatch::Int
  gather_idx::CuVector{Int32}    # CSR slot k reads nzvals[gather_idx[k], j]
  colval_base::Vector{Int}       # per-instance CSR column indices (host, 1-based)
  rowPtr::CuVector{Ti}           # block-diagonal CSR row pointers (m * nbatch + 1)
  csr_nzval::CuMatrix{T}         # (nnz, nbatch) CSR-ordered values, contiguous per instance
  out_buf::CuVector{T}           # contiguous SpMV output (m * nbatch)
  descY::CUSPARSE.CuDenseVectorDescriptor
  layouts::Dict{Tuple{Int, Int}, CuBatchSpMVLayout{T, Ti}}
end

SparseArrays.nnz(op::CuBatchSparseOperator) = length(op.rows)

function CuBatchSparseOperator(nzvals::CuMatrix{T}, rows, cols,
                               rowptr::Vector{Int}, nz_idx::Vector{Int},
                               val_idx::Vector{Int}) where {T}
  m = length(rowptr) - 1
  nbatch = size(nzvals, 2)
  nnz_csr = length(nz_idx)

  nz_idx = copy(nz_idx); val_idx = copy(val_idx)
  @inbounds for r in 1:m
    lo, hi = rowptr[r], rowptr[r + 1] - 1
    if hi > lo
      slice = lo:hi
      p = sortperm(view(val_idx, slice))
      val_idx[slice] = val_idx[slice][p]
      nz_idx[slice]  = nz_idx[slice][p]
    end
  end

  Ti = max(m * nbatch + 1, nnz_csr * nbatch + 1) <= typemax(Int32) ? Int32 : Int64

  rowPtr = Vector{Ti}(undef, m * nbatch + 1)
  rowPtr[1] = Ti(1)
  @inbounds for j in 0:nbatch-1, r in 1:m
    rowPtr[j * m + r + 1] = Ti(rowptr[r + 1] + j * nnz_csr)
  end

  out_buf = CUDA.zeros(T, m * nbatch)
  op = CuBatchSparseOperator{T, Ti, typeof(nzvals), typeof(rows)}(
    nzvals, rows, cols, m, nbatch,
    CuVector{Int32}(Int32.(nz_idx)),
    val_idx,
    CuVector{Ti}(rowPtr),
    CuMatrix{T}(undef, nnz_csr, nbatch),
    out_buf,
    CUSPARSE.CuDenseVectorDescriptor(out_buf),
    Dict{Tuple{Int, Int}, CuBatchSpMVLayout{T, Ti}}(),
  )
  sync_batch_operator!(op)
  return op
end

_build_op(nzvals::CuMatrix, rows, cols, rowptr, nz_map, val_map, colidx) =
  CuBatchSparseOperator(nzvals, rows, cols,
                        Vector{Int}(rowptr),
                        Vector{Int}(nz_map[colidx]),
                        Vector{Int}(val_map[colidx]))

function Adapt.adapt_structure(::Type{<:CuArray}, op::HostBatchSparseOperator)
  return CuBatchSparseOperator(
    Adapt.adapt(CuArray, op.nzvals),
    Adapt.adapt(CuArray, op.rows),
    Adapt.adapt(CuArray, op.cols),
    Vector{Int}(op.rowptr), Vector{Int}(op.nz_idx), Vector{Int}(op.val_idx))
end

@kernel function _gather_csr_values_kernel!(dst, @Const(src), @Const(idx))
  k, j = @index(Global, NTuple)
  @inbounds dst[k, j] = src[idx[k], j]
end

function sync_batch_operator!(op::CuBatchSparseOperator)
  nnz_csr = size(op.csr_nzval, 1)
  nnz_csr == 0 && return op
  _gather_csr_values_kernel!(CUDABackend())(
    op.csr_nzval, op.nzvals, op.gather_idx; ndrange = (nnz_csr, op.nbatch))
  return op
end

function _spmv_layout!(op::CuBatchSparseOperator{T, Ti}, ld::Int, offset::Int) where {T, Ti}
  return get!(op.layouts, (ld, offset)) do
    nnz_csr = size(op.csr_nzval, 1)
    nb = op.nbatch
    ld * nb + 1 <= typemax(Ti) || throw(ArgumentError(
      "block-diagonal column extent $(ld * nb) overflows $(Ti); reduce the batch size"))
    colVal = Vector{Ti}(undef, nnz_csr * nb)
    @inbounds for j in 0:nb-1, k in 1:nnz_csr
      colVal[j * nnz_csr + k] = Ti(op.colval_base[k] + offset + j * ld)
    end
    A = CUSPARSE.CuSparseMatrixCSR{T, Ti}(
      op.rowPtr, CuVector{Ti}(colVal), vec(op.csr_nzval),
      (op.m * nb, ld * nb))
    descA = CUSPARSE.CuSparseMatrixDescriptor(A, 'O')
    descX = CUSPARSE.CuDenseVectorDescriptor(T, ld * nb)
    alpha = Ref{T}(one(T)); beta = Ref{T}(zero(T))
    algo = CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT
    buf_size = Ref{Csize_t}()
    CUSPARSE.cusparseSpMV_bufferSize(
      CUSPARSE.handle(), 'N', alpha, descA, descX, beta, op.descY, T, algo, buf_size)
    buffer = CuVector{UInt8}(undef, buf_size[])
    if CUSPARSE.version() >= v"12.3"
      CUSPARSE.cusparseSpMV_preprocess(
        CUSPARSE.handle(), 'N', alpha, descA, descX, beta, op.descY, T, algo, buffer)
    end
    CuBatchSpMVLayout{T, Ti}(A, descA, descX, ld * nb, buffer)
  end
end

@kernel function _scatter_spmv_out_kernel!(out, @Const(buf), m, alpha, beta)
  i, j = @index(Global, NTuple)
  @inbounds begin
    v = alpha * buf[(j - 1) * m + i]
    out[i, j] = iszero(beta) ? v : v + beta * out[i, j]
  end
end

function _batch_spmv_impl!(out::AbstractMatrix{T}, op::CuBatchSparseOperator{T},
                           B::AbstractMatrix{T}, alpha::T, beta::T,
                           val_offset::Int) where {T <: BlasFloat}
  m, nb = op.m, op.nbatch
  size(out, 1) == m || throw(DimensionMismatch("size(out, 1) != $m"))
  size(out, 2) == nb || throw(DimensionMismatch("size(out, 2) != $nb"))
  size(B, 2) == nb || throw(DimensionMismatch("size(B, 2) != $nb"))
  (m == 0 || nb == 0) && return out

  if size(op.csr_nzval, 1) == 0
    iszero(beta) ? fill!(out, zero(T)) : (out .*= beta)
    return out
  end

  Bd = B isa DenseCuMatrix ? B :
    throw(ArgumentError("CuBatchSparseOperator requires a contiguous CuMatrix operand, got $(typeof(B))."))
  layout = _spmv_layout!(op, size(Bd, 1), val_offset)
  length(Bd) == layout.xlen ||
    throw(DimensionMismatch("length(B) != $(layout.xlen)"))

  one_ref = Ref{T}(one(T)); zero_ref = Ref{T}(zero(T))
  CUSPARSE.cusparseDnVecSetValues(layout.descX, Bd)
  CUSPARSE.cusparseSpMV(
    CUSPARSE.handle(), 'N', one_ref, layout.descA, layout.descX, zero_ref,
    op.descY, T, CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT, layout.buffer)

  _scatter_spmv_out_kernel!(CUDABackend())(
    out, op.out_buf, m, alpha, beta; ndrange = (m, nb))
  return out
end
