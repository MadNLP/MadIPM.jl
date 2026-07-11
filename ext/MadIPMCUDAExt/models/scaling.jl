const CuSparseCOO{T} = Models.Scaling.SparseCOO{T, Ti, V, Vi} where
  {Ti, V <: CuVector{T}, Vi <: CuVector{Ti}}

@inline _atomic_max!(dst, i, value) = begin
  old = dst[i]
  while value > old
    result = Atomix.@atomicreplace dst[i] old => value
    old = result.old
    result.success && break
  end
  return
end

@inline _atomic_min!(dst, i, value) = begin
  old = dst[i]
  while value < old
    result = Atomix.@atomicreplace dst[i] old => value
    old = result.old
    result.success && break
  end
  return
end

@kernel function _coo_scaled_maxabs_kernel!(rownrm, colnrm, @Const(rowpair), @Const(colpair),
                                            @Const(rowval), @Const(colval), @Const(nzval),
                                            @Const(drow), @Const(dcol))
  k = @index(Global, Linear)
  @inbounds begin
    i = rowval[k]; j = colval[k]
    ri = rowpair[i] > 0
    cj = colpair[j] > 0
    if ri || cj
      s = abs(nzval[k]) / (drow[i] * dcol[j])
      ri && _atomic_max!(rownrm, i, s)
      cj && _atomic_max!(colnrm, j, s)
    end
  end
end

@kernel function _coo_scaled_argmax_kernel!(rowcand, colcand, @Const(rownrm), @Const(colnrm),
                                            @Const(rowpair), @Const(colpair),
                                            @Const(rowval), @Const(colval), @Const(nzval),
                                            @Const(drow), @Const(dcol))
  k = @index(Global, Linear)
  @inbounds begin
    i = rowval[k]; j = colval[k]
    ri = rowpair[i] > 0 && rownrm[i] > 0
    cj = colpair[j] > 0 && colnrm[j] > 0
    if ri || cj
      s = abs(nzval[k]) / (drow[i] * dcol[j])
      ri && s == rownrm[i] && _atomic_min!(rowcand, i, Int(j))
      cj && s == colnrm[j] && _atomic_min!(colcand, j, Int(i))
    end
  end
end

function Models.Scaling.scaled_maxabs!(rownrm::CuVector{T}, colnrm::CuVector{T},
                                       rowpair::CuVector{Int}, colpair::CuVector{Int},
                                       A::CuSparseCOO{T},
                                       drow::CuVector{T}, dcol::CuVector{T}) where {T <: AbstractFloat}
  isempty(A.nzval) && return rownrm, colnrm
  _coo_scaled_maxabs_kernel!(CUDABackend())(
    rownrm, colnrm, rowpair, colpair, A.rowval, A.colval, A.nzval, drow, dcol;
    ndrange = length(A.nzval))
  return rownrm, colnrm
end

function Models.Scaling.scaled_argmax!(rowcand::CuVector{Int}, colcand::CuVector{Int},
                                       rownrm::CuVector{T}, colnrm::CuVector{T},
                                       rowpair::CuVector{Int}, colpair::CuVector{Int},
                                       A::CuSparseCOO{T},
                                       drow::CuVector{T}, dcol::CuVector{T}) where {T <: AbstractFloat}
  isempty(A.nzval) && return rowcand, colcand
  _coo_scaled_argmax_kernel!(CUDABackend())(
    rowcand, colcand, rownrm, colnrm, rowpair, colpair,
    A.rowval, A.colval, A.nzval, drow, dcol;
    ndrange = length(A.nzval))
  return rowcand, colcand
end

@kernel function _settle_pairs_kernel!(rowpair, colpair)
  j = @index(Global, Linear)
  @inbounds begin
    i = colpair[j]
    if i > 0 && rowpair[i] == j
      rowpair[i] = 0
      colpair[j] = 0
    end
  end
end

function Models.Scaling.settle_pairs!(rowpair::CuVector{Int}, colpair::CuVector{Int})
  isempty(colpair) && return nothing
  _settle_pairs_kernel!(CUDABackend())(rowpair, colpair; ndrange = length(colpair))
  return nothing
end
