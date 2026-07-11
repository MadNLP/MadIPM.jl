Base.@propagate_inbounds _src_getindex(srcs::Tuple, i, j) =
  (srcs[1][i, j], _src_getindex(Base.tail(srcs), i, j)...)
Base.@propagate_inbounds _src_getindex(srcs::Tuple{Any}, i, j) = (srcs[1][i, j],)
Base.@propagate_inbounds _src_getindex(srcs::Tuple{}, i, j) = ()

const _AnyCuMat{T} = Union{CuMatrix{T}, SubArray{T, 2, <:CuArray{T, 2}, <:Tuple, false}}

_batch_mapreduce_kernel(f::F, op::OP, neutral::T, out, srcs::Tuple{Vararg{Any, N}}) where {F, OP, T, N} = begin
  j = blockIdx().x; bs = size(out, 2); nrows = size(first(srcs), 1)
  @inbounds if j <= bs
    val = neutral
    i = threadIdx().x
    while i <= nrows
      val = op(val, f(_src_getindex(srcs, i, j)...))
      i += blockDim().x
    end
    val = CUDA.reduce_block(op, val, neutral, Val(true))
    if threadIdx().x == 1
      out[1, j] = val
    end
  end
  return nothing
end

function batch_mapreduce!(f, op, neutral::T, out::_AnyCuMat{T}, srcs::_AnyCuMat{T}...) where {T}
  nrows = size(first(srcs), 1)
  fill!(out, neutral)
  nrows == 0 && return out
  kernel = @cuda launch = false _batch_mapreduce_kernel(f, op, neutral, out, srcs)
  config = launch_configuration(kernel.fun)
  threads = max(1, (config.threads ÷ 32) * 32)
  kernel(f, op, neutral, out, srcs; threads, blocks = size(out, 2))
  return out
end
