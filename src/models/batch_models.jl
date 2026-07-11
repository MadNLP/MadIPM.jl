struct BatchModel{T, MT <: AbstractMatrix{T},
                  AOp, QOp, CT, C0T, HX} <: NLPModels.AbstractBatchNLPModel{T, MT}
  meta::NLPModels.BatchNLPModelMeta{T, MT}
  c_batch::CT
  c0_batch::C0T
  A::AOp
  Q::QOp
  _HX::HX
end

const _AnyOp = Union{AbstractSparseOperator, BatchSparseOperator}

const BatchQuadraticModel{T, MT, AOp <: _AnyOp, QOp <: _AnyOp, CT, C0T} =
  BatchModel{T, MT, AOp, QOp, CT, C0T, MT}
const ObjRHSBatchQuadraticModel{T, MT, AOp <: AbstractSparseOperator,
                                QOp <: AbstractSparseOperator, CT, C0T} =
  BatchModel{T, MT, AOp, QOp, CT, C0T, MT}
const UniformBatchQuadraticModel{T, MT, AOp <: BatchSparseOperator,
                                 QOp <: BatchSparseOperator, CT, C0T} =
  BatchModel{T, MT, AOp, QOp, CT, C0T, MT}

const BatchLinearModel{T, MT, AOp <: _AnyOp, CT, C0T} =
  BatchModel{T, MT, AOp, Nothing, CT, C0T, Nothing}
const ObjRHSBatchLinearModel{T, MT, AOp <: AbstractSparseOperator, CT, C0T} =
  BatchModel{T, MT, AOp, Nothing, CT, C0T, Nothing}
const UniformBatchLinearModel{T, MT, AOp <: BatchSparseOperator, CT, C0T} =
  BatchModel{T, MT, AOp, Nothing, CT, C0T, Nothing}

function _batch_meta(::Type{T}, ::Type{MT}, meta, nbatch;
                     x0, y0 = fill!(MT(undef, meta.ncon, nbatch), zero(T)),
                     lvar, uvar, lcon, ucon,
                     nnzh = meta.nnzh, islp = meta.islp, name = meta.name) where {T, MT}
  return NLPModels.BatchNLPModelMeta{T, MT}(nbatch, meta.nvar;
    x0, lvar, uvar, ncon = meta.ncon, y0, lcon, ucon,
    nnzj = meta.nnzj, nnzh, minimize = meta.minimize, islp, name)
end

function _adapt_batch_meta(to, meta::NLPModels.BatchNLPModelMeta{T}) where {T}
  x0 = Adapt.adapt(to, meta.x0)
  return _batch_meta(T, typeof(x0), meta, meta.nbatch;
    x0,
    y0   = Adapt.adapt(to, meta.y0),
    lvar = Adapt.adapt(to, meta.lvar),
    uvar = Adapt.adapt(to, meta.uvar),
    lcon = Adapt.adapt(to, meta.lcon),
    ucon = Adapt.adapt(to, meta.ucon),
  )
end

function _stacked_batch(models::Vector, shared_A, shared_c, shared_c0, name, MT)
  @assert !isempty(models) "Need at least one model"
  m1     = first(models)
  T      = eltype(m1.data.c)
  nbatch = length(models)
  MT     = MT === nothing ? typeof(similar(m1.data.c, T, 0, 0)) : MT

  shared_A  = shared_A  === nothing ? _all_equal(models, m -> _sparse_values(m.data.A)) : shared_A
  shared_c  = shared_c  === nothing ? _all_equal(models, m -> m.data.c)                 : shared_c
  shared_c0 = shared_c0 === nothing ? _all_equal(models, m -> @inbounds m.data.c0[1])   : shared_c0

  c_batch  = shared_c ? copyto!(similar(MT(undef, 0, 0), T, m1.meta.nvar), m1.data.c) :
                        _stack_columns(MT, models, m -> m.data.c)
  c0_batch = shared_c0 ? (@inbounds m1.data.c0[1]) : T[@inbounds(m.data.c0[1]) for m in models]

  A = shared_A ? sparse_operator(m1.data.A) :
        _jacobian_op(m1, _stack_columns(MT, models, m -> _sparse_values(m.data.A)))

  meta = _batch_meta(T, MT, m1.meta, nbatch;
    x0   = _stack_columns(MT, models, m -> m.meta.x0),
    y0   = _stack_columns(MT, models, m -> m.meta.y0),
    lvar = _stack_columns(MT, models, m -> m.meta.lvar),
    uvar = _stack_columns(MT, models, m -> m.meta.uvar),
    lcon = _stack_columns(MT, models, m -> m.meta.lcon),
    ucon = _stack_columns(MT, models, m -> m.meta.ucon),
    name)
  return m1, MT, meta, c_batch, c0_batch, A
end

"""
    batch_model(models::Vector; shared_A, shared_Q, shared_c, shared_c0, name, MT)

Build a batched model from scalar LPs or QPs. Each `shared_*` kwarg picks
shared (`true`) or per-instance (`false`) storage; `nothing` auto-detects
(shared iff values match across the batch). Callers must ensure all instances
share sparsity patterns and bound kinds.
"""
function batch_model(models::Vector{<:ScalarModel};
                     shared_A  = nothing, shared_Q  = nothing,
                     shared_c  = nothing, shared_c0 = nothing,
                     name::String = "BatchModel",
                     MT = nothing)
  m1, MT, meta, c_batch, c0_batch, A = _stacked_batch(models, shared_A, shared_c, shared_c0, name, MT)
  Q, HX = if m1 isa QuadraticModel
    shared_Q = shared_Q === nothing ? _all_equal(models, m -> _sparse_values(m.data.Q)) : shared_Q
    Qop = shared_Q ? sparse_operator(m1.data.Q; symmetric = true) :
            _hessian_op(m1, _stack_columns(MT, models, m -> _sparse_values(m.data.Q)))
    Qop, MT(undef, m1.meta.nvar, length(models))
  else
    nothing, nothing
  end
  return BatchModel(meta, c_batch, c0_batch, A, Q, HX)
end

BatchQuadraticModel(qps::Vector{<:QuadraticModel}; name::String = "BatchQP", kwargs...) =
  batch_model(qps; name, kwargs...)
BatchLinearModel(lps::Vector{<:LinearModel}; name::String = "BatchLP", kwargs...) =
  batch_model(lps; name, kwargs...)
ObjRHSBatchQuadraticModel(qps::Vector{<:QuadraticModel}; name::String = "ObjRHSBatchQP", MT = nothing) =
  batch_model(qps; shared_A = true, shared_Q = true, name, MT)
ObjRHSBatchLinearModel(lps::Vector{<:LinearModel}; name::String = "ObjRHSBatchLP", MT = nothing) =
  batch_model(lps; shared_A = true, name, MT)

function Adapt.adapt_structure(to, bm::BatchModel)
  return BatchModel(
    _adapt_batch_meta(to, bm.meta),
    Adapt.adapt(to, bm.c_batch),
    Adapt.adapt(to, bm.c0_batch),
    Adapt.adapt(to, bm.A),
    Adapt.adapt(to, bm.Q),
    Adapt.adapt(to, bm._HX),
  )
end


NLPModels.cons!(bm::BatchModel, bx::AbstractMatrix, bc::AbstractMatrix) =
  (mul!(bc, bm.A, bx); bc)

NLPModels.jac_structure!(bm::BatchModel, jrows::AbstractVector{<:Integer}, jcols::AbstractVector{<:Integer}) =
  (@lencheck bm.meta.nnzj jrows jcols; _copy_sparse_structure!(bm.A, jrows, jcols))

NLPModels.jac_coord!(bm::BatchModel, ::AbstractMatrix, bjvals::AbstractMatrix) =
  _copy_sparse_values!(bjvals, bm.A)

function NLPModels.obj!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  bqp._HX .= bqp.c_batch
  mul!(bqp._HX, bqp.Q, bx, T(0.5), one(T))
  batch_mapreduce!(*, +, zero(T), reshape(bf, 1, length(bf)), bqp._HX, bx)
  bf .+= bqp.c0_batch
  return bf
end

function NLPModels.grad!(bqp::BatchQuadraticModel{T}, bx::AbstractMatrix, bg::AbstractMatrix) where {T}
  mul!(bg, bqp.Q, bx)
  bg .+= bqp.c_batch
  return bg
end

NLPModels.hess_structure!(bqp::BatchQuadraticModel, hrows::AbstractVector{<:Integer}, hcols::AbstractVector{<:Integer}) =
  _copy_sparse_structure!(bqp.Q, hrows, hcols)

function NLPModels.hess_coord!(bqp::BatchQuadraticModel, ::AbstractMatrix, ::AbstractMatrix,
                                weights::AbstractVector, bhvals::AbstractMatrix)
  bhvals .= _sparse_values(bqp.Q) .* weights'
  return bhvals
end

function NLPModels.obj!(blm::BatchLinearModel{T}, bx::AbstractMatrix, bf::AbstractVector) where {T}
  _lp_obj!(reshape(bf, 1, length(bf)), blm.c_batch, bx)
  bf .+= blm.c0_batch
  return bf
end

_lp_obj!(out::AbstractMatrix{T}, c::AbstractVector, bx::AbstractMatrix) where {T} =
  (mul!(vec(out), transpose(bx), c); out)
_lp_obj!(out::AbstractMatrix{T}, c::AbstractMatrix, bx::AbstractMatrix) where {T} =
  batch_mapreduce!(*, +, zero(T), out, c, bx)

function NLPModels.grad!(blm::BatchLinearModel, ::AbstractMatrix, bg::AbstractMatrix)
  bg .= blm.c_batch
  return bg
end

NLPModels.hess_structure!(::BatchLinearModel, rows::AbstractVector{<:Integer}, cols::AbstractVector{<:Integer}) = (rows, cols)
NLPModels.hess_coord!(::BatchLinearModel, ::AbstractMatrix, ::AbstractMatrix,
                      ::AbstractVector, bhvals::AbstractMatrix) = bhvals
