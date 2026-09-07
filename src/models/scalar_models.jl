struct ModelData{T,VT,W,MQ,MA}
    A::MA
    Q::MQ
    lcon::VT
    ucon::VT
    lvar::VT
    uvar::VT
    c::VT
    c0::VT
    _v::W
end

const LPData{T,VT,MA} = ModelData{T,VT,Nothing,Nothing,MA}
const QPData{T,VT,W,MQ<:AbstractMatrix,MA} = ModelData{T,VT,W,MQ,MA}

_wrap_c0(c::AbstractVector{T}, c0::AbstractVector) where {T} = c0
_wrap_c0(c::AbstractVector{T}, c0::Base.RefValue) where {T} =
    (v = similar(c, T, 1); v[1] = c0[]; v)
_wrap_c0(c::AbstractVector{T}, c0::Number) where {T} = (v = similar(c, T, 1); v .= T(c0); v)

function QPData(
    A,
    c::VT,
    Q;
    lcon::VT = fill!(similar(c, size(A, 1)), eltype(c)(-Inf)),
    ucon::VT = fill!(similar(c, size(A, 1)), eltype(c)(Inf)),
    lvar::VT = fill!(similar(c), eltype(c)(-Inf)),
    uvar::VT = fill!(similar(c), eltype(c)(Inf)),
    c0 = zero(eltype(c)),
    _v = similar(c),
) where {VT}
    A_op = sparse_operator(A)
    Q_op = sparse_operator(Q; symmetric = true)
    return ModelData{eltype(c),VT,typeof(_v),typeof(Q_op),typeof(A_op)}(
        A_op,
        Q_op,
        lcon,
        ucon,
        lvar,
        uvar,
        c,
        _wrap_c0(c, c0),
        _v,
    )
end

function LPData(
    A,
    c::VT;
    lcon::VT = fill!(similar(c, size(A, 1)), eltype(c)(-Inf)),
    ucon::VT = fill!(similar(c, size(A, 1)), eltype(c)(Inf)),
    lvar::VT = fill!(similar(c), eltype(c)(-Inf)),
    uvar::VT = fill!(similar(c), eltype(c)(Inf)),
    c0 = zero(eltype(c)),
) where {VT}
    A_op = sparse_operator(A)
    return ModelData{eltype(c),VT,Nothing,Nothing,typeof(A_op)}(
        A_op,
        nothing,
        lcon,
        ucon,
        lvar,
        uvar,
        c,
        _wrap_c0(c, c0),
        nothing,
    )
end

mutable struct ScalarModel{T,VT,W,MQ,MA} <: NLPModels.AbstractNLPModel{T,VT}
    data::ModelData{T,VT,W,MQ,MA}
    meta::NLPModels.NLPModelMeta{T,VT}
    counters::NLPModels.Counters
end

const LinearModel{T,VT,MA} = ScalarModel{T,VT,Nothing,Nothing,MA}
const QuadraticModel{T,VT,W,MQ<:AbstractMatrix,MA} = ScalarModel{T,VT,W,MQ,MA}

function _scalar_meta(data; nnzh, islp, x0, y0, minimize, name)
    T = eltype(data.c);
    VT = typeof(data.c)
    isempty(data.c) &&
        throw(ArgumentError("Trivial models with no decision variables are not supported."))
    return NLPModels.NLPModelMeta{T,VT}(
        length(data.c);
        lvar = data.lvar,
        uvar = data.uvar,
        ncon = size(data.A, 1),
        lcon = data.lcon,
        ucon = data.ucon,
        nnzj = nnz(data.A),
        nnzh,
        x0,
        y0,
        minimize,
        islp,
        name,
    )
end

function ScalarModel(
    data::ModelData{T,VT};
    x0::VT = fill!(similar(data.c), zero(T)),
    y0::VT = fill!(similar(data.c, size(data.A, 1)), zero(T)),
    minimize::Bool = true,
    name::String = data.Q === nothing ? "LinearModel" : "QuadraticModel",
) where {T,VT}
    nnzh = data.Q === nothing ? 0 : nnz(data.Q)
    meta = _scalar_meta(data; nnzh, islp = nnzh == 0, x0, y0, minimize, name)
    return ScalarModel(data, meta, NLPModels.Counters())
end

LinearModel(data::LPData; kwargs...) = ScalarModel(data; kwargs...)
QuadraticModel(data::QPData; kwargs...) = ScalarModel(data; kwargs...)


NLPModels.jac_structure!(
    m::ScalarModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) = _copy_sparse_structure!(m.data.A, rows, cols)

function NLPModels.jac_coord!(m::ScalarModel, x::AbstractVector, jac::AbstractVector)
    NLPModels.increment!(m, :neval_jac)
    return _copy_sparse_values!(m.data.A, jac)
end

function NLPModels.cons!(m::ScalarModel, x::AbstractVector, c::AbstractVector)
    NLPModels.increment!(m, :neval_cons)
    mul!(c, m.data.A, x)
    return c
end

function NLPModels.jprod!(
    m::ScalarModel,
    x::AbstractVector,
    v::AbstractVector,
    jv::AbstractVector,
)
    NLPModels.increment!(m, :neval_jprod)
    mul!(jv, m.data.A, v)
    return jv
end

function NLPModels.jtprod!(
    m::ScalarModel,
    x::AbstractVector,
    v::AbstractVector,
    jtv::AbstractVector,
)
    NLPModels.increment!(m, :neval_jtprod)
    _mul_jt!(jtv, m.data.A, v)
    return jtv
end

@inline _scalar_c0(c0::Vector) = @inbounds c0[1]
@inline _scalar_c0(c0::AbstractVector) = sum(c0)

function NLPModels.obj(lp::LinearModel, x::AbstractVector)
    NLPModels.increment!(lp, :neval_obj)
    return _scalar_c0(lp.data.c0) + dot(lp.data.c, x)
end

function NLPModels.grad!(lp::LinearModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(lp, :neval_grad)
    copyto!(g, lp.data.c)
    return g
end

NLPModels.hess_structure!(
    ::LinearModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) = (rows, cols)
NLPModels.hess_coord!(
    ::LinearModel,
    ::AbstractVector,
    hess::AbstractVector;
    obj_weight::Real = 1,
) = hess
NLPModels.hess_coord!(
    ::LinearModel,
    ::AbstractVector,
    ::AbstractVector,
    hess::AbstractVector;
    obj_weight::Real = 1,
) = hess

function NLPModels.obj(qp::QuadraticModel, x::AbstractVector)
    NLPModels.increment!(qp, :neval_obj)
    mul!(qp.data._v, qp.data.Q, x)
    return _scalar_c0(qp.data.c0) + dot(qp.data.c, x) + dot(qp.data._v, x) / 2
end

function NLPModels.grad!(qp::QuadraticModel, x::AbstractVector, g::AbstractVector)
    NLPModels.increment!(qp, :neval_grad)
    copyto!(g, qp.data.c)
    mul!(g, qp.data.Q, x, one(eltype(x)), one(eltype(x)))
    return g
end

NLPModels.hess_structure!(
    qp::QuadraticModel,
    rows::AbstractVector{<:Integer},
    cols::AbstractVector{<:Integer},
) = _copy_sparse_structure!(qp.data.Q, rows, cols)

function NLPModels.hess_coord!(
    qp::QuadraticModel,
    x::AbstractVector,
    hess::AbstractVector;
    obj_weight::Real = 1,
)
    NLPModels.increment!(qp, :neval_hess)
    _copy_sparse_values!(qp.data.Q, hess)
    hess .*= obj_weight
    return hess
end

NLPModels.hess_coord!(
    qp::QuadraticModel,
    x::AbstractVector,
    y::AbstractVector,
    hess::AbstractVector;
    obj_weight::Real = 1,
) = NLPModels.hess_coord!(qp, x, hess; obj_weight)
