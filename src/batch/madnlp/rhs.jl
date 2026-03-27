struct BatchUnreducedKKTVector{T, MT<:AbstractMatrix{T}, VI}
    values::MT
    n::Int
    m::Int
    nlb::Int
    nub::Int
    ind_lb::VI
    ind_ub::VI
    _primal::SubArray
    _dual::SubArray
    _primal_dual::SubArray
    _dual_lb::SubArray
    _dual_ub::SubArray
    _xp_lr::SubArray
    _xp_ur::SubArray
end

function BatchUnreducedKKTVector(
    ::Type{MT}, ::Type{VT},
    n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int,
    ind_lb, ind_ub,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = n + m + nlb + nub
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    return BatchUnreducedKKTVector{T, MT, typeof(ind_lb)}(
        values, n, m, nlb, nub, ind_lb, ind_ub,
        view(values, 1:n, :),
        view(values, n+1:n+m, :),
        view(values, 1:n+m, :),
        view(values, n+m+1:n+m+nlb, :),
        view(values, n+m+nlb+1:n+m+nlb+nub, :),
        view(values, ind_lb, :),
        view(values, ind_ub, :),
    )
end

MadNLP.full(bv::BatchUnreducedKKTVector) = bv.values
MadNLP.primal(bv::BatchUnreducedKKTVector) = bv._primal
MadNLP.dual(bv::BatchUnreducedKKTVector) = bv._dual
MadNLP.primal_dual(bv::BatchUnreducedKKTVector) = bv._primal_dual
MadNLP.dual_lb(bv::BatchUnreducedKKTVector) = bv._dual_lb
MadNLP.dual_ub(bv::BatchUnreducedKKTVector) = bv._dual_ub
xp_lr(bv::BatchUnreducedKKTVector) = bv._xp_lr
xp_ur(bv::BatchUnreducedKKTVector) = bv._xp_ur

struct BatchPrimalVector{T, MT<:AbstractMatrix{T}, VI}
    values::MT
    nx::Int
    ns::Int
    ind_lb::VI
    ind_ub::VI
    _variable::SubArray
    _slack::SubArray
    _lower::SubArray
    _upper::SubArray
end

function BatchPrimalVector(
    ::Type{MT}, ::Type{VT},
    nx::Int, ns::Int, batch_size::Int,
    ind_lb, ind_ub,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = nx + ns
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    return BatchPrimalVector{T, MT, typeof(ind_lb)}(
        values, nx, ns, ind_lb, ind_ub,
        view(values, 1:nx, :),
        view(values, nx+1:nx+ns, :),
        view(values, ind_lb, :),
        view(values, ind_ub, :),
    )
end

MadNLP.variable(bpv::BatchPrimalVector) = bpv._variable
MadNLP.slack(bpv::BatchPrimalVector) = bpv._slack
lower(bpv::BatchPrimalVector) = bpv._lower
upper(bpv::BatchPrimalVector) = bpv._upper
MadNLP.full(bpv::BatchPrimalVector) = bpv.values
MadNLP.primal(bpv::BatchPrimalVector) = bpv.values
