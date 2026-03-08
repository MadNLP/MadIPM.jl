struct BatchUnreducedKKTVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT
    views::Vector{VT}
    n::Int
    m::Int
    nlb::Int
    nub::Int
    ind_lb::VI
    ind_ub::VI
end

function BatchUnreducedKKTVector(
    ::Type{MT}, ::Type{VT},
    n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int,
    ind_lb, ind_ub,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = n + m + nlb + nub
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    views = Vector{VT}(undef, batch_size)
    for i in 1:batch_size
        col_start = (i-1) * total + 1
        views[i] = _madnlp_unsafe_column_wrap(values, total, col_start, VT)
    end

    return BatchUnreducedKKTVector{T, MT, VT, typeof(ind_lb)}(values, views, n, m, nlb, nub, ind_lb, ind_ub)
end

MadNLP.full(bv::BatchUnreducedKKTVector) = bv.values
MadNLP.primal(bv::BatchUnreducedKKTVector) = view(bv.values, 1:bv.n, :)
MadNLP.dual(bv::BatchUnreducedKKTVector) = view(bv.values, bv.n+1:bv.n+bv.m, :)
MadNLP.primal_dual(bv::BatchUnreducedKKTVector) = view(bv.values, 1:bv.n+bv.m, :)
MadNLP.dual_lb(bv::BatchUnreducedKKTVector) = view(bv.values, bv.n+bv.m+1:bv.n+bv.m+bv.nlb, :)
MadNLP.dual_ub(bv::BatchUnreducedKKTVector) = view(bv.values, bv.n+bv.m+bv.nlb+1:bv.n+bv.m+bv.nlb+bv.nub, :)
xp_lr(bv::BatchUnreducedKKTVector) = view(bv.values, bv.ind_lb, :)
xp_ur(bv::BatchUnreducedKKTVector) = view(bv.values, bv.ind_ub, :)

struct BatchPrimalVector{T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}, VI}
    values::MT
    views::Vector{VT}
    nx::Int
    ns::Int
    ind_lb::VI
    ind_ub::VI
end

function BatchPrimalVector(
    ::Type{MT}, ::Type{VT},
    nx::Int, ns::Int, batch_size::Int,
    ind_lb, ind_ub,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    total = nx + ns
    values = MT(undef, total, batch_size)
    fill!(values, zero(T))

    views = Vector{VT}(undef, batch_size)
    for i in 1:batch_size
        col_start = (i-1) * total + 1
        views[i] = _madnlp_unsafe_column_wrap(values, total, col_start, VT)
    end

    return BatchPrimalVector{T, MT, VT, typeof(ind_lb)}(values, views, nx, ns, ind_lb, ind_ub)
end

MadNLP.variable(bpv::BatchPrimalVector) = view(bpv.values, 1:bpv.nx, :)
MadNLP.slack(bpv::BatchPrimalVector) = view(bpv.values, bpv.nx+1:bpv.nx+bpv.ns, :)
lower(bpv::BatchPrimalVector) = view(bpv.values, bpv.ind_lb, :)
upper(bpv::BatchPrimalVector) = view(bpv.values, bpv.ind_ub, :)
MadNLP.full(bpv::BatchPrimalVector) = bpv.values
MadNLP.primal(bpv::BatchPrimalVector) = bpv.values
