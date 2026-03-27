function get_inf_pr!(inf_pr, c)
    batch_mapreduce!(abs, max, zero(eltype(inf_pr)), inf_pr, c)
    return inf_pr
end

function get_inf_du!(inf_du, f_vals, zl_vals, zu_vals, jacl_vals)
    batch_mapreduce!((f, zl, zu, jl) -> abs(f - zl + zu + jl), max, zero(eltype(inf_du)),
                     inf_du, f_vals, zl_vals, zu_vals, jacl_vals)
    return inf_du
end

function get_inf_compl!(inf_compl, x, xl, zl, xu, zu, sum_lb, sum_ub, nlb, nub)
    T = eltype(inf_compl)
    if nlb > 0
        batch_mapreduce!((x, xl, z) -> abs(x - xl) * z, max, zero(T),
                         sum_lb, lower(x), lower(xl), lower(zl))
    else
        fill!(sum_lb, zero(T))
    end
    if nub > 0
        batch_mapreduce!((xu, x, z) -> abs(xu - x) * z, max, zero(T),
                         sum_ub, upper(xu), upper(x), upper(zu))
    else
        fill!(sum_ub, zero(T))
    end
    @. inf_compl = max(sum_lb, sum_ub)
    return inf_compl
end

_adjust_bound_lb(x_lr::T, xl_r, c1, c2) where T =
    x_lr - xl_r < c1 ? xl_r - c2 * max(one(T), abs(x_lr)) : xl_r
_adjust_bound_ub(x_ur::T, xu_r, c1, c2) where T =
    xu_r - x_ur < c1 ? xu_r + c2 * max(one(T), abs(x_ur)) : xu_r

function MadNLP.adjust_boundary!(
    x_lr::AbstractMatrix{T},
    xl_r::AbstractMatrix{T},
    x_ur::AbstractMatrix{T},
    xu_r::AbstractMatrix{T},
    mu,
) where T
    c1 = eps(T) .* mu
    c2 = T(eps(T)^(3/4))
    xl_r .= _adjust_bound_lb.(x_lr, xl_r, c1, c2)
    xu_r .= _adjust_bound_ub.(x_ur, xu_r, c1, c2)
end
