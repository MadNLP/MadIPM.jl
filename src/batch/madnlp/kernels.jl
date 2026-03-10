function get_inf_pr!(inf_pr, c, scratch)
    @. scratch = abs(c)
    maximum!(inf_pr, scratch)
    return inf_pr
end

function get_inf_du!(inf_du, f_vals, zl_vals, zu_vals, jacl_vals, scratch)
    @. scratch = abs(f_vals - zl_vals + zu_vals + jacl_vals)
    maximum!(inf_du, scratch)
    return inf_du
end

function get_inf_compl!(inf_compl, x, xl, zl, xu, zu,
                        scratch_lb, scratch_ub, sum_lb, sum_ub, nlb, nub)
    T = eltype(inf_compl)
    if nlb > 0
        x_lr = lower(x); xl_r = lower(xl); zl_r = lower(zl)
        @. scratch_lb = abs(x_lr - xl_r) * zl_r
        maximum!(sum_lb, scratch_lb)
    else
        fill!(sum_lb, zero(T))
    end
    if nub > 0
        xu_r = upper(xu); x_ur = upper(x); zu_r = upper(zu)
        @. scratch_ub = abs(xu_r - x_ur) * zu_r
        maximum!(sum_ub, scratch_ub)
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
