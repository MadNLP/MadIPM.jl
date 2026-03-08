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
