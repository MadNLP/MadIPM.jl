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

