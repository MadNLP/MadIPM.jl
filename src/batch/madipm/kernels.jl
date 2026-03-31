function dual_objective!(dual_obj, y_vals, rhs_vals, zl_r, xl_r, zu_r, xu_r,
                         sum_lb, sum_ub, nlb, nub)
    T = eltype(dual_obj)
    batch_mapreduce!(*, +, zero(T), dual_obj, y_vals, rhs_vals)
    dual_obj .*= -one(T)
    if nlb > 0
        batch_mapreduce!(*, +, zero(T), sum_lb, zl_r, xl_r)
        dual_obj .+= sum_lb
    end
    if nub > 0
        batch_mapreduce!(*, +, zero(T), sum_ub, zu_r, xu_r)
        dual_obj .-= sum_ub
    end
    return dual_obj
end

function set_initial_primal_rhs!(solver::AbstractBatchMPCSolver)
    p = solver.p
    fill!(MadNLP.full(p), 0.0)
    py = MadNLP.dual(p)
    b = MadNLP.full(solver.c)

    py .= .- b
    return
end

function set_initial_dual_rhs!(solver::AbstractBatchMPCSolver)
    p = solver.p
    fill!(MadNLP.full(p), 0.0)
    px = MadNLP.primal(p)
    c = MadNLP.primal(solver.f)

    px .= .- c
    return
end

function set_predictive_rhs!(solver::AbstractBatchMPCSolver, kkt::AbstractBatchKKTSystem)
    px = MadNLP.primal(solver.p)
    py = MadNLP.dual(solver.p)
    pzl = MadNLP.dual_lb(solver.p)
    pzu = MadNLP.dual_ub(solver.p)
    f = MadNLP.primal(solver.f)
    c = MadNLP.full(solver.c)
    zl = MadNLP.full(solver.zl)
    zu = MadNLP.full(solver.zu)
    jacl = MadNLP.full(solver.jacl)
    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)

    fill!(MadNLP.full(solver.p), 0.0)

    px .= .-f .+ zl .- zu .- jacl
    py .= .-c
    pzl .= (xl_r .- x_lr) .* zl_r
    pzu .= (xu_r .- x_ur) .* zu_r
    return
end

function set_correction_rhs!(bs::AbstractBatchMPCSolver, kkt::AbstractBatchKKTSystem, mu, correction_lb, correction_ub, ind_lb, ind_ub)
    px = MadNLP.primal(bs.p)
    py = MadNLP.dual(bs.p)
    pzl = MadNLP.dual_lb(bs.p)
    pzu = MadNLP.dual_ub(bs.p)
    f = MadNLP.primal(bs.f)
    c = MadNLP.full(bs.c)
    zl = MadNLP.full(bs.zl)
    zu = MadNLP.full(bs.zu)
    jacl = MadNLP.full(bs.jacl)
    xl_r = lower(bs.xl)
    x_lr = lower(bs.x)
    zl_r = lower(bs.zl)
    xu_r = upper(bs.xu)
    x_ur = upper(bs.x)
    zu_r = upper(bs.zu)

    px .= .-f .+ zl .- zu .- jacl
    py .= .-c
    pzl .= (xl_r .- x_lr) .* zl_r .+ mu .- correction_lb
    pzu .= (xu_r .- x_ur) .* zu_r .- mu .- correction_ub
    return
end

function get_correction!(
    batch_solver::AbstractBatchMPCSolver,
    correction_lb,
    correction_ub
)
    dlb = MadNLP.dual_lb(batch_solver.d)
    dub = MadNLP.dual_ub(batch_solver.d)

    dx_lr = xp_lr(batch_solver.d)
    dx_ur = xp_ur(batch_solver.d)

    correction_lb .= dx_lr .* dlb
    correction_ub .= dx_ur .* dub
    return
end

function _set_aug_diagonal_reg_unmasked!(kkt, solver::AbstractBatchMPCSolver)
    kkt.reg .= solver.del_w
    du_diag(kkt) .= solver.del_c
    kkt.l_diag .= lower(solver.xl) .- lower(solver.x)
    kkt.u_diag .= upper(solver.x) .- upper(solver.xu)
    kkt.l_lower .= lower(solver.zl)
    kkt.u_lower .= upper(solver.zu)
    pr_diag(kkt) .= kkt.reg
    pr_diag_lb = view(kkt.nzVals, _get_ind_lb(solver), :)
    pr_diag_ub = view(kkt.nzVals, _get_ind_ub(solver), :)
    pr_diag_lb .-= kkt.l_lower ./ kkt.l_diag
    pr_diag_ub .-= kkt.u_lower ./ kkt.u_diag
    return
end

function _set_aug_diagonal_reg_masked!(kkt, solver::AbstractBatchMPCSolver)
    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)
    mask = solver.workspace.active_mask
    _du = du_diag(kkt)
    _pr = pr_diag(kkt)
    @. kkt.reg = ifelse(mask == 1, solver.del_w, kkt.reg)
    @. _du = ifelse(mask == 1, solver.del_c, _du)
    @. kkt.l_diag = ifelse(mask == 1, xl_r - x_lr, kkt.l_diag)
    @. kkt.u_diag = ifelse(mask == 1, x_ur - xu_r, kkt.u_diag)
    @. kkt.l_lower = ifelse(mask == 1, zl_r, kkt.l_lower)
    @. kkt.u_lower = ifelse(mask == 1, zu_r, kkt.u_lower)
    @. _pr = ifelse(mask == 1, kkt.reg, _pr)
    pr_diag_lb = view(kkt.nzVals, _get_ind_lb(solver), :)
    pr_diag_ub = view(kkt.nzVals, _get_ind_ub(solver), :)
    @. pr_diag_lb = ifelse(mask == 1, pr_diag_lb - kkt.l_lower / kkt.l_diag, pr_diag_lb)
    @. pr_diag_ub = ifelse(mask == 1, pr_diag_ub - kkt.u_lower / kkt.u_diag, pr_diag_ub)
    return
end

function set_aug_diagonal_reg!(kkt, solver::AbstractBatchMPCSolver)
    if is_identity_view(active_view(solver.batch_views))
        _set_aug_diagonal_reg_unmasked!(kkt, solver)
    else
        _set_aug_diagonal_reg_masked!(kkt, solver)
    end
end

function get_complementarity_measure!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    nlb, nub = solver.d.nlb, solver.d.nub
    T = eltype(ws.mu_curr)

    if nlb + nub == 0
        fill!(ws.mu_curr, zero(T))
        return ws.mu_curr
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)

    batch_mapreduce!((x, xl, z) -> (x - xl) * z, +, zero(T), ws.sum_lb, x_lr, xl_r, zl_r)
    batch_mapreduce!((xu, x, z) -> (xu - x) * z, +, zero(T), ws.sum_ub, xu_r, x_ur, zu_r)
    @. ws.mu_curr = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_curr
end

function get_affine_complementarity_measure!(solver::AbstractBatchMPCSolver, alpha_p, alpha_d)
    ws = solver.workspace
    nlb, nub = solver.d.nlb, solver.d.nub
    T = eltype(ws.mu_affine)

    if nlb + nub == 0
        fill!(ws.mu_affine, zero(T))
        return ws.mu_affine
    end

    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)
    dx_lr = xp_lr(solver.d)
    dx_ur = xp_ur(solver.d)
    dzlb = MadNLP.dual_lb(solver.d)
    dzub = MadNLP.dual_ub(solver.d)

    _affine_compl_lb!(ws.sum_lb, x_lr, xl_r, zl_r, dx_lr, dzlb, alpha_p, alpha_d)
    _affine_compl_ub!(ws.sum_ub, xu_r, x_ur, zu_r, dx_ur, dzub, alpha_p, alpha_d)
    @. ws.mu_affine = (ws.sum_lb + ws.sum_ub) / (nlb + nub)
    return ws.mu_affine
end

function update_barrier!(::Mehrotra, solver::AbstractBatchMPCSolver, mu_affine)
    ws = solver.workspace
    T = eltype(ws.mu_curr)

    has_inequalities = (solver.d.nlb + solver.d.nub) > 0

    get_complementarity_measure!(solver)

    if has_inequalities
        @. ws.mu_batch = clamp((ws.mu_affine / ws.mu_curr) ^ 3, T(1e-6), T(10.0))
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_batch * ws.mu_curr)
    else
        @. ws.mu_batch = max(solver.opt.mu_min, ws.mu_curr)
    end
    return
end

function get_fraction_to_boundary_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    nlb, nub = d.nlb, d.nub
    T = eltype(ws.alpha_p)

    if nlb > 0
        _ftb_primal_lb!(ws.alpha_xl, xp_lr(d), lower(x), lower(xl), ws.tau)
        _ftb_dual_lb!(ws.alpha_zl, MadNLP.dual_lb(d), lower(zl), ws.tau)
    else
        fill!(ws.alpha_xl, one(T))
        fill!(ws.alpha_zl, one(T))
    end

    if nub > 0
        _ftb_primal_ub!(ws.alpha_xu, xp_ur(d), upper(x), upper(xu), ws.tau)
        _ftb_dual_ub!(ws.alpha_zu, MadNLP.dual_ub(d), upper(zu), ws.tau)
    else
        fill!(ws.alpha_xu, one(T))
        fill!(ws.alpha_zu, one(T))
    end

    ws.alpha_p .= min.(ws.alpha_xl, ws.alpha_xu, one(T))
    ws.alpha_d .= min.(ws.alpha_zl, ws.alpha_zu, one(T))
    return
end

function _ftb_primal_lb!(alpha_out, dx, x, xb, tau)
    T = eltype(alpha_out)
    n, bs = size(dx)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d < zero(T) || continue
            a = min(a, (-x[i, j] + xb[i, j]) * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _ftb_primal_ub!(alpha_out, dx, x, xb, tau)
    T = eltype(alpha_out)
    n, bs = size(dx)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dx[i, j]
            d > zero(T) || continue
            a = min(a, (-x[i, j] + xb[i, j]) * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _ftb_dual_lb!(alpha_out, dz, z, tau)
    T = eltype(alpha_out)
    n, bs = size(dz)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dz[i, j]
            d < zero(T) || continue
            a = min(a, -z[i, j] * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _ftb_dual_ub!(alpha_out, dz, z, tau)
    T = eltype(alpha_out)
    n, bs = size(dz)
    @inbounds for j in 1:bs
        a = T(Inf)
        τ = tau[1, j]
        for i in 1:n
            d = dz[i, j]
            (d < zero(T) && z[i, j] + d < zero(T)) || continue
            a = min(a, -z[i, j] * τ / d)
        end
        alpha_out[1, j] = a
    end
end

function _affine_compl_lb!(out, x, xl, z, dx, dz, αp, αd)
    T = eltype(out)
    n, bs = size(x)
    @inbounds for j in 1:bs
        s = zero(T)
        ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (x[i,j] + ap * dx[i,j] - xl[i,j]) * (z[i,j] + ad * dz[i,j])
        end
        out[1, j] = s
    end
end

function _affine_compl_ub!(out, xu, x, z, dx, dz, αp, αd)
    T = eltype(out)
    n, bs = size(x)
    @inbounds for j in 1:bs
        s = zero(T)
        ap = αp[1, j]; ad = αd[1, j]
        for i in 1:n
            s += (xu[i,j] - (x[i,j] + ap * dx[i,j])) * (z[i,j] + ad * dz[i,j])
        end
        out[1, j] = s
    end
end

function set_tau!(rule::ConservativeStep, batch_solver::AbstractBatchMPCSolver)
    fill!(batch_solver.workspace.tau, rule.tau)
end
function set_tau!(rule::AdaptiveStep, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    ws.tau .= max.(1 .- ws.mu_batch, rule.tau_min)
end
function update_step!(rule::Union{ConservativeStep, AdaptiveStep}, batch_solver::AbstractBatchMPCSolver)
    set_tau!(rule, batch_solver)
    get_fraction_to_boundary_step!(batch_solver)
    return
end

function _mehrotra_step!(
    alpha_p, alpha_d, mu, gamma_f,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off, dub_off,
)
    for j in axes(alpha_p, 2)
        _mehrotra_step_column!(
            j, alpha_p, alpha_d, mu[1, j], gamma_f,
            dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
            dx_ur, x_ur, xu_r, nub, dzub, zu_r,
            d_vals, ind_lb, ind_ub, dlb_off, dub_off,
        )
    end
end

@inline function _mehrotra_step_column!(
    j, alpha_p, alpha_d, mu_j::T, gamma_f::T,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off, dub_off,
) where T
    max_ap = alpha_p[1, j]
    max_ad = alpha_d[1, j]

    # primal lb
    best_xl = T(Inf); i_xl = 0
    @inbounds for i in 1:nlb
        d = dx_lr[i, j]
        d < zero(T) || continue
        v = (xl_r[i, j] - x_lr[i, j]) / d
        v < best_xl && (best_xl = v; i_xl = i)
    end
    # primal ub
    best_xu = T(Inf); i_xu = 0
    @inbounds for i in 1:nub
        d = dx_ur[i, j]
        d > zero(T) || continue
        v = (xu_r[i, j] - x_ur[i, j]) / d
        v < best_xu && (best_xu = v; i_xu = i)
    end
    # dual lb
    best_zl = T(Inf); i_zl = 0
    @inbounds for i in 1:nlb
        d = dzlb[i, j]
        d < zero(T) || continue
        v = -zl_r[i, j] / d
        v < best_zl && (best_zl = v; i_zl = i)
    end
    # dual ub
    best_zu = T(Inf); i_zu = 0
    @inbounds for i in 1:nub
        d = dzub[i, j]
        (d < zero(T) && zu_r[i, j] + d < zero(T)) || continue
        v = -zu_r[i, j] / d
        v < best_zu && (best_zu = v; i_zu = i)
    end

    # primal step
    corrected_p = one(T)
    @inbounds if max_ap < one(T)
        if best_xl <= best_xu && i_xl > 0
            idx = ind_lb[i_xl]
            zl_stepped = zl_r[i_xl, j] + max_ad * d_vals[dlb_off + i_xl, j]
            corrected_p = (x_lr[i_xl, j] - xl_r[i_xl, j] - mu_j / zl_stepped) / (-dx_lr[i_xl, j])
        elseif i_xu > 0
            idx = ind_ub[i_xu]
            zu_stepped = zu_r[i_xu, j] + max_ad * d_vals[dub_off + i_xu, j]
            corrected_p = (xu_r[i_xu, j] - x_ur[i_xu, j] - mu_j / zu_stepped) / dx_ur[i_xu, j]
        end
    end
    alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

    # dual step
    corrected_d = one(T)
    @inbounds if max_ad < one(T)
        if best_zl <= best_zu && i_zl > 0
            idx = ind_lb[i_zl]
            x_gap = x_lr[i_zl, j] + max_ap * dx_lr[i_zl, j] - xl_r[i_zl, j]
            corrected_d = -(zl_r[i_zl, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
        elseif i_zu > 0
            idx = ind_ub[i_zu]
            x_gap = xu_r[i_zu, j] - x_ur[i_zu, j] - max_ap * dx_ur[i_zu, j]
            corrected_d = -(zu_r[i_zu, j] - mu_j / x_gap) / d_vals[dub_off + i_zu, j]
        end
    end
    alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
    return
end

function update_step!(rule::MehrotraAdaptiveStep, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    nlb, nub = d.nlb, d.nub
    T = eltype(ws.alpha_p)
    gamma_f = T(rule.gamma_f)
    gamma_a = one(T) / (one(T) - gamma_f)

    fill!(ws.tau, one(T))
    get_fraction_to_boundary_step!(batch_solver)

    get_affine_complementarity_measure!(batch_solver, ws.alpha_p, ws.alpha_d)
    mu_full = ws.mu_curr
    @. mu_full = ws.mu_affine / gamma_a

    dlb_off = d.n + d.m
    dub_off = d.n + d.m + d.nlb
    bs = batch_solver.batch_size

    _mehrotra_step!(
        ws.alpha_p, ws.alpha_d, mu_full, gamma_f,
        xp_lr(d), lower(x), lower(xl), nlb, MadNLP.dual_lb(d), lower(zl),
        xp_ur(d), upper(x), upper(xu), nub, MadNLP.dual_ub(d), upper(zu),
        d.values, d.ind_lb, d.ind_ub, dlb_off, dub_off,
    )
    return
end

# FIXME: make it a kernel
function _adjust_boundary_active!(x_lr::AbstractMatrix{T}, xl_r, x_ur, xu_r, mu, mask) where {T}
    c2 = eps(T)^(T(3)/T(4))
    c1 = eps(T)
    xl_r .= ifelse.(
        (mask .!= 0) .& (x_lr .- xl_r .< (c1 .* mu)),
        xl_r .- c2 .* max.(one(T), abs.(x_lr)),
        xl_r,
    )
    xu_r .= ifelse.(
        (mask .!= 0) .& (xu_r .- x_ur .< (c1 .* mu)),
        xu_r .+ c2 .* max.(one(T), abs.(x_ur)),
        xu_r,
    )
end

function init_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, 0.0)
end
update_regularization!(solver::AbstractBatchMPCSolver, reg) =
    update_regularization!(solver, reg, solver.workspace.active_mask)
function update_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization, mask)
    solver.del_w .= ifelse.(mask .== 1, 0.0, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, 0.0, solver.del_c)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization, mask)
    solver.del_w .= ifelse.(mask .== 1, reg.delta_p, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, reg.delta_d, solver.del_c)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization, mask)
    reg.delta_p = max(reg.delta_p / 10.0, reg.delta_min)
    reg.delta_d = min(reg.delta_d / 10.0, -reg.delta_min)
    solver.del_w .= ifelse.(mask .== 1, reg.delta_p, solver.del_w)
    solver.del_c .= ifelse.(mask .== 1, reg.delta_d, solver.del_c)
end
