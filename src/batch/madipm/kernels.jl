function dual_objective!(dual_obj, y_vals, rhs_vals, zl_r, xl_r, zu_r, xu_r,
                         scratch_m, scratch_lb, scratch_ub, sum_lb, sum_ub, nlb, nub)
    @. scratch_m = y_vals * rhs_vals
    sum!(dual_obj, scratch_m)
    dual_obj .*= -one(eltype(dual_obj))
    if nlb > 0
        @. scratch_lb = zl_r * xl_r
        sum!(sum_lb, scratch_lb)
        dual_obj .+= sum_lb
    end
    if nub > 0
        @. scratch_ub = zu_r * xu_r
        sum!(sum_ub, scratch_ub)
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

function set_aug_diagonal_reg!(kkt, solver::AbstractBatchMPCSolver)
    xl_r = lower(solver.xl)
    x_lr = lower(solver.x)
    zl_r = lower(solver.zl)
    xu_r = upper(solver.xu)
    x_ur = upper(solver.x)
    zu_r = upper(solver.zu)

    kkt.reg .= solver.del_w
    du_diag(kkt) .= solver.del_c

    kkt.l_diag .= xl_r .- x_lr
    kkt.u_diag .= x_ur .- xu_r

    kkt.l_lower .= zl_r
    kkt.u_lower .= zu_r

    pr_diag(kkt) .= kkt.reg
    pr_diag_lb = view(kkt.nzVals, _get_ind_lb(solver), :)
    pr_diag_ub = view(kkt.nzVals, _get_ind_ub(solver), :)
    pr_diag_lb .-= kkt.l_lower ./ kkt.l_diag
    pr_diag_ub .-= kkt.u_lower ./ kkt.u_diag
    return
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

    ws.sum_lb .= mapreduce(
        (x, xl, z) -> (x - xl) * z, +, x_lr, xl_r, zl_r;
        dims=1, init=zero(T),
    )
    ws.sum_ub .= mapreduce(
        (xu, x, z) -> (xu - x) * z, +, xu_r, x_ur, zu_r;
        dims=1, init=zero(T),
    )
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

    _scratch_lb = MadNLP.dual_lb(solver._w2)
    @. _scratch_lb = (x_lr + alpha_p * dx_lr - xl_r) * (zl_r + alpha_d * dzlb)
    sum!(ws.sum_lb, _scratch_lb)

    _scratch_ub = MadNLP.dual_ub(solver._w2)
    @. _scratch_ub = (xu_r - (x_ur + alpha_p * dx_ur)) * (zu_r + alpha_d * dzub)
    sum!(ws.sum_ub, _scratch_ub)

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

    # can't use mapreduce since tau is (1, bs), not (nlb, bs)
    if nlb > 0
        _dx_lr = xp_lr(d); _xl_r = lower(xl); _x_lr = lower(x)  # (nlb, bs)
        _dzlb = MadNLP.dual_lb(d); _zl_r = lower(zl)              # (nlb, bs)
        _scratch_lb = MadNLP.dual_lb(batch_solver._w2)             # (nlb, bs)

        @. _scratch_lb = ifelse(_dx_lr < 0, (-_x_lr + _xl_r) * ws.tau / _dx_lr, T(Inf))
        minimum!(ws.alpha_xl, _scratch_lb)

        @. _scratch_lb = ifelse(_dzlb < 0, (-_zl_r) * ws.tau / _dzlb, T(Inf))
        minimum!(ws.alpha_zl, _scratch_lb)
    else
        fill!(ws.alpha_xl, one(T))
        fill!(ws.alpha_zl, one(T))
    end

    if nub > 0
        _dx_ur = xp_ur(d); _xu_r = upper(xu); _x_ur = upper(x)
        _dzub = MadNLP.dual_ub(d); _zu_r = upper(zu)
        _scratch_ub = MadNLP.dual_ub(batch_solver._w2)

        @. _scratch_ub = ifelse(_dx_ur > 0, (-_x_ur + _xu_r) * ws.tau / _dx_ur, T(Inf))
        minimum!(ws.alpha_xu, _scratch_ub)

        @. _scratch_ub = ifelse((_dzub < 0) & (_zu_r + _dzub < 0), (-_zu_r) * ws.tau / _dzub, T(Inf))
        minimum!(ws.alpha_zu, _scratch_ub)
    else
        fill!(ws.alpha_xu, one(T))
        fill!(ws.alpha_zu, one(T))
    end

    ws.alpha_p .= min.(ws.alpha_xl, ws.alpha_xu, one(T))
    ws.alpha_d .= min.(ws.alpha_zl, ws.alpha_zu, one(T))
    return
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

function _mehrotra_correct_steps!(
    alpha_p, alpha_d, mu,
    val_xl, idx_xl, val_xu, idx_xu,
    val_zl, idx_zl, val_zu, idx_zu,
    d_vals, x_vals, xl_vals, xu_vals, zl_vals, zu_vals,
    ind_lb, ind_ub, dlb_off::Int, dub_off::Int, gamma_f,
)
    T = eltype(alpha_p)
    @inbounds for j in axes(alpha_p, 2)
        mu_j = mu[1, j]
        max_ap = alpha_p[1, j]
        max_ad = alpha_d[1, j]

        # primal step
        corrected_p = one(T)
        if max_ap < one(T)
            i_xl = idx_xl[1, j]
            i_xu = idx_xu[1, j]
            if val_xl[1, j] <= val_xu[1, j] && i_xl > 0
                idx = ind_lb[i_xl]
                zl_stepped = zl_vals[idx, j] + max_ad * d_vals[dlb_off + i_xl, j]
                corrected_p = (x_vals[idx, j] - xl_vals[idx, j] - mu_j / zl_stepped) / (-d_vals[idx, j])
            elseif i_xu > 0
                idx = ind_ub[i_xu]
                zu_stepped = zu_vals[idx, j] + max_ad * d_vals[dub_off + i_xu, j]
                corrected_p = (xu_vals[idx, j] - x_vals[idx, j] - mu_j / zu_stepped) / d_vals[idx, j]
            end
        end
        alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

        # dual step
        corrected_d = one(T)
        if max_ad < one(T)
            i_zl = idx_zl[1, j]
            i_zu = idx_zu[1, j]
            if val_zl[1, j] <= val_zu[1, j] && i_zl > 0
                idx = ind_lb[i_zl]
                x_gap = x_vals[idx, j] + max_ap * d_vals[idx, j] - xl_vals[idx, j]
                corrected_d = -(zl_vals[idx, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
            elseif i_zu > 0
                idx = ind_ub[i_zu]
                x_gap = xu_vals[idx, j] - x_vals[idx, j] - max_ap * d_vals[idx, j]
                corrected_d = -(zu_vals[idx, j] - mu_j / x_gap) / d_vals[dub_off + i_zu, j]
            end
        end
        alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
    end
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

    if nlb > 0
        _scratch_lb = MadNLP.dual_lb(batch_solver._w2)
        _dx_lr = xp_lr(d); _xl_r = lower(xl); _x_lr = lower(x)
        _dzlb = MadNLP.dual_lb(d); _zl_r = lower(zl)

        map!((dx, xl, x) -> dx < 0 ? (xl - x) / dx : T(Inf), _scratch_lb, _dx_lr, _xl_r, _x_lr)
        _vals, _inds = findmin(_scratch_lb; dims=1)
        copyto!(ws.alpha_xl, _vals)
        ws.idx_xl .= getindex.(_inds, 1)

        map!((dz, z) -> dz < 0 ? -z / dz : T(Inf), _scratch_lb, _dzlb, _zl_r)
        _vals, _inds = findmin(_scratch_lb; dims=1)
        copyto!(ws.alpha_zl, _vals)
        ws.idx_zl .= getindex.(_inds, 1)
    else
        fill!(ws.alpha_xl, one(T)); fill!(ws.idx_xl, 0)
        fill!(ws.alpha_zl, one(T)); fill!(ws.idx_zl, 0)
    end

    if nub > 0
        _scratch_ub = MadNLP.dual_ub(batch_solver._w2)
        _dx_ur = xp_ur(d); _xu_r = upper(xu); _x_ur = upper(x)
        _dzub = MadNLP.dual_ub(d); _zu_r = upper(zu)

        map!((dx, xu, x) -> dx > 0 ? (xu - x) / dx : T(Inf), _scratch_ub, _dx_ur, _xu_r, _x_ur)
        _vals, _inds = findmin(_scratch_ub; dims=1)
        copyto!(ws.alpha_xu, _vals)
        ws.idx_xu .= getindex.(_inds, 1)

        map!((dz, z) -> (dz < 0) & (z + dz < 0) ? -z / dz : T(Inf), _scratch_ub, _dzub, _zu_r)
        _vals, _inds = findmin(_scratch_ub; dims=1)
        copyto!(ws.alpha_zu, _vals)
        ws.idx_zu .= getindex.(_inds, 1)
    else
        fill!(ws.alpha_xu, one(T)); fill!(ws.idx_xu, 0)
        fill!(ws.alpha_zu, one(T)); fill!(ws.idx_zu, 0)
    end

    _mehrotra_correct_steps!(
        ws.alpha_p, ws.alpha_d, mu_full,
        ws.alpha_xl, ws.idx_xl, ws.alpha_xu, ws.idx_xu,
        ws.alpha_zl, ws.idx_zl, ws.alpha_zu, ws.idx_zu,
        d.values, x.values, xl.values, xu.values, zl.values, zu.values,
        d.ind_lb, d.ind_ub, dlb_off, dub_off, gamma_f,
    )

    return
end

function init_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, 0.0)
end
function update_regularization!(solver::AbstractBatchMPCSolver, ::NoRegularization)
    fill!(solver.del_w, 0.0)
    fill!(solver.del_c, 0.0)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::FixedRegularization)
    fill!(solver.del_w, reg.delta_p)
    fill!(solver.del_c, reg.delta_d)
end
function init_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization)
    fill!(solver.del_w, 1.0)
    fill!(solver.del_c, reg.delta_d)
end
function update_regularization!(solver::AbstractBatchMPCSolver, reg::AdaptiveRegularization)
    reg.delta_p = max(reg.delta_p / 10.0, reg.delta_min)
    reg.delta_d = min(reg.delta_d / 10.0, -reg.delta_min)
    fill!(solver.del_w, reg.delta_p)
    fill!(solver.del_c, reg.delta_d)
end
