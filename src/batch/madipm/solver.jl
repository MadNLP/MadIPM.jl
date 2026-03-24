function init_starting_point!(batch_solver::AbstractBatchMPCSolver{T}) where T
    bkkt = batch_solver.kkt
    bs = batch_solver.batch_size
    n = batch_solver.d.n
    m = batch_solver.d.m

    bx, bxl, bxu = batch_solver.x, batch_solver.xl, batch_solver.xu
    bzl, bzu = batch_solver.zl, batch_solver.zu
    x = MadNLP.primal(bx)
    l, u = MadNLP.full(bxl), MadNLP.full(bxu)
    lb, ub = lower(bxl), upper(bxu)
    zl, zu = lower(bzl), upper(bzu)
    xl, xu = lower(bx), upper(bx)
    # use jacl as a buffer
    res = MadNLP.full(batch_solver.jacl)

    # Add initial primal-dual regularization
    bkkt.reg .= batch_solver.del_w
    pr_diag(bkkt) .= batch_solver.del_w
    du_diag(bkkt) .= batch_solver.del_c

    # Step 0: factorize initial KKT system
    MadNLP.factorize_wrapper!(batch_solver)

    # Step 1: Compute initial primal variable as x0 = x + dx, with dx the
    #         least square solution of the system A * dx = (b - A*x)
    set_initial_primal_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    # x0 = x + dx
    x .+= MadNLP.primal(batch_solver.d)

    # Step 2: Compute initial dual variable as the least square solution of A' * y = -f
    set_initial_dual_rhs!(batch_solver)
    solve_system!(batch_solver.d, batch_solver, batch_solver.p)
    MadNLP.full(batch_solver.y) .= MadNLP.dual(batch_solver.d)

    # Step 3: init bounds multipliers using c + A' * y - zl + zu = 0
    # A' * y
    MadNLP.jtprod!(res, bkkt, batch_solver.y)
    # A'*y + c
    res .+= MadNLP.primal(batch_solver.f)
    # Initialize bounds multipliers
    map!(
        (r_, l_, u_, zl_) -> begin
            val = if isfinite(l_) && isfinite(u_)
                0.5 * r_
            elseif isfinite(l_)
                r_
            else
                zl_
            end
            val
        end,
        MadNLP.full(batch_solver.zl), res, l, u, MadNLP.full(batch_solver.zl),
    )
    map!(
        (r_, l_, u_, zu_) -> begin
            val = if isfinite(l_) && isfinite(u_)
                -0.5 * r_
            elseif isfinite(u_)
                -r_
            else
                zu_
            end
            val
        end,
        MadNLP.full(batch_solver.zu), res, l, u, MadNLP.full(batch_solver.zu),
    )

    ws = batch_solver.workspace
    nlb_init, nub_init = batch_solver.d.nlb, batch_solver.d.nub
    _s1 = ws.alpha_xl  # (1,bs) scratch
    _s2 = ws.alpha_xu  # (1,bs) scratch

    # delta_x = max(0, -1.5 * min(xl-lb, 0), -1.5 * min(ub-xu, 0))
    if nlb_init > 0
        _scratch_lb = MadNLP.dual_lb(batch_solver._w2)
        @. _scratch_lb = xl - lb
        minimum!(_s1, _scratch_lb)
        @. _s1 = min(_s1, zero(T))  # clamp: init=0.0 behavior
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        _scratch_ub = MadNLP.dual_ub(batch_solver._w2)
        @. _scratch_ub = ub - xu
        minimum!(_s2, _scratch_ub)
        @. _s2 = min(_s2, zero(T))
    else
        fill!(_s2, zero(T))
    end
    delta_x = ws.mu_batch  # (1,bs) scratch
    @. delta_x = max(zero(T), T(-1.5) * _s1, T(-1.5) * _s2)

    # delta_s = max(0, -1.5 * min(zl, 0), -1.5 * min(zu, 0))
    if nlb_init > 0
        minimum!(_s1, zl)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        minimum!(_s2, zu)
        @. _s2 = min(_s2, zero(T))
    else
        fill!(_s2, zero(T))
    end
    delta_s = ws.mu_curr  # (1,bs) scratch
    @. delta_s = max(zero(T), T(-1.5) * _s1, T(-1.5) * _s2)

    xl .+= delta_x
    xu .-= delta_x
    zl .+= 1.0 .+ delta_s
    zu .+= 1.0 .+ delta_s

    # μ = sum((xl-lb)*zl) + sum((ub-xu)*zu)
    μ = ws.mu_affine  # (1,bs) scratch
    fill!(μ, zero(T))
    if nlb_init > 0
        _scratch_lb = MadNLP.dual_lb(batch_solver._w2)
        @. _scratch_lb = xl * zl
        sum!(ws.sum_lb, _scratch_lb)
        μ .+= ws.sum_lb
        @. _scratch_lb = lb * zl
        sum!(ws.sum_lb, _scratch_lb)
        μ .-= ws.sum_lb
    end
    if nub_init > 0
        _scratch_ub = MadNLP.dual_ub(batch_solver._w2)
        @. _scratch_ub = ub * zu
        sum!(ws.sum_ub, _scratch_ub)
        @. _scratch_ub = xu * zu
        sum!(ws.sum_lb, _scratch_ub)
        ws.sum_ub .-= ws.sum_lb
        μ .+= ws.sum_ub
    end

    # delta_x2 = μ / (2 * (sum(zl) + sum(zu)))
    if nlb_init > 0
        sum!(ws.sum_lb, zl)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        sum!(ws.sum_ub, zu)
    else
        fill!(ws.sum_ub, zero(T))
    end
    delta_x2 = _s1  # reuse (1,bs) scratch
    @. delta_x2 = μ / (2 * (ws.sum_lb + ws.sum_ub))

    # delta_s2 = μ / (2 * (sum(xl-lb) + sum(ub-xu)))
    if nlb_init > 0
        _scratch_lb = MadNLP.dual_lb(batch_solver._w2)
        @. _scratch_lb = xl - lb
        sum!(ws.sum_lb, _scratch_lb)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        _scratch_ub = MadNLP.dual_ub(batch_solver._w2)
        @. _scratch_ub = ub - xu
        sum!(ws.sum_ub, _scratch_ub)
    else
        fill!(ws.sum_ub, zero(T))
    end
    delta_s2 = _s2  # reuse (1,bs) scratch
    @. delta_s2 = μ / (2 * (ws.sum_lb + ws.sum_ub))

    xl .+= delta_x2
    xu .-= delta_x2
    zl .+= delta_s2
    zu .+= delta_s2

    # Use Ipopt's heuristic to project x back on the interval [l, u]
    kappa = batch_solver.opt.bound_fac
    map!(
        (l_, u_, x_) -> begin
            out = if x_ < l_
                pl = min(kappa * max(1.0, l_), kappa * (u_ - l_))
                l_ + pl
            elseif u_ < x_
                pu = min(kappa * max(1.0, u_), kappa * (u_ - l_))
                u_ - pu
            else
                x_
            end
            out
        end,
        x, l, u, x,
    )
    return
end

function initialize!(batch_solver::AbstractBatchMPCSolver{T}) where T
    opt = batch_solver.opt
    bcb = batch_solver.bcb
    ws = batch_solver.workspace

    MadNLP.initialize!(
        bcb,
        batch_solver.x,
        batch_solver.xl,
        batch_solver.xu,
        MadNLP.full(batch_solver.y),
        MadNLP.full(batch_solver.rhs),
        bcb.ind_ineq,
        ws.bx;
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )
    fill!(MadNLP.full(batch_solver.jacl), zero(T))

    if opt.scaling
        MadNLP.set_scaling!(
            bcb,
            batch_solver.x,
            batch_solver.xl,
            batch_solver.xu,
            MadNLP.full(batch_solver.y),
            MadNLP.full(batch_solver.rhs),
            bcb.ind_ineq,
            T(opt.nlp_scaling_max_gradient),
            ws.bx,
        )
    end

    MadNLP.initialize!(batch_solver.kkt)
    init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)

    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    init_starting_point!(batch_solver)

    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(T))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, zero(T))
    fill!(ws.inf_du, zero(T))
    fill!(ws.inf_compl, zero(T))
    fill!(ws.dual_obj, zero(T))
    fill!(ws.alpha_p, zero(T))
    fill!(ws.alpha_d, zero(T))
    t_now = time()
    batch_solver.batch_cnt.start_time[] = t_now
    fill!(batch_solver.batch_cnt.k, 0)
    batch_solver.batch_cnt.linear_solver_time[] = 0.0
    batch_solver.batch_cnt.eval_function_time[] = 0.0
    batch_solver.batch_cnt.obj_cnt[] = 0
    batch_solver.batch_cnt.obj_grad_cnt[] = 0
    batch_solver.batch_cnt.con_cnt[] = 0

    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)

    return
end

function update_termination_criteria!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    opt = batch_solver.opt
    bcnt = batch_solver.batch_cnt
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu = batch_solver.zl, batch_solver.zu
    bs = batch_solver.batch_size
    nlb, nub = batch_solver.d.nlb, batch_solver.d.nub

    _scratch_n = MadNLP.primal(batch_solver._w2)
    _scratch_m = MadNLP.dual(batch_solver._w2)
    _scratch_lb = MadNLP.dual_lb(batch_solver._w2)
    _scratch_ub = MadNLP.dual_ub(batch_solver._w2)

    get_inf_pr!(ws.inf_pr, MadNLP.full(batch_solver.c), _scratch_m)
    @. ws.inf_pr /= max(one(T), ws.norm_b)

    get_inf_du!(ws.inf_du, MadNLP.full(batch_solver.f), MadNLP.full(zl),
                MadNLP.full(zu), MadNLP.full(batch_solver.jacl), _scratch_n)
    @. ws.inf_du /= max(one(T), ws.norm_c)

    get_inf_compl!(ws.inf_compl, x, xl, zl, xu, zu,
        _scratch_lb, _scratch_ub, ws.sum_lb, ws.sum_ub, nlb, nub)
    @. ws.inf_compl /= max(one(T), ws.norm_c)
    @. ws.best_complementarity = min(ws.best_complementarity, ws.inf_compl)

    dual_objective!(ws.dual_obj, MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs),
        lower(zl), lower(xl), upper(zu), upper(xu),
        _scratch_m, _scratch_lb, _scratch_ub, ws.sum_lb, ws.sum_ub, nlb, nub)

    ds = T(opt.divergence_scale)
    tol = T(opt.tol)
    div_tol = T(opt.divergence_tol)
    Int_SOLVED = Int(MadNLP.SOLVE_SUCCEEDED)
    Int_INFEASIBLE = Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED)
    Int_DIVERGING = Int(MadNLP.DIVERGING_ITERATES)
    Int_REGULAR = Int(MadNLP.REGULAR)
    @. ws._term_gpu = ifelse(
        max(ws.inf_pr, ws.inf_du, ws.inf_compl) <= tol,
        Int_SOLVED,
        ifelse(
            (ws.inf_compl > div_tol * ws.best_complementarity) &
            (ws.dual_obj > max(ds * abs(ws.obj_val), one(T))),
            Int_INFEASIBLE,
            ifelse(
                ws.obj_val < -(div_tol * max(ds, abs(ws.dual_obj), one(T))),
                Int_DIVERGING,
                Int_REGULAR,
            ),
        ),
    )
    copyto!(ws._term_cpu, vec(ws._term_gpu))

    walltime_hit = time() - bcnt.start_time[] >= opt.max_wall_time
    @inbounds for i in 1:bs
        ws.status[i] != MadNLP.REGULAR && continue
        code = MadNLP.Status(ws._term_cpu[i])
        if code != MadNLP.REGULAR
            ws.status[i] = code
        elseif bcnt.k[i] >= opt.max_iter
            ws.status[i] = MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        elseif walltime_hit
            ws.status[i] = MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        end
    end
    return
end

function solve_system!(
    d::BatchUnreducedKKTVector{T},
    batch_solver::AbstractBatchMPCSolver{T},
    p::BatchUnreducedKKTVector{T},
) where T
    opt = batch_solver.opt
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(batch_solver.kkt, batch_solver)

    w = batch_solver._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, batch_solver.kkt, d, -one(T), one(T))

    bkkt = batch_solver.kkt
    bs = bkkt.batch_size
    @inbounds for i in 1:bs
        if bkkt.batch_map[i] == 0
            view(MadNLP.full(w), :, i) .= zero(T)
            view(MadNLP.full(p), :, i) .= zero(T)
        end
    end

    ws = batch_solver.workspace
    _fw = MadNLP.full(w)
    _fw .= abs.(_fw)
    maximum!(ws._norm_gpu, _fw)                          # (1,bs) per-instance norm_w
    copyto!(ws._norm_cpu, vec(ws._norm_gpu))
    _fw .= abs.(MadNLP.full(p))
    maximum!(ws._norm_gpu, _fw)                          # (1,bs) per-instance norm_p
    copyto!(ws._norm_cpu2, vec(ws._norm_gpu))

    @inbounds for i in 1:bs
        bkkt.batch_map[i] == 0 && continue
        nw = ws._norm_cpu[i]
        np = ws._norm_cpu2[i]
        ratio = nw / max(one(T), np)
        if isnan(ratio) || (opt.check_residual && (ratio > opt.tol_linear_solve))
            ws.status[i] = MadNLP.INTERNAL_ERROR
        end
    end
    return d
end

function increment_k!(batch_solver::AbstractBatchMPCSolver)
    bcnt = batch_solver.batch_cnt
    ws = batch_solver.workspace
    for i in 1:batch_solver.batch_size
        if ws.status[i] == MadNLP.REGULAR
            bcnt.k[i] += 1
        end
    end
end

function update_solution!(stats::BatchExecutionStats, batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    x, zl, zu = batch_solver.x, batch_solver.zl, batch_solver.zu

    stats.status .= ws.status
    stats.iter .= batch_solver.batch_cnt.k

    MadNLP.unpack_x!(stats.solution, bcb, x)
    MadNLP.unpack_y!(stats.multipliers, bcb, MadNLP.full(batch_solver.y))
    MadNLP.unpack_z!(stats.multipliers_L, bcb, MadNLP.variable(zl))
    MadNLP.unpack_z!(stats.multipliers_U, bcb, MadNLP.variable(zu))
    stats.objective .= MadNLP.unpack_obj(bcb, ws.obj_val)
    MadNLP.unpack_cons!(stats.constraints, bcb, MadNLP.full(batch_solver.c), MadNLP.full(batch_solver.rhs), bcb.ind_ineq, MadNLP.slack(x))

    stats.dual_feas .= vec(ws.inf_du)
    stats.primal_feas .= vec(ws.inf_pr)
    stats.total_time .= batch_solver.batch_cnt.total_time
    return stats
end

function affine_direction!(solver::AbstractBatchMPCSolver)
    set_predictive_rhs!(solver, solver.kkt)
    solve_system!(solver.d, solver, solver.p)
end

function prediction_step!(solver::AbstractBatchMPCSolver)
    ws = solver.workspace
    affine_direction!(solver)

    fill!(ws.tau, one(eltype(ws.tau)))
    get_fraction_to_boundary_step!(solver)
    zero_inactive_step!(solver)
    get_affine_complementarity_measure!(solver, ws.alpha_p, ws.alpha_d)
    get_correction!(solver, MadNLP.full(solver.correction_lb), MadNLP.full(solver.correction_ub))
    update_barrier!(solver.opt.barrier_update, solver, ws.mu_affine)
    return
end

function mehrotra_correction_direction!(solver::AbstractBatchMPCSolver)
    set_correction_rhs!(solver, solver.kkt, solver.workspace.mu_batch, MadNLP.full(solver.correction_lb), MadNLP.full(solver.correction_ub), nothing, nothing)
    solve_system!(solver.d, solver, solver.p)
    return
end

function factorize_system!(batch_solver::AbstractBatchMPCSolver)
    update_regularization!(batch_solver, batch_solver.opt.regularization)
    max_trials = 3
    for _ in 1:max_trials
        set_aug_diagonal_reg!(batch_solver.kkt, batch_solver)
        MadNLP.factorize_wrapper!(batch_solver)
        is_factorized(batch_solver.kkt.batch_solver) && break  # exit once all are factorized
        batch_solver.del_w .*= 100.0
        batch_solver.del_c .*= 100.0
    end
    return
end


function apply_step!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    x, y, xl, xu = batch_solver.x, batch_solver.y, batch_solver.xl, batch_solver.xu
    zl, zu, d = batch_solver.zl, batch_solver.zu, batch_solver.d
    batch_size = batch_solver.batch_size
    nlb, nub = d.nlb, d.nub

    # x += alpha_p * dx
    MadNLP.full(x) .+= ws.alpha_p .* MadNLP.primal(d)

    # y += alpha_d * d_dual
    MadNLP.full(y) .+= ws.alpha_d .* MadNLP.dual(d)

    # zl_r += alpha_d * dzl, zu_r += alpha_d * dzu
    if nlb > 0
        lower(zl) .+= ws.alpha_d .* MadNLP.dual_lb(d)
    end
    if nub > 0
        upper(zu) .+= ws.alpha_d .* MadNLP.dual_ub(d)
    end

    MadNLP.adjust_boundary!(lower(x), lower(xl), upper(x), upper(xu), ws.mu_batch)
    increment_k!(batch_solver)
    return
end

function evaluate_model!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return
end

function mpc_step!(batch_solver::AbstractBatchMPCSolver)
    factorize_system!(batch_solver)
    prediction_step!(batch_solver)
    mehrotra_correction_direction!(batch_solver)
    update_step!(batch_solver.opt.step_rule, batch_solver)
    zero_inactive_step!(batch_solver)
    apply_step!(batch_solver)
    evaluate_model!(batch_solver)
end

function _update_active_mask!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    bmap = batch_solver.kkt.batch_map
    copyto!(ws.active_mask, reshape(T.(bmap .!= 0), 1, :))
end

function mpc!(batch_solver::AbstractBatchMPCSolver)
    while true
        MadNLP.print_iter(batch_solver)
        update_termination_criteria!(batch_solver)
        update_active_set!(batch_solver.kkt, batch_solver.workspace.status)
        batch_solver.kkt.active_batch_size[] == 0 && return
        _update_active_mask!(batch_solver)
        mpc_step!(batch_solver)
    end
end

function solve!(batch_solver::AbstractBatchMPCSolver{T, MT, VT}) where {T, MT, VT}
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    bs = batch_solver.batch_size

    nvar_nlp = bcb.nlp.meta.nvar
    ncon = bcb.ncon
    stats = BatchExecutionStats(MT, VT, nvar_nlp, ncon, bs)

    try
        MadNLP.@notice(batch_solver.logger, "MadIPM batch solve ($bs problems)\n")
        initialize!(batch_solver)
        mpc!(batch_solver)
    catch e
        for i in 1:bs
            if ws.status[i] == MadNLP.REGULAR
                ws.status[i] = MadNLP.INTERNAL_ERROR
            end
        end
        batch_solver.opt.rethrow_error && rethrow(e)
    finally
        bcnt = batch_solver.batch_cnt
        t_end = time()
        bcnt.total_time .= t_end .- bcnt.start_time[]
        update_solution!(stats, batch_solver)
        status_counts = Dict{MadNLP.Status, Int}()
        for i in 1:bs
            s = ws.status[i]
            status_counts[s] = get(status_counts, s, 0) + 1
        end
        for (s, cnt) in status_counts
            MadNLP.@notice(batch_solver.logger, "$(MadNLP.get_status_output(s, batch_solver.opt)): $cnt/$bs")
        end
    end

    return stats
end

function madipm_batch(bnlp::NLPModels.AbstractBatchNLPModel; kwargs...)
    batch_solver = UniformBatchMPCSolver(bnlp; kwargs...)
    return solve!(batch_solver)
end

function IPMOptions(
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    kkt_system = MadNLP.SparseKKTSystem,
    linear_solver = MadNLP.LDLSolver,
    tol = T(1e-8),
) where T
    return IPMOptions(
        tol = tol,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
    )
end

function MadNLP.print_iter(batch_solver::AbstractBatchMPCSolver)
    logger = batch_solver.logger
    MadNLP.get_level(logger) > MadNLP.INFO && return
    ws = batch_solver.workspace
    bcnt = batch_solver.batch_cnt
    na = batch_solver.kkt.active_batch_size[]
    bs = batch_solver.batch_size
    k = maximum(bcnt.k)

    mod(k, 10) == 0 && MadNLP.@info(logger, @sprintf(
        " iter  active  max_inf_pr  max_inf_du  max_inf_compl  max_alpha_p"))
    MadNLP.@info(logger, @sprintf(
        "%4i   %3i/%3i   %6.2e     %6.2e      %7.2e      %6.2e",
        k, na, bs,
        maximum(ws.inf_pr), maximum(ws.inf_du),
        maximum(ws.inf_compl), maximum(ws.alpha_p),
    ))
    return
end
