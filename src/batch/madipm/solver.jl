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
    bs = batch_solver.batch_size
    _s1 = ws.alpha_xl  # (1,bs) scratch
    _s2 = ws.alpha_xu  # (1,bs) scratch

    # delta_x = max(0, -1.5 * min(xl-lb, 0), -1.5 * min(ub-xu, 0))
    if nlb_init > 0
        batch_mapreduce!(-, min, T(Inf), _s1, xl, lb)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(-, min, T(Inf), _s2, ub, xu)
        @. _s2 = min(_s2, zero(T))
    else
        fill!(_s2, zero(T))
    end
    delta_x = ws.mu_batch  # (1,bs) scratch
    @. delta_x = max(zero(T), T(-1.5) * _s1, T(-1.5) * _s2)

    # delta_s = max(0, -1.5 * min(zl, 0), -1.5 * min(zu, 0))
    if nlb_init > 0
        batch_mapreduce!(identity, min, T(Inf), _s1, zl)
        @. _s1 = min(_s1, zero(T))
    else
        fill!(_s1, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(identity, min, T(Inf), _s2, zu)
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
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, xl, zl)
        μ .+= ws.sum_lb
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, lb, zl)
        μ .-= ws.sum_lb
    end
    if nub_init > 0
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_ub, ub, zu)
        batch_mapreduce!((a, b) -> a * b, +, zero(T), ws.sum_lb, xu, zu)
        ws.sum_ub .-= ws.sum_lb
        μ .+= ws.sum_ub
    end

    # delta_x2 = μ / (2 * (sum(zl) + sum(zu)))
    if nlb_init > 0
        batch_mapreduce!(identity, +, zero(T), ws.sum_lb, zl)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(identity, +, zero(T), ws.sum_ub, zu)
    else
        fill!(ws.sum_ub, zero(T))
    end
    delta_x2 = _s1  # reuse (1,bs) scratch
    @. delta_x2 = μ / (2 * (ws.sum_lb + ws.sum_ub))

    # delta_s2 = μ / (2 * (sum(xl-lb) + sum(ub-xu)))
    if nlb_init > 0
        batch_mapreduce!(-, +, zero(T), ws.sum_lb, xl, lb)
    else
        fill!(ws.sum_lb, zero(T))
    end
    if nub_init > 0
        batch_mapreduce!(-, +, zero(T), ws.sum_ub, ub, xu)
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

    batch_mapreduce!(abs, max, typemin(T), ws.norm_b, MadNLP.full(batch_solver.rhs))
    batch_mapreduce!(abs, max, typemin(T), ws.norm_c, MadNLP.full(batch_solver.f))

    init_starting_point!(batch_solver)
    initialize_solver_state!(batch_solver)

    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return
end

function initialize_solver_state!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    opt = batch_solver.opt
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(T))
    fill!(ws.status, MadNLP.REGULAR)
    reset_active_view!(batch_solver.batch_views)
    _update_active_mask!(batch_solver)
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
    return
end

function compute_term_gpu!(ws::UniformBatchWorkspace{T}, opt) where T
    ds = T(opt.divergence_scale)
    tol = T(opt.tol)
    div_tol = T(opt.divergence_tol)
    Int_ERROR = Int(MadNLP.INTERNAL_ERROR)
    Int_SOLVED = Int(MadNLP.SOLVE_SUCCEEDED)
    Int_INFEASIBLE = Int(MadNLP.INFEASIBLE_PROBLEM_DETECTED)
    Int_DIVERGING = Int(MadNLP.DIVERGING_ITERATES)
    Int_REGULAR = Int(MadNLP.REGULAR)
    @. ws._term_gpu = ifelse(
        ws._ls_error > zero(Int32),
        Int_ERROR,
        ifelse(
            max(ws.inf_pr, ws.inf_du, ws.inf_compl) <= tol,
            Int_SOLVED,
            ifelse(
                (ws.inf_compl > div_tol * ws.best_complementarity) &
                (ws.dual_obj > max(ds * abs(ws.obj_val), one(T))),
                Int_INFEASIBLE,
                ifelse(
                    ws.obj_val < -(div_tol * max(ds * abs(ws.dual_obj), one(T))),
                    Int_DIVERGING,
                    Int_REGULAR,
                ),
            ),
        ),
    )
    minimum!(ws._any_nonregular_gpu, ws._term_gpu)
end

function update_termination_criteria!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    opt = batch_solver.opt
    bcnt = batch_solver.batch_cnt
    x, xl, xu = batch_solver.x, batch_solver.xl, batch_solver.xu
    zl, zu = batch_solver.zl, batch_solver.zu
    bs = batch_solver.batch_size
    nlb, nub = batch_solver.d.nlb, batch_solver.d.nub

    get_inf_pr!(ws.inf_pr, MadNLP.full(batch_solver.c))
    @. ws.inf_pr /= max(one(T), ws.norm_b)

    get_inf_du!(ws.inf_du, MadNLP.full(batch_solver.f), MadNLP.full(zl),
                MadNLP.full(zu), MadNLP.full(batch_solver.jacl))
    @. ws.inf_du /= max(one(T), ws.norm_c)

    get_inf_compl!(ws.inf_compl, x, xl, zl, xu, zu,
        ws.sum_lb, ws.sum_ub, nlb, nub)
    @. ws.inf_compl /= max(one(T), ws.norm_c)
    @. ws.best_complementarity = min(ws.best_complementarity, ws.inf_compl)

    dual_objective!(ws.dual_obj, MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs),
        lower(zl), lower(xl), upper(zu), upper(xu),
        ws.sum_lb, ws.sum_ub, nlb, nub)

    compute_term_gpu!(ws, opt)
    return
end

function update_termination_status!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    opt = batch_solver.opt
    bcnt = batch_solver.batch_cnt
    bs = batch_solver.batch_size
    Int_REGULAR = Int64(Int(MadNLP.REGULAR))

    walltime_hit = time() - bcnt.start_time[] >= opt.max_wall_time
    max_iter_hit = walltime_hit ? false :
        any(ws.status[i] == MadNLP.REGULAR && bcnt.k[i] >= opt.max_iter for i in 1:bs)

    if !walltime_hit && !max_iter_hit
        copyto!(ws._any_nonregular_cpu, ws._any_nonregular_gpu)
        ws._any_nonregular_cpu[1] == Int_REGULAR && return false
    end

    copyto!(ws._term_cpu, ws._term_gpu)
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
    return true
end

function solve_system!(
    d::BatchUnreducedKKTVector{T},
    batch_solver::AbstractBatchMPCSolver{T},
    p::BatchUnreducedKKTVector{T},
) where T
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(batch_solver.kkt, batch_solver)

    w = batch_solver._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, batch_solver.kkt, d, -one(T), one(T))

    ws = batch_solver.workspace
    MadNLP.full(w) .*= ws.active_mask
    MadNLP.full(p) .*= ws.active_mask

    opt = batch_solver.opt
    check_res = opt.check_residual
    tol_ls = T(opt.tol_linear_solve)
    _fw = MadNLP.full(w)
    _fw .= abs.(_fw)
    batch_maximum!(ws._norm_gpu_w, _fw)                  # (1,bs) per-instance norm_w
    _fw .= abs.(MadNLP.full(p))
    batch_maximum!(ws._norm_gpu_p, _fw)                  # (1,bs) per-instance norm_p
    @. ws._norm_gpu_w /= max(one(T), ws._norm_gpu_p)    # ratio in-place
    @. ws._ls_error |= isnan(ws._norm_gpu_w) | (check_res & (ws._norm_gpu_w > tol_ls))
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

function _bump_failed_regularization!(batch_solver::AbstractBatchMPCSolver{T}, failed_locals, nfailed::Int) where T
    factor_view = active_view(batch_solver.batch_views)
    ws = batch_solver.workspace
    # build root-level mask from local failed idx
    fill!(ws.active_mask_cpu, zero(T))
    @inbounds for k in 1:nfailed
        j = factor_view.local_to_root[failed_locals[k]]
        ws.active_mask_cpu[1, j] = one(T)
    end
    copyto!(ws.active_mask, ws.active_mask_cpu)
    mask = ws.active_mask
    @. batch_solver.del_w = ifelse(mask == one(T), T(100) * batch_solver.del_w, batch_solver.del_w)
    @. batch_solver.del_c = ifelse(mask == one(T), T(100) * batch_solver.del_c, batch_solver.del_c)
    # restore active mask
    # this is required to not throw away any successful factorization that we need later
    _update_active_mask!(batch_solver)
    return
end

function factorize_system!(batch_solver::AbstractBatchMPCSolver)
    ws = batch_solver.workspace
    batch_views = batch_solver.batch_views
    update_regularization!(batch_solver, batch_solver.opt.regularization)
    max_trials = 3
    factor_view = active_view(batch_views)
    failed_locals = batch_views.selected_local_buffer

    for _ in 1:max_trials
        set_aug_diagonal_reg!(batch_solver.kkt, batch_solver)
        MadNLP.factorize_wrapper!(batch_solver)
        nfailed = is_factorized!(
            failed_locals, batch_solver.kkt.batch_solver, factor_view,
        )
        nfailed == 0 && break
        _bump_failed_regularization!(batch_solver, failed_locals, nfailed)
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

    _adjust_boundary_active!(lower(x), lower(xl), upper(x), upper(xu), ws.mu_batch, ws.active_mask)
    increment_k!(batch_solver)  # this is CPU work, ends up overlapped
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
    fill!(batch_solver.workspace._ls_error, zero(Int32))
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
    buf = ws.active_mask_cpu
    fill_batch_view_mask!(buf, active_view(batch_solver.batch_views))
    copyto!(ws.active_mask, buf)
end

function mpc!(batch_solver::AbstractBatchMPCSolver)
    while true
        MadNLP.print_iter(batch_solver)
        update_termination_criteria!(batch_solver)
        changed = update_termination_status!(batch_solver)
        if changed
            update_active_set!(batch_solver)
            active_batch_size(batch_solver) == 0 && return
            _update_active_mask!(batch_solver)
        end
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
    na = active_batch_size(batch_solver)
    bs = batch_solver.batch_size
    k = maximum(bcnt.k)

    active_str = "$na/$bs"
    mod(k, 10) == 0 && MadNLP.@info(logger, @sprintf(
        " iter  active  max_inf_pr  max_inf_du  max_inf_compl  max_alpha_p"))
    MadNLP.@info(logger, @sprintf(
        "%4i  ", k) * lpad(active_str, 6) * @sprintf(
        "   %6.2e     %6.2e      %7.2e      %6.2e",
        maximum(ws.inf_pr), maximum(ws.inf_du),
        maximum(ws.inf_compl), maximum(ws.alpha_p),
    ))
    return
end
