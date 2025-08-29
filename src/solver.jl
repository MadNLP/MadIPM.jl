
#=
    Initialization
=#

function init_starting_point!(solver::MadNLP.AbstractMadNLPSolver)
    x = MadNLP.primal(solver.x)
    l, u = solver.xl.values, solver.xu.values
    lb, ub = solver.xl_r, solver.xu_r
    zl, zu = solver.zl_r, solver.zu_r
    xl, xu = solver.x_lr, solver.x_ur
    # use jacl as a buffer
    res = solver.jacl

    # Add initial primal-dual regularization
    solver.kkt.reg .= solver.del_w
    solver.kkt.pr_diag .= solver.del_w
    solver.kkt.du_diag .= solver.del_c

    # Step 0: factorize initial KKT system
    MadNLP.factorize_wrapper!(solver)

    # Step 1: Compute initial primal variable as x0 = x + dx, with dx the
    #         least square solution of the system A * dx = (b - A*x)
    set_initial_primal_rhs!(solver)
    solve_system!(solver.d, solver, solver.p)
    # x0 = x + dx
    axpy!(1.0, MadNLP.primal(solver.d), x)

    # Step 2: Compute initial dual variable as the least square solution of A' * y = -f
    set_initial_dual_rhs!(solver)
    solve_system!(solver.d, solver, solver.p)
    solver.y .= MadNLP.dual(solver.d)

    # Step 3: init bounds multipliers using c + A' * y - zl + zu = 0
    # A' * y
    MadNLP.jtprod!(res, solver.kkt, solver.y)
    # A'*y + c
    axpy!(1.0, MadNLP.primal(solver.f), res)
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
        solver.zl.values, res, l, u, solver.zl.values,
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
        solver.zu.values, res, l, u, solver.zu.values,
    )

    delta_x = max(
        0.0,
        -1.5 * minimum(xl .- lb; init=0.0),
        -1.5 * minimum(ub .- xu; init=0.0),
    )

    delta_s = max(
        0.0,
        -1.5 * minimum(zl; init=0.0),
        -1.5 * minimum(zu; init=0.0),
    )

    xl .= xl .+ delta_x
    xu .= xu .- delta_x
    zl .+= 1.0 + delta_s
    zu .+= 1.0 + delta_s

    μ = 0.0
    if length(zl) > 0
        μ += dot(xl, zl) - dot(lb, zl)
    end
    if length(zu) > 0
        μ += dot(ub, zu) - dot(xu, zu)
    end

    delta_x2 = μ / (2 * (sum(zl) + sum(zu)))
    delta_s2 = μ / (2 * (sum(xl .- lb) + sum(ub .- xu)))

    xl .+= delta_x2
    xu .-= delta_x2
    zl .+= delta_s2
    zu .+= delta_s2

    # Use Ipopt's heuristic to project x back on the interval [l, u]
    kappa = solver.opt.bound_fac
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
        x,
        l, u, x,
    )

    @assert all(solver.zl_r .> 0.0)
    @assert all(solver.zu_r .> 0.0)
    @assert all(solver.x_lr .> solver.xl_r)
    @assert all(solver.x_ur .< solver.xu_r)
    return
end

function initialize!(solver::MadNLP.AbstractMadNLPSolver{T}) where T
    opt = solver.opt

    # Ensure the initial point is inside its bounds
    MadNLP.initialize!(
        solver.cb,
        solver.x,
        solver.xl,
        solver.xu,
        solver.y,
        solver.rhs,
        solver.ind_ineq;
        tol=opt.bound_relax_factor,
        bound_push=opt.bound_push,
        bound_fac=opt.bound_fac,
    )

    fill!(solver.jacl, zero(T))

    # Initializing scaling factors
    # TODO: Implement Ruiz equilibration scaling here
    if opt.scaling
        MadNLP.set_scaling!(
            solver.cb,
            solver.x,
            solver.xl,
            solver.xu,
            solver.y,
            solver.rhs,
            solver.ind_ineq,
            T(100),
        )
    end

    # Initializing KKT system
    MadNLP.initialize!(solver.kkt)
    init_regularization!(solver, solver.opt.regularization)

    # Initializing callbacks
    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)

    # Normalization factors
    solver.norm_b = norm(solver.rhs, Inf)
    solver.norm_c = norm(MadNLP.primal(solver.f), Inf)

    # Find initial position
    init_starting_point!(solver)

    solver.mu = opt.mu_init

    solver.cnt.start_time = time()

    return MadNLP.REGULAR
end

#=
    MPC Algorithm
=#

function affine_direction!(solver::MadNLP.AbstractMadNLPSolver)
    set_predictive_rhs!(solver, solver.kkt)
    solve_system!(solver.d, solver, solver.p)
    return
end

function mehrotra_correction_direction!(solver, correction_lb, correction_ub)
    set_correction_rhs!(solver, solver.kkt, solver.mu, correction_lb, correction_ub, solver.ind_lb, solver.ind_ub)
    solve_system!(solver.d, solver, solver.p)
    return
end

function gondzio_correction_direction!(solver, correction_lb, correction_ub, mu_curr, max_ncorr)
    δ = 0.1
    γ = 0.1
    βmin = 0.1
    βmax = 10.0
    tau = 0.995
    # Load buffer for descent direction.
    Δp = solver._w2.values

    # TODO: this may be redundant with (alpha_p, alpha_d) computed in Mehrotra correction step.
    alpha_p, alpha_d = get_fraction_to_boundary_step(solver, tau)

    for ncorr in 1:max_ncorr
        # Enlarge step sizes in primal and dual spaces.
        tilde_alpha_p = min(alpha_p + δ, 1.0)
        tilde_alpha_d = min(alpha_d + δ, 1.0)
        # Apply Mehrotra's heuristic for centering parameter mu.
        ga = get_affine_complementarity_measure(solver, tilde_alpha_p, tilde_alpha_d)
        g = mu_curr
        mu = (ga / g)^2 * ga  # Eq. (12)
        # Add additional correction.
        set_extra_correction!(
            solver, correction_lb, correction_ub,
            tilde_alpha_p, tilde_alpha_d, βmin, βmax, mu,
        )
        # Update RHS.
        set_correction_rhs!(
            solver,
            solver.kkt,
            mu,
            correction_lb,
            correction_ub,
            solver.ind_lb,
            solver.ind_ub,
        )
        # Solve KKT linear system.
        copyto!(Δp, solver.d.values)
        solve_system!(solver.d, solver, solver.p)
        hat_alpha_p, hat_alpha_d = get_fraction_to_boundary_step(solver, tau)

        # Stop extra correction if the stepsize does not increase sufficiently
        if (hat_alpha_p < 1.005 * alpha_p) || (hat_alpha_d < 1.005 * alpha_d)
            copyto!(solver.d.values, Δp)
            break
        else
            alpha_p = hat_alpha_p
            alpha_d = hat_alpha_d
        end
    end

    return alpha_p, alpha_d
end

# Predictor-corrector method
function mpc!(solver::MadNLP.AbstractMadNLPSolver)
    nlb, nub = length(solver.ind_lb), length(solver.ind_ub)

    while true
        # A' y
        MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)

        #####
        # Update info
        #####
        solver.inf_pr = MadNLP.get_inf_pr(solver.c) / max(1.0, solver.norm_b)
        solver.inf_du = MadNLP.get_inf_du(
            MadNLP.full(solver.f),
            MadNLP.full(solver.zl),
            MadNLP.full(solver.zu),
            solver.jacl,
            1.0,
        ) / max(1.0, solver.norm_c)
        solver.inf_compl = get_optimality_gap(solver) / max(1.0, solver.norm_c)

        MadNLP.print_iter(solver)

        #####
        # Termination criteria
        #####
        if max(solver.inf_pr, solver.inf_du, solver.inf_compl) <= solver.opt.tol
            return MadNLP.SOLVE_SUCCEEDED
        elseif solver.cnt.k >= solver.opt.max_iter
            return MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        elseif time()-solver.cnt.start_time >= solver.opt.max_wall_time
            return MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        end

        #####
        # Factorize KKT system
        #####
        update_regularization!(solver, solver.opt.regularization)
        factorize_regularized_system!(solver)

        #####
        # Prediction step
        #####
        affine_direction!(solver)
        alpha_aff_p, alpha_aff_d = get_fraction_to_boundary_step(solver, 1.0)
        mu_affine = get_affine_complementarity_measure(solver, alpha_aff_p, alpha_aff_d)
        get_correction!(solver, solver.correction_lb, solver.correction_ub)

        #####
        # Update barrier
        #####
        mu_curr = update_barrier!(solver.opt.barrier_update, solver, mu_affine)

        #####
        # Mehrotra's Correction step
        #####
        mehrotra_correction_direction!(
            solver,
            solver.correction_lb,
            solver.correction_ub,
        )

        #####
        # Gondzio's additional correction
        #####
        if solver.opt.max_ncorr > 0
            alpha_p, alpha_d = gondzio_correction_direction!(
                solver,
                solver.correction_lb,
                solver.correction_ub,
                mu_curr,
                solver.opt.max_ncorr,
            )
        end

        #####
        # Next trial point
        #####
        update_step!(solver.opt.step_rule, solver)

        # Update primal-dual solution
        axpy!(solver.alpha_p, MadNLP.primal(solver.d), MadNLP.primal(solver.x))
        axpy!(solver.alpha_d, MadNLP.dual(solver.d), solver.y)
        solver.zl_r .+= solver.alpha_d .* MadNLP.dual_lb(solver.d)
        solver.zu_r .+= solver.alpha_d .* MadNLP.dual_ub(solver.d)

        # Update callbacks
        solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
        MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
        MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)

        MadNLP.adjust_boundary!(solver.x_lr,solver.xl_r,solver.x_ur,solver.xu_r,solver.mu)
        solver.cnt.k += 1
    end
end

function solve!(
    solver::MadNLP.AbstractMadNLPSolver;
    kwargs...
)
    stats = MadNLP.MadNLPExecutionStats(solver)
    nlp = solver.nlp
    solver.cnt.start_time = time()

    if !isempty(kwargs)
        MadNLP.set_options!(solver.opt, kwargs)
    end

    try
        MadNLP.@notice(solver.logger,"This is MadLP, running with $(MadNLP.introduce(solver.kkt.linear_solver))\n")
        MadNLP.print_init(solver)
        initialize!(solver)
        solver.status = mpc!(solver)
    catch e
        if e isa MadNLP.InvalidNumberException
            if e.callback == :obj
                solver.status=MadNLP.INVALID_NUMBER_OBJECTIVE
            elseif e.callback == :grad
                solver.status=MadNLP.INVALID_NUMBER_GRADIENT
            elseif e.callback == :cons
                solver.status=MadNLP.INVALID_NUMBER_CONSTRAINTS
            elseif e.callback == :jac
                solver.status=MadNLP.INVALID_NUMBER_JACOBIAN
            elseif e.callback == :hess
                solver.status=MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN
            else
                solver.status=MadNLP.INVALID_NUMBER_DETECTED
            end
        elseif e isa MadNLP.NotEnoughDegreesOfFreedomException
            solver.status=MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM
        elseif e isa MadNLP.LinearSolverException
            solver.status=MadNLP.ERROR_IN_STEP_COMPUTATION;
            solver.opt.rethrow_error && rethrow(e)
        elseif e isa InterruptException
            solver.status=MadNLP.USER_REQUESTED_STOP
            solver.opt.rethrow_error && rethrow(e)
        else
            solver.status=MadNLP.INTERNAL_ERROR
            solver.opt.rethrow_error && rethrow(e)
        end
    finally
        solver.cnt.total_time = time() - solver.cnt.start_time
        if !(solver.status < MadNLP.SOLVE_SUCCEEDED)
            MadNLP.print_summary(solver)
        end
        MadNLP.@notice(solver.logger,"EXIT: $(MadNLP.get_status_output(solver.status, solver.opt))")
        finalize(solver.logger)

        MadNLP.update!(stats,solver)
    end

    return stats
end

"""
    madipm(m; kwargs...)

Solve the model `m` using the MadIPM solver.
"""
function madipm(m; kwargs...)
    solver = MadIPM.MPCSolver(m; kwargs...)
    return MadIPM.solve!(solver)
end
