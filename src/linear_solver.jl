
#=
    Interface to direct solver for solving KKT system
=#

function factorize_regularized_system!(solver)
    max_trials = 3
    for ntrial in 1:max_trials
        set_aug_diagonal_reg!(solver.kkt, solver)
        MadNLP.factorize_wrapper!(solver)
        if is_factorized(solver.kkt.linear_solver)
            break
        end
        solver.del_w *= 100.0
        solver.del_c *= 100.0
    end
end

function solve_system!(
    d::MadNLP.UnreducedKKTVector{T},
    solver::MadNLP.AbstractMadNLPSolver{T},
    p::MadNLP.UnreducedKKTVector{T},
) where T
    opt = solver.opt
    copyto!(MadNLP.full(d), MadNLP.full(p))
    MadNLP.solve_kkt!(solver.kkt, d)

    # Check residual
    w = solver._w1
    copyto!(MadNLP.full(w), MadNLP.full(p))
    mul!(w, solver.kkt, d, -one(T), one(T))
    norm_w = norm(MadNLP.full(w), Inf)
    norm_p = norm(MadNLP.full(p), Inf)

    residual_ratio = norm_w / max(one(T), norm_p)
    MadNLP.@debug(
        solver.logger,
        @sprintf("Residual after linear solve: %6.2e", residual_ratio),
    )
    if isnan(residual_ratio) || (opt.check_residual && (residual_ratio > opt.tol_linear_solve))
        throw(MadNLP.SolveException)
    end
    return d
end
