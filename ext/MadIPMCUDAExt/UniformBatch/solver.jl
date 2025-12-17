batch_init_starting_point_solve!(batch_solver::UniformBatchSolver) = begin
    for_active(batch_solver,
        MadIPM.set_initial_regularization!,
        MadNLP.build_kkt!
    )
    batch_factorize!(batch_solver.bkkt)

    for_active(batch_solver,
        MadIPM.set_initial_primal_rhs!
    )
    batch_solve_system!(batch_solver)
    for_active(batch_solver,
        MadIPM.update_primal_start!,
        MadIPM.set_initial_dual_rhs!
    )
    batch_solve_system!(batch_solver)
    return
end
batch_factorize_regularized_system!(batch_solver::UniformBatchSolver) = begin
    for_active(batch_solver,
        MadIPM.set_aug_diagonal_reg!,
        MadNLP.build_kkt!
    )
    batch_factorize!(batch_solver.bkkt)
    return
end
batch_solve_system!(batch_solver::UniformBatchSolver) = begin
    for_active(batch_solver,
        pre_solve!
    )
    batch_solve!(batch_solver.bkkt)
    for_active(batch_solver,
        post_solve!
    )
    return
end

batch_func(batch_solver::UniformBatchSolver, ::typeof(MadIPM.factorize_regularized_system!)) = batch_factorize_regularized_system!(batch_solver)
batch_func(batch_solver::UniformBatchSolver, ::typeof(MadIPM.solve_system!)) = batch_solve_system!(batch_solver)
batch_func(batch_solver::UniformBatchSolver, ::typeof(MadIPM.init_starting_point_solve!)) = batch_init_starting_point_solve!(batch_solver)

@inline function batch_func(batch_solver::AbstractBatchSolver, func)
    for i in 1:batch_solver.bkkt.active_batch_size[]
        solver_idx = batch_solver.bkkt.batch_map_rev[i]
        solver = batch_solver[solver_idx]
        func(solver);
    end
end

@inline function for_active(batch_solver, funcs...)
    for func in funcs
        batch_func(batch_solver, func)
    end
end

function batch_initialize!(batch_solver::AbstractBatchSolver)
    for_active(batch_solver,
        MadIPM.pre_initialize!,
        MadIPM.init_starting_point_solve!,
        MadIPM.post_initialize!
    )
    return
end
function batch_mpc!(batch_solver::AbstractBatchSolver)
    while true
        # Check termination criteria
        for_active(batch_solver,
            MadNLP.print_iter,
            MadIPM.update_termination_criteria!,
        )
        update_batch!(batch_solver)
        all_done(batch_solver) && return

        # Run MPC step
        for_active(batch_solver,
            MadIPM.update_regularization!,
            MadIPM.factorize_regularized_system!,
            MadIPM.set_predictive_rhs!,
            MadIPM.solve_system!,
            MadIPM.prediction_step_size!,
            MadIPM.set_correction_rhs!,
            MadIPM.solve_system!,
            MadIPM.gondzio_correction_direction!,
            MadIPM.update_step_size!,
            MadIPM.apply_step!,
            MadIPM.evaluate_model!
        )
    end
end


function MadIPM.solve!(batch_solver::AbstractBatchSolver)
    batch_stats = [MadNLP.MadNLPExecutionStats(solver) for solver in batch_solver]  # TODO: BatchExecutionStats?

    try
        MadNLP.@notice(first(batch_solver).logger,"This is MadIPM, running with $(MadNLP.introduce(batch_solver.bkkt.linear_solver)), batch size $(length(batch_solver))\n")
        batch_initialize!(batch_solver)
        batch_mpc!(batch_solver)
    catch e
        rethrow(e)  # FIXME
    finally
        for (stats, solver) in zip(batch_stats, batch_solver.solvers)
            MadIPM.finalize!(stats, solver)
        end
    end

    return batch_stats
end


function MadIPM.madipm(ms::AbstractVector{NLPModel}; kwargs...) where {NLPModel <: NLPModels.AbstractNLPModel}
    solvers = MadIPM.MPCSolver.(ms; linear_solver = NoLinearSolver, kwargs...) # TODO: special constructor to share kkt/cb memory/set NoLinearSolver
    batch_solver = UniformBatchSolver(solvers, linear_solver = MadNLPGPU.CUDSSSolver)  # TODO: add some detection for the best BatchSolver to use (for now we only have UniformBatchSolver anyway)
    return MadIPM.solve!(batch_solver)
end