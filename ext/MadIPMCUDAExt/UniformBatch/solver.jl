batch_init_starting_point_solve!(batch_solver::AbstractBatchSolver) = begin
    MadIPM.set_initial_regularization!.(batch_solver)
    MadNLP.build_kkt!.(batch_solver)
    batch_factorize!(batch_solver.bkkt)

    MadIPM.set_initial_primal_rhs!.(batch_solver)
    batch_solve_system!(batch_solver)
    MadIPM.update_primal_start!.(batch_solver)

    MadIPM.set_initial_dual_rhs!.(batch_solver)
    batch_solve_system!(batch_solver)
    return
end
batch_initialize!(batch_solver::AbstractBatchSolver) = begin
    MadIPM.pre_initialize!.(batch_solver)
    batch_init_starting_point_solve!(batch_solver)
    MadIPM.post_initialize!.(batch_solver)
    return
end
batch_factorize_regularized_system!(batch_solver::AbstractBatchSolver) = begin
    MadIPM.set_aug_diagonal_reg!.(batch_solver)
    MadNLP.build_kkt!.(batch_solver)
    batch_factorize!(batch_solver.bkkt)
    return
end
batch_solve_system!(batch_solver::AbstractBatchSolver) = begin
    pre_solve!.(batch_solver)
    batch_solve!(batch_solver.bkkt)
    post_solve!.(batch_solver)
    return
end

function batch_mpc!(batch_solver::AbstractBatchSolver)
    while true
        # Check termination criteria
        MadNLP.print_iter.(batch_solver)
        MadIPM.update_termination_criteria!.(batch_solver)
        update_batch!(batch_solver)
        all_done(batch_solver) && return

        # Factorize KKT system
        MadIPM.update_regularization!.(batch_solver)
        batch_factorize_regularized_system!(batch_solver)

        # Affine direction
        MadIPM.set_predictive_rhs!.(batch_solver)
        batch_solve_system!(batch_solver)

        # Prediction step size
        MadIPM.prediction_step_size!.(batch_solver)

        # Mehrotra's Correction direction
        MadIPM.set_correction_rhs!.(batch_solver)
        batch_solve_system!(batch_solver)

        # Gondzio's additional correction direction  FIXME
        # batch_gondzio_correction_direction!(batch_solver)

        # Update step size
        MadIPM.update_step_size!.(batch_solver)

        # Apply step
        MadIPM.apply_step!.(batch_solver)

        # Evaluate model at new iterate
        MadIPM.evaluate_model!.(batch_solver)
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