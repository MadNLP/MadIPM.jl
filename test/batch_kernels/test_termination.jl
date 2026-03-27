@testset "Termination status branches" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        @testset "SOLVE_SUCCEEDED" begin
            # Run enough iterations to converge
            qp = make_qp()
            bat = build_batch(qp)

            # Run several iterations until convergence
            for _ in 1:50
                bat.workspace.status[1] != MadNLP.REGULAR && break
                MadIPM.update_termination_criteria!(bat)
                changed = MadIPM.update_termination_status!(bat)
                if changed
                    MadIPM.update_active_set!(bat)
                    MadIPM.local_batch_size(MadIPM.active_view(bat.batch_views)) == 0 && break
                    MadIPM._update_active_mask!(bat)
                end
                MadIPM.factorize_system!(bat)
                MadIPM.prediction_step!(bat)
                MadIPM.mehrotra_correction_direction!(bat)
                MadIPM.update_step!(bat.opt.step_rule, bat)
                MadIPM.zero_inactive_step!(bat)
                MadIPM.apply_step!(bat)
                MadIPM.evaluate_model!(bat)
            end
            @test bat.workspace.status[1] == MadNLP.SOLVE_SUCCEEDED
        end

        @testset "MAXIMUM_ITERATIONS_EXCEEDED" begin
            qp = make_qp()
            bat = build_batch(qp)
            bat.opt.max_iter = 0  # no iterations allowed
            MadIPM.update_termination_criteria!(bat)
            MadIPM.update_termination_status!(bat)
            @test bat.workspace.status[1] == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED
        end

        @testset "MAXIMUM_WALLTIME_EXCEEDED" begin
            qp = make_qp()
            bat = build_batch(qp)
            bat.opt.max_wall_time = 0.0  # zero walltime
            bat.batch_cnt.start_time[] = time() - 1.0  # started 1s ago
            MadIPM.update_termination_criteria!(bat)
            MadIPM.update_termination_status!(bat)
            @test bat.workspace.status[1] == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
        end
    end
end
