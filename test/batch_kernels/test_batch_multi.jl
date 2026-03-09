@testset "Multi-batch (batch_size > 1)" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        @testset "batch_size=3 consistency" begin
            # All instances identical: every column should match the bs=1 result
            qp = make_qp()
            bat1 = build_batch(qp)
            bat3 = build_batch_n(qp, 3)
            do_first_factorize!(bat1, bat3)

            # Run one full iteration on both
            MadIPM.prediction_step!(bat1)
            MadIPM.prediction_step!(bat3)
            MadIPM.mehrotra_correction_direction!(bat1)
            MadIPM.mehrotra_correction_direction!(bat3)
            MadIPM.update_step!(bat1.opt.step_rule, bat1)
            MadIPM.update_step!(bat3.opt.step_rule, bat3)
            MadIPM.apply_step!(bat1)
            MadIPM.apply_step!(bat3)

            # Every column of batch_size=3 should match the single-instance result
            for i in 1:3
                @test cmp(col1(MadNLP.full(bat1.x)), view(MadNLP.full(bat3.x), :, i)) < 1e-10
                @test cmp(col1(MadNLP.full(bat1.y)), view(MadNLP.full(bat3.y), :, i)) < 1e-10
                @test cmp(col1(MadNLP.full(bat1.zl)), view(MadNLP.full(bat3.zl), :, i)) < 1e-10
                @test cmp(col1(MadNLP.full(bat1.zu)), view(MadNLP.full(bat3.zu), :, i)) < 1e-10
            end
            @test bat3.workspace.alpha_p[1] == bat3.workspace.alpha_p[2] == bat3.workspace.alpha_p[3]
            @test bat3.workspace.alpha_d[1] == bat3.workspace.alpha_d[2] == bat3.workspace.alpha_d[3]
        end

        @testset "active-set deactivation" begin
            qp = make_qp()
            bat = build_batch_n(qp, 3)

            # Mark instance 2 as converged
            bat.workspace.status[2] = MadNLP.SOLVE_SUCCEEDED
            MadIPM.update_active_set!(bat.kkt, bat.workspace.status)

            @test bat.kkt.active_batch_size[] == 2
            @test bat.kkt.batch_map[2] == 0  # deactivated
            @test bat.kkt.batch_map[1] != 0  # still active
            @test bat.kkt.batch_map[3] != 0  # still active

            # zero_inactive_step! should zero out the deactivated instance
            fill!(bat.workspace.active_mask, 1.0)
            MadIPM._update_active_mask!(bat)
            fill!(bat.workspace.alpha_p, 0.5)
            fill!(bat.workspace.alpha_d, 0.5)
            MadIPM.zero_inactive_step!(bat)
            @test bat.workspace.alpha_p[2] == 0.0
            @test bat.workspace.alpha_d[2] == 0.0
            @test bat.workspace.alpha_p[1] == 0.5
            @test bat.workspace.alpha_d[1] == 0.5
        end
    end
end
