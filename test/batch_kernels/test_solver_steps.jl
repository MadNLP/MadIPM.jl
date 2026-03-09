@testset "High-level solver steps" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        @testset "init_starting_point!" begin
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)

            @test cmp(MadNLP.full(seq.x), col1(MadNLP.full(bat.x))) < 1e-10
            @test cmp(MadNLP.full(seq.xl), col1(MadNLP.full(bat.xl))) < 1e-10
            @test cmp(MadNLP.full(seq.xu), col1(MadNLP.full(bat.xu))) < 1e-10
            @test cmp(seq.y, col1(MadNLP.full(bat.y))) < 1e-10
            @test cmp(MadNLP.full(seq.zl), col1(MadNLP.full(bat.zl))) < 1e-10
            @test cmp(MadNLP.full(seq.zu), col1(MadNLP.full(bat.zu))) < 1e-10
        end

        @testset "Full first IPM iteration" begin
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)
            do_first_factorize!(seq, bat)

            # --- affine_direction! ---
            MadIPM.affine_direction!(seq)
            MadIPM.affine_direction!(bat)
            @test cmp(MadNLP.full(seq.d), col1(MadNLP.full(bat.d))) < 1e-10

            # --- prediction_step! (from post-factorize state) ---
            # Rebuild fresh solvers since affine_direction! mutated state
            seq = build_seq(qp)
            bat = build_batch(qp)
            do_first_factorize!(seq, bat)

            MadIPM.prediction_step!(seq)
            MadIPM.prediction_step!(bat)

            # Check alpha
            seq_ap, seq_ad = MadIPM.get_fraction_to_boundary_step(seq, 1.0)
            @test abs(seq_ap - bat.workspace.alpha_p[1]) < 1e-10
            @test abs(seq_ad - bat.workspace.alpha_d[1]) < 1e-10
            # Check mu
            @test abs(seq.mu - bat.workspace.mu_batch[1]) < 1e-10
            # Check corrections
            @test cmp(seq.correction_lb, col1(MadNLP.full(bat.correction_lb))) < 1e-10
            @test cmp(seq.correction_ub, col1(MadNLP.full(bat.correction_ub))) < 1e-10

            # --- mehrotra_correction_direction! ---
            MadIPM.mehrotra_correction_direction!(seq)
            MadIPM.mehrotra_correction_direction!(bat)
            @test cmp(MadNLP.full(seq.d), col1(MadNLP.full(bat.d))) < 1e-10

            # --- update_step! ---
            MadIPM.update_step!(seq.opt.step_rule, seq)
            MadIPM.update_step!(bat.opt.step_rule, bat)
            @test abs(seq.alpha_p - bat.workspace.alpha_p[1]) < 1e-10
            @test abs(seq.alpha_d - bat.workspace.alpha_d[1]) < 1e-10

            # --- apply_step! ---
            MadIPM.apply_step!(seq)
            MadIPM.apply_step!(bat)
            @test cmp(MadNLP.full(seq.x), col1(MadNLP.full(bat.x))) < 1e-10
            @test cmp(seq.y, col1(MadNLP.full(bat.y))) < 1e-10
            @test cmp(MadNLP.full(seq.zl), col1(MadNLP.full(bat.zl))) < 1e-10
            @test cmp(MadNLP.full(seq.zu), col1(MadNLP.full(bat.zu))) < 1e-10

            # --- evaluate_model! ---
            MadIPM.evaluate_model!(seq)
            MadIPM.evaluate_model!(bat)
            @test cmp(MadNLP.primal(seq.f), col1(MadNLP.primal(bat.f))) < 1e-10
            @test cmp(seq.c, col1(MadNLP.full(bat.c))) < 1e-10
            @test cmp(seq.jacl, col1(MadNLP.full(bat.jacl))) < 1e-10
        end

        @testset "update_termination_criteria!" begin
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)
            # Run one full iteration first
            do_first_factorize!(seq, bat)
            MadIPM.prediction_step!(seq)
            MadIPM.prediction_step!(bat)
            MadIPM.mehrotra_correction_direction!(seq)
            MadIPM.mehrotra_correction_direction!(bat)
            MadIPM.update_step!(seq.opt.step_rule, seq)
            MadIPM.update_step!(bat.opt.step_rule, bat)
            MadIPM.apply_step!(seq)
            MadIPM.apply_step!(bat)
            MadIPM.evaluate_model!(seq)
            MadIPM.evaluate_model!(bat)

            # Now check termination criteria
            MadIPM.update_termination_criteria!(seq)
            MadIPM.update_termination_criteria!(bat)

            @test abs(seq.inf_pr - bat.workspace.inf_pr[1]) < 1e-10
            @test abs(seq.inf_du - bat.workspace.inf_du[1]) < 1e-10
            @test abs(seq.inf_compl - bat.workspace.inf_compl[1]) < 1e-10
            @test seq.status == bat.workspace.status[1]
        end
    end
end
