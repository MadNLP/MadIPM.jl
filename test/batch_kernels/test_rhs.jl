@testset "RHS setup functions" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)
        do_first_factorize!(seq, bat)

        @testset "set_initial_primal_rhs!" begin
            MadIPM.set_initial_primal_rhs!(seq)
            MadIPM.set_initial_primal_rhs!(bat)
            @test cmp(MadNLP.full(seq.p), col1(MadNLP.full(bat.p))) < 1e-12
        end

        @testset "set_initial_dual_rhs!" begin
            MadIPM.set_initial_dual_rhs!(seq)
            MadIPM.set_initial_dual_rhs!(bat)
            @test cmp(MadNLP.full(seq.p), col1(MadNLP.full(bat.p))) < 1e-12
        end

        @testset "set_predictive_rhs!" begin
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            @test cmp(MadNLP.primal(seq.p), col1(MadNLP.primal(bat.p))) < 1e-12
            @test cmp(MadNLP.dual(seq.p), col1(MadNLP.dual(bat.p))) < 1e-12
            @test cmp(MadNLP.dual_lb(seq.p), col1(MadNLP.dual_lb(bat.p))) < 1e-12
            @test cmp(MadNLP.dual_ub(seq.p), col1(MadNLP.dual_ub(bat.p))) < 1e-12
        end

        # Need an affine direction to test correction/correction_rhs
        @testset "get_correction!" begin
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            MadIPM.get_correction!(seq, seq.correction_lb, seq.correction_ub)
            MadIPM.get_correction!(bat, MadNLP.full(bat.correction_lb), MadNLP.full(bat.correction_ub))
            @test cmp(seq.correction_lb, col1(MadNLP.full(bat.correction_lb))) < 1e-10
            @test cmp(seq.correction_ub, col1(MadNLP.full(bat.correction_ub))) < 1e-10
        end

        @testset "set_correction_rhs!" begin
            # Use the state from get_correction! above
            mu_val = seq.mu
            MadIPM.set_correction_rhs!(seq, seq.kkt, mu_val, seq.correction_lb, seq.correction_ub, seq.ind_lb, seq.ind_ub)
            MadIPM.set_correction_rhs!(bat, bat.kkt, bat.workspace.mu_batch, MadNLP.full(bat.correction_lb), MadNLP.full(bat.correction_ub), nothing, nothing)
            @test cmp(MadNLP.primal(seq.p), col1(MadNLP.primal(bat.p))) < 1e-10
            @test cmp(MadNLP.dual(seq.p), col1(MadNLP.dual(bat.p))) < 1e-10
            @test cmp(MadNLP.dual_lb(seq.p), col1(MadNLP.dual_lb(bat.p))) < 1e-10
            @test cmp(MadNLP.dual_ub(seq.p), col1(MadNLP.dual_ub(bat.p))) < 1e-10
        end
    end
end
