@testset "Step size computation" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        @testset "get_fraction_to_boundary_step!" begin
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)
            do_first_factorize!(seq, bat)
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            for tau_val in [1.0, 0.99, 0.995]
                seq_ap, seq_ad = MadIPM.get_fraction_to_boundary_step(seq, tau_val)
                fill!(bat.workspace.tau, tau_val)
                MadIPM.get_fraction_to_boundary_step!(bat)
                @test abs(seq_ap - bat.workspace.alpha_p[1]) < 1e-10
                @test abs(seq_ad - bat.workspace.alpha_d[1]) < 1e-10
            end
        end

        @testset "set_tau! (ConservativeStep)" begin
            qp = make_qp()
            bat = build_batch(qp)
            rule = MadIPM.ConservativeStep(0.99)
            MadIPM.set_tau!(rule, bat)
            @test bat.workspace.tau[1] == 0.99
        end

        @testset "set_tau! (AdaptiveStep)" begin
            qp = make_qp()
            bat = build_batch(qp)
            rule = MadIPM.AdaptiveStep(0.99)
            MadIPM.set_tau!(rule, bat)
            expected_tau = max(1.0 - bat.workspace.mu_batch[1], 0.99)
            @test bat.workspace.tau[1] ≈ expected_tau atol=1e-12
        end

        @testset "update_step! ($rule_name)" for (rule_name, make_rule) in [
            ("ConservativeStep", () -> MadIPM.ConservativeStep(0.99)),
            ("AdaptiveStep", () -> MadIPM.AdaptiveStep(0.99)),
            ("MehrotraAdaptiveStep", () -> MadIPM.MehrotraAdaptiveStep(0.99)),
        ]
            qp = make_qp()
            seq = build_seq(qp)
            bat = build_batch(qp)
            do_first_factorize!(seq, bat)

            # Standard flow: prediction → correction → update_step
            MadIPM.prediction_step!(seq)
            MadIPM.prediction_step!(bat)
            MadIPM.mehrotra_correction_direction!(seq)
            MadIPM.mehrotra_correction_direction!(bat)

            rule = make_rule()
            seq.opt.step_rule = rule
            bat.opt.step_rule = rule
            MadIPM.update_step!(rule, seq)
            MadIPM.update_step!(rule, bat)
            @test abs(seq.alpha_p - bat.workspace.alpha_p[1]) < 1e-10
            @test abs(seq.alpha_d - bat.workspace.alpha_d[1]) < 1e-10
        end
    end
end
