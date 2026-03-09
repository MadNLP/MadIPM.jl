@testset "Barrier / complementarity" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)

        @testset "get_complementarity_measure!" begin
            seq_mu = MadIPM.get_complementarity_measure(seq)
            MadIPM.get_complementarity_measure!(bat)
            bat_mu = bat.workspace.mu_curr[1]
            @test abs(seq_mu - bat_mu) < 1e-12
        end

        @testset "get_affine_complementarity_measure!" begin
            do_first_factorize!(seq, bat)
            # Compute affine direction
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            # Compute alpha with tau=1
            seq_ap, seq_ad = MadIPM.get_fraction_to_boundary_step(seq, 1.0)
            fill!(bat.workspace.tau, 1.0)
            MadIPM.get_fraction_to_boundary_step!(bat)

            seq_mu_aff = MadIPM.get_affine_complementarity_measure(seq, seq_ap, seq_ad)
            MadIPM.get_affine_complementarity_measure!(bat, bat.workspace.alpha_p, bat.workspace.alpha_d)
            bat_mu_aff = bat.workspace.mu_affine[1]
            @test abs(seq_mu_aff - bat_mu_aff) < 1e-10
        end

        @testset "update_barrier! (Mehrotra)" begin
            do_first_factorize!(seq, bat)
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            seq_ap, seq_ad = MadIPM.get_fraction_to_boundary_step(seq, 1.0)
            fill!(bat.workspace.tau, 1.0)
            MadIPM.get_fraction_to_boundary_step!(bat)

            seq_mu_aff = MadIPM.get_affine_complementarity_measure(seq, seq_ap, seq_ad)
            MadIPM.get_affine_complementarity_measure!(bat, bat.workspace.alpha_p, bat.workspace.alpha_d)

            seq.mu_curr = MadIPM.update_barrier!(seq.opt.barrier_update, seq, seq_mu_aff)
            MadIPM.update_barrier!(bat.opt.barrier_update, bat, bat.workspace.mu_affine)

            @test abs(seq.mu - bat.workspace.mu_batch[1]) < 1e-10
        end
    end
end
