@testset "KKT / augmented system" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        qp = make_qp()
        seq = build_seq(qp)
        bat = build_batch(qp)

        @testset "set_aug_diagonal_reg!" begin
            MadIPM.update_regularization!(seq, seq.opt.regularization)
            MadIPM.update_regularization!(bat, bat.opt.regularization)
            MadIPM.set_aug_diagonal_reg!(seq.kkt, seq)
            MadIPM.set_aug_diagonal_reg!(bat.kkt, bat)

            skkt = seq.kkt
            bkkt = bat.kkt
            @test cmp(skkt.reg, col1(bkkt.reg)) < 1e-12
            @test cmp(skkt.l_diag, col1(bkkt.l_diag)) < 1e-12
            @test cmp(skkt.u_diag, col1(bkkt.u_diag)) < 1e-12
            @test cmp(skkt.l_lower, col1(bkkt.l_lower)) < 1e-12
            @test cmp(skkt.u_lower, col1(bkkt.u_lower)) < 1e-12
            @test cmp(skkt.pr_diag, col1(MadIPM.pr_diag(bkkt))) < 1e-12
        end

        @testset "build_kkt!" begin
            MadIPM.update_regularization!(seq, seq.opt.regularization)
            MadIPM.update_regularization!(bat, bat.opt.regularization)
            MadIPM.set_aug_diagonal_reg!(seq.kkt, seq)
            MadIPM.set_aug_diagonal_reg!(bat.kkt, bat)
            MadNLP.build_kkt!(seq.kkt)
            MadNLP.build_kkt!(bat.kkt)

            # Compare COO values
            seq_V = seq.kkt.aug_raw.V
            bat_V = bat.kkt.nzVals[:, 1]
            @test cmp(seq_V, bat_V) < 1e-12

            # Compare CSC nzvals
            seq_csc = SparseArrays.nonzeros(seq.kkt.aug_com)
            bat_csc = bat.kkt.aug_com_nzvals[:, 1]
            @test cmp(seq_csc, bat_csc) < 1e-12
        end

        @testset "factorize + solve" begin
            do_first_factorize!(seq, bat)

            # Set same RHS
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            @test cmp(MadNLP.full(seq.d), col1(MadNLP.full(bat.d))) < 1e-10
        end

        @testset "mul! (KKT)" begin
            do_first_factorize!(seq, bat)

            # Set up input vector x from predictive rhs
            MadIPM.set_predictive_rhs!(seq, seq.kkt)
            MadIPM.set_predictive_rhs!(bat, bat.kkt)
            MadIPM.solve_system!(seq.d, seq, seq.p)
            MadIPM.solve_system!(bat.d, bat, bat.p)

            # w = K * d
            w_seq = seq._w1
            w_bat = bat._w1
            fill!(MadNLP.full(w_seq), 0.0)
            fill!(MadNLP.full(w_bat), 0.0)
            mul!(w_seq, seq.kkt, seq.d)
            mul!(w_bat, bat.kkt, bat.d)
            @test cmp(MadNLP.full(w_seq), col1(MadNLP.full(w_bat))) < 1e-10
        end

        @testset "jtprod!" begin
            # Explicit jtprod! call with initialized solvers
            seq_jacl = similar(seq.jacl)
            fill!(seq_jacl, 0.0)
            MadNLP.jtprod!(seq_jacl, seq.kkt, seq.y)

            bat_jacl = similar(MadNLP.full(bat.jacl))
            fill!(bat_jacl, 0.0)
            MadNLP.jtprod!(bat_jacl, bat.kkt, bat.y)

            @test cmp(seq_jacl, col1(bat_jacl)) < 1e-12
        end
    end
end
