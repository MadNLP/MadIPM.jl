@testset "Regularization" begin
    @testset "$label" for (label, make_qp) in ALL_TEST_PROBLEMS
        @testset "NoRegularization" begin
            qp = make_qp()
            seq = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, regularization=MadIPM.NoRegularization())
            bat_nlp = ObjRHSBatchQuadraticModel([qp])
            bat = MadIPM.UniformBatchMPCSolver(bat_nlp; print_level=MadNLP.ERROR, regularization=MadIPM.NoRegularization())

            MadIPM.init_regularization!(seq, MadIPM.NoRegularization())
            MadIPM.init_regularization!(bat, MadIPM.NoRegularization())
            @test seq.del_w == bat.del_w[1]
            @test seq.del_c == bat.del_c[1]

            MadIPM.update_regularization!(seq, MadIPM.NoRegularization())
            MadIPM.update_regularization!(bat, MadIPM.NoRegularization())
            @test seq.del_w == bat.del_w[1]
            @test seq.del_c == bat.del_c[1]
        end

        @testset "FixedRegularization" begin
            reg = MadIPM.FixedRegularization(1e-8, -1e-9)
            qp = make_qp()
            seq = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, regularization=reg)
            reg2 = MadIPM.FixedRegularization(1e-8, -1e-9)
            bat_nlp = ObjRHSBatchQuadraticModel([qp])
            bat = MadIPM.UniformBatchMPCSolver(bat_nlp; print_level=MadNLP.ERROR, regularization=reg2)

            MadIPM.init_regularization!(seq, reg)
            MadIPM.init_regularization!(bat, reg2)
            @test seq.del_w == bat.del_w[1]
            @test seq.del_c == bat.del_c[1]

            MadIPM.update_regularization!(seq, reg)
            MadIPM.update_regularization!(bat, reg2)
            @test seq.del_w == bat.del_w[1]
            @test seq.del_c == bat.del_c[1]
        end

        @testset "AdaptiveRegularization" begin
            reg = MadIPM.AdaptiveRegularization(1e-8, -1e-9, 1e-9)
            qp = make_qp()
            seq = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, regularization=reg)
            reg2 = MadIPM.AdaptiveRegularization(1e-8, -1e-9, 1e-9)
            bat_nlp = ObjRHSBatchQuadraticModel([qp])
            bat = MadIPM.UniformBatchMPCSolver(bat_nlp; print_level=MadNLP.ERROR, regularization=reg2)

            MadIPM.init_regularization!(seq, reg)
            MadIPM.init_regularization!(bat, reg2)
            @test seq.del_w == bat.del_w[1]
            @test seq.del_c == bat.del_c[1]

            # Multiple updates to test the adaptive decay
            for _ in 1:3
                MadIPM.update_regularization!(seq, reg)
                MadIPM.update_regularization!(bat, reg2)
                @test abs(seq.del_w - bat.del_w[1]) < 1e-15
                @test abs(seq.del_c - bat.del_c[1]) < 1e-15
            end
        end
    end
end
