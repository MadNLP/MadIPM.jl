using KernelAbstractions
using MadNLPGPU

@testset "MadIPMCUDA" begin
    qp = simple_lp()
    # Move problem to the GPU
    qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp)

    for (kkt, algo) in ((MadNLP.ScaledSparseKKTSystem, MadNLP.LDL     ),
                        (MadNLP.SparseKKTSystem      , MadNLP.LDL     ),
                        (MadIPM.NormalKKTSystem      , MadNLP.CHOLESKY))
        solver = MadIPM.MPCSolver(
            qp_gpu;
            kkt_system=kkt,
            linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=algo,
            print_level=MadNLP.ERROR,
        )
        results = MadIPM.solve!(solver)
        @test results.status == MadNLP.SOLVE_SUCCEEDED

        if algo == MadNLP.LDL
            perm_gurobi = MadIPM.permutation_schur(qp)

            solver = MadIPM.MPCSolver(
                qp_gpu;
                kkt_system=kkt,
                linear_solver=MadNLPGPU.CUDSSSolver,
                cudss_algorithm=algo,
                cudss_perm=perm_gurobi,
                print_level=MadNLP.ERROR,
            )
            results = MadIPM.solve!(solver)
            @test results.status == MadNLP.SOLVE_SUCCEEDED
        end
    end
end
