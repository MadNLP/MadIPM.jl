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
    end
end

@testset "MadIPM CUDSS Uniform Batch" begin
    n_batch = 5
    qps = [simple_lp(Avals=[i * 0.5; i * 2]) for i in 1:n_batch]
    for (i, qp) in enumerate(qps)
        qp.data.c .+= i * 0.5
    end
    # Move problems to GPU
    qps_gpu = [convert(QuadraticModel{Float64, CuVector{Float64}}, qp) for qp in qps]

    individual_stats = Vector{MadNLP.MadNLPExecutionStats}(undef, n_batch)
    for (i, qp_gpu) in enumerate(qps_gpu)
        individual_stats[i] = MadIPM.madipm(
            qp_gpu;
            linear_solver=MadNLPGPU.CUDSSSolver,
            print_level=MadNLP.ERROR,
        )
        @test individual_stats[i].status == MadNLP.SOLVE_SUCCEEDED
    end

    batch_stats = MadIPM.madipm(
        qps_gpu;
        print_level=MadNLP.ERROR,
    )
    @test length(batch_stats) == n_batch
    for i in 1:n_batch
        @test batch_stats[i].status == individual_stats[i].status
        @test batch_stats[i].objective ≈ individual_stats[i].objective atol=1e-6
        @test batch_stats[i].solution ≈ individual_stats[i].solution atol=1e-6
    end
end
