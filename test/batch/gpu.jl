using KernelAbstractions
using MadNLPGPU
using BatchQuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel

function _gpu_batch(qps; Model=ObjRHSBatchQuadraticModel, atol=1e-6, batch_kwargs...)
    bs = length(qps)
    refs = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]
    for r in refs
        @test r.status == MadNLP.SOLVE_SUCCEEDED
    end
    cpu_bnlp = Model(qps)

    # TODO: make this consistent in BQM
    CuModel = Model{Float64, Model <: ObjRHSBatchQuadraticModel ? CuVector{Float64} : CuMatrix{Float64}}
    gpu_bnlp = convert(CuModel, cpu_bnlp)
    stats = MadIPM.madipm_batch(gpu_bnlp;
        print_level=MadNLP.ERROR,
        uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
        batch_kwargs...)
    CUDACore.@allowscalar for i in 1:bs
        @test stats[i].status == MadNLP.SOLVE_SUCCEEDED
        @test stats[i].objective ≈ refs[i].objective atol=atol
        @test Array(stats[i].solution) ≈ refs[i].solution atol=atol
    end
end

@testset "Batch solver (CUDA)" begin

@testset "GPU gather/scatter" begin
    cpu_bnlp = BatchQuadraticModel([_lp() for _ in 1:4])
    gpu_bnlp = convert(BatchQuadraticModel{Float64, CuMatrix{Float64}}, cpu_bnlp)
    solver = MadIPM.UniformBatchMPCSolver(gpu_bnlp;
        print_level=MadNLP.ERROR, uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver)
    bvs = solver.batch_views

    saved = MadIPM.select_local!(bvs, [2, 4])
    child = MadIPM.active_view(bvs)

    src = cu(reshape(collect(1.0:12.0), 3, 4))
    gathered = similar(src, 3, 2)
    MadIPM.gather_batch_view_columns!(gathered, src, child)
    @test Array(gathered) == Array(src[:, [2, 4]])

    scattered = CUDACore.fill(-1.0, 3, 4)
    MadIPM.scatter_batch_view_columns!(scattered, gathered, child)
    @test Array(scattered[:, [2, 4]]) == Array(src[:, [2, 4]])
    @test all(Array(scattered[:, [1, 3]]) .== -1.0)

    MadIPM.restore_state!(bvs, saved)
end

@testset "ObjRHSBatch LP bs=4" begin
    _gpu_batch([_lp() for _ in 1:4]; atol=1e-5)
end

@testset "ObjRHSBatch QP bs=3" begin
    _gpu_batch([_qp() for _ in 1:3])
end

@testset "ObjRHSBatch QP doubly bs=2" begin
    _gpu_batch([_qp_db() for _ in 1:2])
end

@testset "ObjRHSBatch QP dense bs=2" begin
    _gpu_batch([_qp_dense() for _ in 1:2])
end

@testset "ObjRHSBatch different data bs=3" begin
    _gpu_batch([
        QuadraticModel([1.0,1.0], Int[], Int[], Float64[];
            Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
            lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=ones(2)),
        QuadraticModel([2.0,0.5], Int[], Int[], Float64[];
            Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
            lcon=[2.0], ucon=[2.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=ones(2)),
        QuadraticModel([0.5,3.0], Int[], Int[], Float64[];
            Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
            lcon=[0.5], ucon=[0.5], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=ones(2)),
    ]; atol=1e-5)
end

@testset "bs=1" begin
    _gpu_batch([_lp()]; atol=1e-5)
end

@testset "bs=8" begin
    _gpu_batch([_qp() for _ in 1:8])
end

@testset "FullBatch identical QP bs=3" begin
    _gpu_batch([_qp() for _ in 1:3]; Model=BatchQuadraticModel)
end

@testset "FullBatch different H/A bs=2" begin
    Hrows = [1,2,2]; Hcols = [1,1,2]; Arows = [1,1]; Acols = [1,2]
    qps = [
        QuadraticModel([1.0,-1.0], Hrows, Hcols, [4.0,2.0,3.0];
            Arows=Arows, Acols=Acols, Avals=[1.0,1.0],
            lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=[0.5,0.5]),
        QuadraticModel([-1.0,2.0], Hrows, Hcols, [6.0,1.0,5.0];
            Arows=Arows, Acols=Acols, Avals=[1.5,0.5],
            lcon=[2.0], ucon=[2.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=[1.0,1.0]),
    ]
    _gpu_batch(qps; Model=BatchQuadraticModel)
end

@testset "MehrotraAdaptiveStep" begin
    _gpu_batch([_qp() for _ in 1:3]; step_rule=MadIPM.MehrotraAdaptiveStep(0.99))
end

@testset "residual check INTERNAL_ERROR" begin
    cpu_bnlp = ObjRHSBatchQuadraticModel([_lp() for _ in 1:3])
    gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
    stats = MadIPM.madipm_batch(gpu_bnlp;
        print_level=MadNLP.ERROR,
        uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=MadNLP.LDL,
        check_residual=true, tol_linear_solve=0.0)
    CUDACore.@allowscalar for i in 1:3
        @test stats[i].status == MadNLP.INTERNAL_ERROR
    end
end

end
