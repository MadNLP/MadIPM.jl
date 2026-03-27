using KernelAbstractions
using MadNLPGPU
using BatchQuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel

# ============================================================
#  Test problem constructors for GPU batch tests
# ============================================================

function _gpu_small_qp()
    n, m = 4, 2
    c = [1.0, -2.0, 0.5, 1.0]
    Hrows = [1, 2, 3, 4]
    Hcols = [1, 2, 3, 4]
    Hvals = [2.0, 1.0, 3.0, 1.5]
    Arows = [1, 1, 2, 2]
    Acols = [1, 2, 3, 4]
    Avals = [1.0, 1.0, 1.0, 1.0]
    return QuadraticModel(
        c, Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[2.0, 1.5],
        lvar=zeros(n), uvar=fill(Inf, n),
        x0=ones(n),
    )
end

function _gpu_doubly_bounded_qp()
    n, m = 3, 2
    c = [1.0, -1.0, 0.5]
    Hrows = [1, 2, 3]; Hcols = [1, 2, 3]; Hvals = [2.0, 1.0, 3.0]
    Arows = [1, 1, 2, 2]; Acols = [1, 2, 2, 3]; Avals = [1.0, 1.0, 1.0, 1.0]
    return QuadraticModel(
        c, Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[1.0, 0.5],
        lvar=[0.0, 0.0, 0.0], uvar=[5.0, 5.0, 5.0],
        x0=ones(n),
    )
end

function _gpu_dense_hess_qp()
    Hrows = [1, 2, 2]; Hcols = [1, 1, 2]; Hvals = [4.0, 2.0, 3.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    return QuadraticModel(
        [1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[Inf, Inf],
        x0=[0.5, 0.5],
    )
end

# ============================================================
#  Helper: solve batch on GPU and compare with CPU reference
# ============================================================

function _test_gpu_batch(qps; atol=1e-6, batch_kwargs...)
    bs = length(qps)

    # CPU reference: solve each QP independently
    refs = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]
    for r in refs
        @test r.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Build CPU batch model, convert to GPU, solve
    cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
    gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
    stats = try
        MadIPM.madipm_batch(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            batch_kwargs...,
        )
    catch e
        @error "madipm_batch failed" exception=(e, catch_backtrace())
        rethrow(e)
    end

    CUDA.@allowscalar for i in 1:bs
        si = stats[i]
        if si.status != MadNLP.SOLVE_SUCCEEDED
            @error "Instance $i failed" status=si.status objective=si.objective
        end
        @test si.status == MadNLP.SOLVE_SUCCEEDED
        @test si.objective ≈ refs[i].objective atol=atol
        @test Array(si.solution) ≈ refs[i].solution atol=atol
    end
end

# ============================================================
#  Tests
# ============================================================

@testset "Batch solver (CUDA)" begin
    @testset "Batch views gather/scatter" begin
        cpu_bnlp = BatchQuadraticModel([simple_lp() for _ in 1:4])
        gpu_bnlp = convert(BatchQuadraticModel{Float64, CuMatrix{Float64}}, cpu_bnlp)
        solver = MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        )
        root = MadIPM.root_view(solver.batch_views)
        saved_root = MadIPM.select_local!(solver.batch_views, [2, 4])
        child = MadIPM.active_view(solver.batch_views)
        MadIPM.select_local!(solver.batch_views, [2])
        grandchild = MadIPM.active_view(solver.batch_views)

        src = cu(reshape(collect(1.0:12.0), 3, 4))
        gathered = similar(src, 3, MadIPM.local_batch_size(child))
        MadIPM.gather_batch_view_columns!(gathered, src, child)
        @test Array(gathered) == Array(src[:, [2, 4]])

        gathered_nested = similar(src, 3, MadIPM.local_batch_size(grandchild))
        MadIPM.gather_batch_view_columns!(gathered_nested, src, grandchild)
        @test Array(gathered_nested) == Array(src[:, [4]])

        scattered = CUDA.fill(-1.0, 3, 4)
        MadIPM.scatter_batch_view_columns!(scattered, gathered, child)
        @test Array(scattered[:, 2]) == Array(src[:, 2])
        @test Array(scattered[:, 4]) == Array(src[:, 4])
        @test Array(scattered[:, 1]) == fill(-1.0, 3)
        @test Array(scattered[:, 3]) == fill(-1.0, 3)
        MadIPM.restore_state!(solver.batch_views, saved_root)
    end

    @testset "Batch structure mismatch throws" begin
        qp1 = QuadraticModel(
            [1.0, 1.0], [1, 2], [1, 2], [2.0, 2.0];
            Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0],
            lvar=[0.0, 0.0], uvar=[1.0, 1.0],
            x0=[0.5, 0.5],
        )
        qp2 = QuadraticModel(
            [1.0, 1.0], [1, 2], [1, 2], [2.0, 2.0];
            Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0],
            lvar=[0.0, 0.0], uvar=[0.0, 1.0],
            x0=[0.0, 0.5],
        )
        cpu_bnlp = BatchQuadraticModel([qp1, qp2])
        gpu_bnlp = convert(BatchQuadraticModel{Float64, CuMatrix{Float64}}, cpu_bnlp)
        @test_throws AssertionError MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
        )
        solver = MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
            check_batch_structure=false,
        )
        @test solver.batch_size == 2
    end

    # ----------------------------------------------------------
    # Identical instances (sanity check)
    # ----------------------------------------------------------
    @testset "Identical LP (bs=4)" begin
        _test_gpu_batch([simple_lp() for _ in 1:4]; atol=1e-5)
    end

    @testset "Identical QP (bs=3)" begin
        _test_gpu_batch([_gpu_small_qp() for _ in 1:3])
    end

    @testset "Identical doubly-bounded QP (bs=2)" begin
        _test_gpu_batch([_gpu_doubly_bounded_qp() for _ in 1:2])
    end

    @testset "Identical dense-Hessian QP (bs=2)" begin
        _test_gpu_batch([_gpu_dense_hess_qp() for _ in 1:2])
    end

    # ----------------------------------------------------------
    # Different-data instances via ObjRHSBatch
    # ----------------------------------------------------------
    @testset "Different LP data (bs=3)" begin
        qp1 = QuadraticModel(
            [1.0, 1.0], Int[], Int[], Float64[];
            Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0],
            lvar=[0.0, 0.0], uvar=[Inf, Inf],
            x0=ones(2),
        )
        qp2 = QuadraticModel(
            [2.0, 0.5], Int[], Int[], Float64[];
            Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
            lcon=[2.0], ucon=[2.0],
            lvar=[0.0, 0.0], uvar=[Inf, Inf],
            x0=ones(2),
        )
        qp3 = QuadraticModel(
            [0.5, 3.0], Int[], Int[], Float64[];
            Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
            lcon=[0.5], ucon=[0.5],
            lvar=[0.0, 0.0], uvar=[Inf, Inf],
            x0=ones(2),
        )
        _test_gpu_batch([qp1, qp2, qp3]; atol=1e-5)
    end

    @testset "Different QP data (bs=2)" begin
        n, m = 4, 2
        Hrows = [1, 2, 3, 4]; Hcols = [1, 2, 3, 4]; Hvals = [2.0, 1.0, 3.0, 1.5]
        Arows = [1, 1, 2, 2]; Acols = [1, 2, 3, 4]; Avals = [1.0, 1.0, 1.0, 1.0]
        qp1 = QuadraticModel(
            [1.0, -2.0, 0.5, 1.0], Hrows, Hcols, Hvals;
            Arows=Arows, Acols=Acols, Avals=Avals,
            lcon=[1.0, 0.5], ucon=[2.0, 1.5],
            lvar=zeros(n), uvar=fill(Inf, n), x0=ones(n),
        )
        qp2 = QuadraticModel(
            [-1.0, 1.0, -0.5, 2.0], Hrows, Hcols, Hvals;
            Arows=Arows, Acols=Acols, Avals=Avals,
            lcon=[0.5, 1.0], ucon=[1.5, 2.0],
            lvar=zeros(n), uvar=fill(Inf, n), x0=ones(n),
        )
        _test_gpu_batch([qp1, qp2])
    end

    @testset "Different doubly-bounded QP data (bs=2)" begin
        n, m = 3, 2
        Hrows = [1, 2, 3]; Hcols = [1, 2, 3]; Hvals = [2.0, 1.0, 3.0]
        Arows = [1, 1, 2, 2]; Acols = [1, 2, 2, 3]; Avals = [1.0, 1.0, 1.0, 1.0]
        qp1 = QuadraticModel(
            [1.0, -1.0, 0.5], Hrows, Hcols, Hvals;
            Arows=Arows, Acols=Acols, Avals=Avals,
            lcon=[1.0, 0.5], ucon=[1.0, 0.5],
            lvar=[0.0, 0.0, 0.0], uvar=[5.0, 5.0, 5.0],
            x0=ones(n),
        )
        qp2 = QuadraticModel(
            [-1.0, 2.0, -0.5], Hrows, Hcols, Hvals;
            Arows=Arows, Acols=Acols, Avals=Avals,
            lcon=[0.5, 1.0], ucon=[0.5, 1.0],
            lvar=[0.0, 0.0, 0.0], uvar=[3.0, 3.0, 3.0],
            x0=ones(n),
        )
        _test_gpu_batch([qp1, qp2])
    end

    @testset "Different dense-Hessian QP data (bs=4)" begin
        Hrows = [1, 2, 2]; Hcols = [1, 1, 2]; Hvals = [4.0, 2.0, 3.0]
        Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
        make_qp(c, rhs) = QuadraticModel(
            c, Hrows, Hcols, Hvals;
            Arows=Arows, Acols=Acols, Avals=Avals,
            lcon=[rhs], ucon=[rhs],
            lvar=[0.0, 0.0], uvar=[Inf, Inf],
            x0=[0.5, 0.5],
        )
        _test_gpu_batch([
            make_qp([1.0, -1.0], 1.0),
            make_qp([-1.0, 2.0], 2.0),
            make_qp([0.5, 0.5], 0.5),
            make_qp([2.0, -2.0], 1.5),
        ])
    end

    # ----------------------------------------------------------
    # Batch size variations
    # ----------------------------------------------------------
    @testset "batch_size=1" begin
        _test_gpu_batch([simple_lp()]; atol=1e-5)
    end

    @testset "batch_size=8" begin
        _test_gpu_batch([_gpu_small_qp() for _ in 1:8])
    end

    # ----------------------------------------------------------
    # BatchQuadraticModel (different H/A values per instance)
    # ----------------------------------------------------------
    @testset "FullBatch different H/A (bs=2)" begin
        Hrows = [1, 2, 2]; Hcols = [1, 1, 2]
        Arows = [1, 1]; Acols = [1, 2]
        lvar = [0.0, 0.0]; uvar = [Inf, Inf]

        qp1 = QuadraticModel([1.0, -1.0], Hrows, Hcols, [4.0, 2.0, 3.0];
            Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
        qp2 = QuadraticModel([-1.0, 2.0], Hrows, Hcols, [6.0, 1.0, 5.0];
            Arows=Arows, Acols=Acols, Avals=[1.5, 0.5],
            lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])

        qps = [qp1, qp2]
        refs = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]
        for r in refs; @test r.status == MadNLP.SOLVE_SUCCEEDED; end

        cpu_bnlp = BatchQuadraticModel(qps)
        gpu_bnlp = convert(BatchQuadraticModel{Float64, CuMatrix{Float64}}, cpu_bnlp)
        stats = MadIPM.madipm_batch(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
        )
        CUDA.@allowscalar for i in 1:2
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ refs[i].objective atol=1e-6
            @test Array(si.solution) ≈ refs[i].solution atol=1e-6
        end
    end

    @testset "Residual check marks INTERNAL_ERROR (GPU)" begin
        qps = [simple_lp() for _ in 1:3]
        cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
        gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
        stats = MadIPM.madipm_batch(gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
            check_residual=true,
            tol_linear_solve=0.0,
        )
        CUDA.@allowscalar for i in 1:3
            @test stats[i].status == MadNLP.INTERNAL_ERROR
        end
    end

    @testset "FullBatch identical QP (bs=3)" begin
        qps = [_gpu_small_qp() for _ in 1:3]
        ref = MadIPM.madipm(qps[1]; print_level=MadNLP.ERROR)
        @test ref.status == MadNLP.SOLVE_SUCCEEDED

        cpu_bnlp = BatchQuadraticModel(qps)
        gpu_bnlp = convert(BatchQuadraticModel{Float64, CuMatrix{Float64}}, cpu_bnlp)
        stats = MadIPM.madipm_batch(
            gpu_bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=MadNLPGPU.CUDSSSolver,
            cudss_algorithm=MadNLP.LDL,
        )
        CUDA.@allowscalar for i in 1:3
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ ref.objective atol=1e-6
            @test Array(si.solution) ≈ ref.solution atol=1e-6
        end
    end
end
