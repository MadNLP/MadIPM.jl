using QuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel

function _make_small_qp()
    # Small QP: min 0.5 xᵀHx + cᵀx  s.t.  lcon ≤ Ax ≤ ucon, lvar ≤ x ≤ uvar
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

function _test_batch_lp(; batch_kwargs...)
    qp = simple_lp()
    ref = MadIPM.madipm(qp; print_level=MadNLP.ERROR)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED

    bs = 4
    qps = [simple_lp() for _ in 1:bs]

    @testset "ObjRHSBatch" begin
        bnlp = ObjRHSBatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
        for i in 1:bs
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ ref.objective atol=1e-6
            @test si.solution ≈ ref.solution atol=1e-6
        end
    end

    @testset "FullBatch" begin
        bnlp = BatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
        for i in 1:bs
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ ref.objective atol=1e-6
            @test si.solution ≈ ref.solution atol=1e-6
        end
    end
end

function _test_batch_qp(; batch_kwargs...)
    qp = _make_small_qp()
    ref = MadIPM.madipm(qp; print_level=MadNLP.ERROR)
    @test ref.status == MadNLP.SOLVE_SUCCEEDED

    bs = 3
    qps = [_make_small_qp() for _ in 1:bs]

    @testset "ObjRHSBatch" begin
        bnlp = ObjRHSBatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
        for i in 1:bs
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ ref.objective atol=1e-6
            @test si.solution ≈ ref.solution atol=1e-6
        end
    end

    @testset "FullBatch" begin
        bnlp = BatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
        for i in 1:bs
            si = stats[i]
            @test si.status == MadNLP.SOLVE_SUCCEEDED
            @test si.objective ≈ ref.objective atol=1e-6
            @test si.solution ≈ ref.solution atol=1e-6
        end
    end
end

function _test_fullbatch_different_data(; batch_kwargs...)
    # QP with different H and A values per instance (same sparsity)
    Hrows = [1, 2, 2]; Hcols = [1, 1, 2]
    Arows = [1, 1]; Acols = [1, 2]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]

    qp1 = QuadraticModel([1.0, -1.0], Hrows, Hcols, [4.0, 2.0, 3.0];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    qp2 = QuadraticModel([-1.0, 2.0], Hrows, Hcols, [6.0, 1.0, 5.0];
        Arows=Arows, Acols=Acols, Avals=[1.5, 0.5],
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    qp3 = QuadraticModel([0.5, 0.5], Hrows, Hcols, [3.0, 1.5, 4.0];
        Arows=Arows, Acols=Acols, Avals=[0.5, 2.0],
        lcon=[1.5], ucon=[1.5], lvar=lvar, uvar=uvar, x0=[0.75, 0.75])

    qps = [qp1, qp2, qp3]
    refs = [MadIPM.madipm(qp; print_level=MadNLP.ERROR) for qp in qps]
    for r in refs
        @test r.status == MadNLP.SOLVE_SUCCEEDED
    end

    bnlp = BatchQuadraticModel(qps)
    stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
    for i in 1:3
        si = stats[i]
        @test si.status == MadNLP.SOLVE_SUCCEEDED
        @test si.objective ≈ refs[i].objective atol=1e-6
        @test si.solution ≈ refs[i].solution atol=1e-6
    end
end

@testset "Batch solver (CPU)" begin
    @testset "Batch LP" begin
        _test_batch_lp()
    end
    @testset "Batch QP" begin
        _test_batch_qp()
    end
    @testset "FullBatch different H/A data" begin
        _test_fullbatch_different_data()
    end
end
