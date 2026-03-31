using BatchQuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel

_test_local_to_root(view) = Int[view.local_to_root[i] for i in 1:view.n]

struct RecordingBatchLinearSolver{T, MT} <: MadNLP.AbstractLinearSolver{T}
    nzvals_mat::MT
    call_counts::Vector{Int}
    factorized::Vector{Bool}
    fail_marker::T
end

@kwdef mutable struct RecordingBatchLinearSolverOptions <: MadNLP.AbstractOptions
    fail_marker::Float64 = -999.0
end

MadNLP.default_options(::Type{RecordingBatchLinearSolver}) = RecordingBatchLinearSolverOptions()

function RecordingBatchLinearSolver(
    aug_com,
    nzvals_mat::AbstractMatrix{T},
    n::Int;
    opt::RecordingBatchLinearSolverOptions = RecordingBatchLinearSolverOptions(),
) where T
    batch_size = size(nzvals_mat, 2)
    return RecordingBatchLinearSolver(
        nzvals_mat,
        zeros(Int, batch_size),
        fill(true, batch_size),
        T(opt.fail_marker),
    )
end

function MadIPM.factorize_active!(
    s::RecordingBatchLinearSolver{T, MT},
    factor_view::MadIPM.BatchView,
) where {T, MT}
    @inbounds for j in 1:MadIPM.local_batch_size(factor_view)
        s.call_counts[j] += 1
        col = view(s.nzvals_mat, :, j)
        corrupted = any(==(s.fail_marker), col)
        s.factorized[j] = !(corrupted && s.call_counts[j] == 1)
    end
    return
end

function MadIPM.failed_factorization_local_count!(
    failed_local_buffer::Vector{Int32},
    s::RecordingBatchLinearSolver,
    factor_view::MadIPM.BatchView,
)
    nfailed = 0
    @inbounds for j in 1:MadIPM.local_batch_size(factor_view)
        if !s.factorized[j]
            nfailed += 1
            failed_local_buffer[nfailed] = j
        end
    end
    return nfailed
end

MadIPM.is_factorized(s::RecordingBatchLinearSolver) = all(s.factorized)

MadIPM.solve_active!(s::RecordingBatchLinearSolver, rhs::AbstractMatrix, active::MadIPM.BatchView) = rhs

function _make_batch_solver(qps; batch_kwargs...)
    bnlp = BatchQuadraticModel(qps)
    return MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR, batch_kwargs...)
end

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
    @testset "Batch views" begin
        solver = _make_batch_solver([simple_lp() for _ in 1:4])
        root = MadIPM.root_view(solver.batch_views)

        @test MadIPM.local_batch_size(root) == 4
        @test MadIPM.batch_size_root(root) == 4
        @test MadIPM.is_identity_view(root)
        @test _test_local_to_root(root) == [1, 2, 3, 4]

        saved_root = MadIPM.select_local!(solver.batch_views, [2, 4])
        child = MadIPM.active_view(solver.batch_views)
        @test MadIPM.local_batch_size(child) == 2
        @test _test_local_to_root(child) == [2, 4]
        @test !MadIPM.is_identity_view(child)
        mask = zeros(Float64, 1, 4)
        MadIPM.fill_batch_view_mask!(mask, child)
        @test mask == [0.0 1.0 0.0 1.0]

        saved_child = MadIPM.select_local!(solver.batch_views, [2])
        grandchild = MadIPM.active_view(solver.batch_views)
        @test _test_local_to_root(grandchild) == [4]
        MadIPM.restore_state!(solver.batch_views, saved_child)

        MadIPM.select_local!(solver.batch_views, Int[])
        empty_child = MadIPM.active_view(solver.batch_views)
        @test MadIPM.local_batch_size(empty_child) == 0
        MadIPM.restore_state!(solver.batch_views, saved_root)

        MadIPM.select_local!(solver.batch_views, [1, 2, 3, 4])
        full_child = MadIPM.active_view(solver.batch_views)
        @test MadIPM.is_identity_view(full_child)
        MadIPM.restore_state!(solver.batch_views, root)
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
        bnlp = BatchQuadraticModel([qp1, qp2])
        @test_throws AssertionError MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR)
        solver = MadIPM.UniformBatchMPCSolver(
            bnlp;
            print_level=MadNLP.ERROR,
            check_batch_structure=false,
        )
        @test solver.batch_size == 2
    end

    @testset "Partial active KKT solve preserves rhs" begin
        solver = _make_batch_solver([simple_lp() for _ in 1:3])
        MadIPM.initialize!(solver)
        status = fill(MadNLP.REGULAR, 3)
        status[2] = MadNLP.INTERNAL_ERROR
        solver.workspace.status .= status
        MadIPM.update_active_set!(solver)

        pd_view = MadNLP.primal_dual(solver.d)
        pd_view .= reshape(collect(1.0:length(pd_view)), size(pd_view))
        pd_before = copy(pd_view)

        MadNLP.build_kkt!(solver.kkt)
        MadNLP.factorize_kkt!(solver.kkt)
        MadNLP.solve_kkt!(solver.kkt, solver)

        @test pd_view[:, 2] == pd_before[:, 2]
    end

    @testset "Batch LP" begin
        _test_batch_lp()
    end
    @testset "Batch QP" begin
        _test_batch_qp()
    end
    @testset "FullBatch different H/A data" begin
        _test_fullbatch_different_data()
    end
    @testset "Residual check marks INTERNAL_ERROR (all fail)" begin
        # Force residual check failure with tol_linear_solve=0 — should mark
        # instances as INTERNAL_ERROR instead of throwing a SolveException.
        qps = [simple_lp() for _ in 1:3]
        bnlp = ObjRHSBatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp;
            print_level=MadNLP.ERROR,
            check_residual=true,
            tol_linear_solve=0.0,
        )
        for i in 1:3
            @test stats[i].status == MadNLP.INTERNAL_ERROR
        end
    end

    @testset "Residual check marks INTERNAL_ERROR (partial)" begin
        # Use NaN objective coefficients to produce NaN residuals for one instance.
        # The per-instance residual check should mark only that instance as INTERNAL_ERROR.
        Hrows = [1, 2]; Hcols = [1, 2]
        Arows = [1, 1]; Acols = [1, 2]

        good_qp() = QuadraticModel(
            [1.0, 1.0], Hrows, Hcols, [2.0, 2.0];
            Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0],
            lvar=[0.0, 0.0], uvar=[Inf, Inf], x0=[0.5, 0.5],
        )
        bad_qp = QuadraticModel(
            [NaN, NaN], Hrows, Hcols, [2.0, 2.0];
            Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
            lcon=[1.0], ucon=[1.0],
            lvar=[0.0, 0.0], uvar=[Inf, Inf], x0=[0.5, 0.5],
        )
        qps = [good_qp(), bad_qp, good_qp()]
        bnlp = BatchQuadraticModel(qps)
        stats = MadIPM.madipm_batch(bnlp; print_level=MadNLP.ERROR)
        # Good instances should solve; bad instance should fail gracefully
        @test stats[1].status == MadNLP.SOLVE_SUCCEEDED
        @test stats[3].status == MadNLP.SOLVE_SUCCEEDED
        @test stats[2].status != MadNLP.SOLVE_SUCCEEDED
    end

    @testset "Factorization retry only refactorizes failed instance" begin
        solver = _make_batch_solver(
            [simple_lp() for _ in 1:3];
            uniformbatch_linear_solver=RecordingBatchLinearSolver,
            regularization=MadIPM.FixedRegularization(1.0, -1.0),
        )
        MadIPM.initialize!(solver)
        fill!(solver.kkt.batch_solver.call_counts, 0)
        fill!(solver.kkt.batch_solver.factorized, true)

        corrupt_k = solver.kkt.n_tot + solver.kkt.nnzh + 1
        solver.kkt.nzVals[corrupt_k, 2] = -999.0
        solver.kkt.nzVals[corrupt_k, 3] = -999.0
        # two corruptions -- should see an extra solve in slots 1 and 2
        # and bumped regularizations in instances 2 and 3

        MadIPM.factorize_system!(solver)

        ls = solver.kkt.batch_solver
        @test ls.call_counts == [2, 2, 2]
        @test solver.del_w == [1.0 100.0 100.0]
        @test solver.del_c == [-1.0 -100.0 -100.0]
    end

    @testset "Factorization retry with one inactive instance" begin
        solver = _make_batch_solver(
            [simple_lp() for _ in 1:3];
            uniformbatch_linear_solver=RecordingBatchLinearSolver,
            regularization=MadIPM.FixedRegularization(1.0, -1.0),
        )
        MadIPM.initialize!(solver)
        fill!(solver.kkt.batch_solver.call_counts, 0)
        fill!(solver.kkt.batch_solver.factorized, true)

        # terminate an instance, so the last slot should be never used
        status = fill(MadNLP.REGULAR, 3)
        status[2] = MadNLP.SOLVE_SUCCEEDED
        solver.workspace.status .= status
        MadIPM.update_active_set!(solver)
        MadIPM._update_active_mask!(solver)

        # corrupt instance 3, which is at position 2 after compacting [1,3]
        corrupt_k = solver.kkt.n_tot + solver.kkt.nnzh + 1
        solver.kkt.nzVals[corrupt_k, 3] = -999.0

        MadIPM.factorize_system!(solver)
        ls = solver.kkt.batch_solver
        @test ls.call_counts == [2, 2, 0]

        # instance 1 and 2's regularization is untouched,
        # instance 3's regularization should be bumped
        @test solver.del_w == [1.0 1.0 100.0]
        @test solver.del_c == [-1.0 -1.0 -100.0]
    end
end
