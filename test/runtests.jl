using Test
using MadNLP
using MadIPM
using MadNLPTests
using QuadraticModels
using CUDA

function _compare_with_nlp(n, m, ind_fixed, ind_eq; max_ncorr=0, atol=1e-5)
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m=m)
    # Solve with MadNLP for reference.
    # Set `bound_relax_factor=1e-10` to get same behavior as in MadQP.
    nlp_solver = MadNLP.MadNLPSolver(qp; print_level=MadNLP.ERROR, bound_relax_factor=1e-10)
    nlp_stats = MadNLP.solve!(nlp_solver)

    qp_solver = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, max_ncorr=max_ncorr)
    qp_stats = MadIPM.solve!(qp_solver)

    @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    @test qp_stats.objective ≈ nlp_stats.objective atol=atol
    @test qp_stats.solution ≈ nlp_stats.solution atol=atol
    @test qp_stats.constraints ≈ nlp_stats.constraints atol=atol
    @test qp_stats.multipliers ≈ nlp_stats.multipliers atol=atol
    return
end

function simple_lp()
    c = ones(2)
    Hrows = Int[]
    Hcols = Int[]
    Hvals = Float64[]
    Arows = [1, 1]
    Acols = [1, 2]
    Avals = [1.0; 1.0]
    c0 = 0.0
    lvar = [0.0; 0.0]
    uvar = [Inf; Inf]
    lcon = [1.0]
    ucon = [1.0]
    x0 = ones(2)

    return QuadraticModel(
        c,
        Hrows,
        Hcols,
        Hvals,
        Arows = Arows,
        Acols = Acols,
        Avals = Avals,
        lcon = lcon,
        ucon = ucon,
        lvar = lvar,
        uvar = uvar,
        c0 = c0,
        x0 = x0,
        name = "simpleLP",
    )
end

@testset "Test with DenseDummyQP" begin
    # Test results match with MadNLP
    @testset "Size: ($n, $m)" for (n, m) in [(10, 0), (10, 5), (50, 10)]
        _compare_with_nlp(n, m, Int[], Int[]; atol=1e-4)
    end
    @testset "Equality constraints" begin
        n, m = 20, 15
        # Default Mehrotra-predictor.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol=1e-5, max_ncorr=0)
        # Gondzio's multiple correction.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol=1e-5, max_ncorr=5)
    end
    @testset "Fixed variables" begin
        n, m = 20, 15
        _compare_with_nlp(n, m, Int[1, 2], Int[]; atol=1e-5)
        _compare_with_nlp(n, m, Int[1, 2], Int[1, 2, 3, 8]; atol=1e-5)
    end

    # Test inner working in MadIPM
    n, m = 10, 5
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m=m)

    @testset "Step rule $rule" for rule in [
        MadIPM.AdaptiveStep(0.99),
        MadIPM.ConservativeStep(0.99),
        MadIPM.MehrotraAdaptiveStep(0.99),
    ]
        qp_solver = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            step_rule=rule,
        )
        qp_stats = MadIPM.solve!(qp_solver)
        @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Compute reference solution
    qp_solver = MadIPM.MPCSolver(
        qp;
        print_level=MadNLP.ERROR,
        regularization=MadIPM.NoRegularization(),
    )
    sol_ref = MadIPM.solve!(qp_solver)

    @testset "K2.5 KKT linear system" begin
        qp_k25 = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            kkt_system=MadNLP.ScaledSparseKKTSystem,
        )
        sol_k25 = MadIPM.solve!(qp_k25)
        @test sol_k25.status == MadNLP.SOLVE_SUCCEEDED
        @test sol_k25.iter ≈ sol_ref.iter atol=1e-6
        @test sol_k25.objective ≈ sol_ref.objective atol=1e-6
        @test sol_k25.solution ≈ sol_ref.solution atol=1e-6
        @test sol_k25.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol_k25.multipliers ≈ sol_ref.multipliers atol=1e-6
    end

    @testset "Regularization $(reg)" for reg in [
        MadIPM.FixedRegularization(1e-8, -1e-9),
        MadIPM.AdaptiveRegularization(1e-8, -1e-9, 1e-9),
    ]
        solver = MadIPM.MPCSolver(
            qp;
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            regularization=reg,
            rethrow_error=true,
        )
        sol = MadIPM.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end

end

@testset "Test with simple LP" begin
    qp = simple_lp()

    qp_solver = MadIPM.MPCSolver(
        qp;
        print_level=MadNLP.ERROR,
        regularization=MadIPM.NoRegularization(),
    )
    sol_ref = MadIPM.solve!(qp_solver)

    @testset "Presolve" begin
        new_qp, flag = MadIPM.presolve_qp(qp)
        @test flag
    end

    @testset "Standard formulation" begin
        new_qp = MadIPM.standard_form_qp(qp)
        solver = MadIPM.MPCSolver(new_qp; print_level=MadNLP.ERROR)
        sol = MadIPM.solve!(solver)
        @test sol.objective == sol_ref.objective
    end

    @testset "NormalKKTSystem implementation" begin
        # Test
        linear_solver = MadNLP.LapackCPUSolver
        ind_cons = MadNLP.get_index_constraints(qp)
        cb = MadNLP.create_callback(
            MadNLP.SparseCallback, qp,
        )
        kkt = MadNLP.create_kkt_system(
            MadIPM.NormalKKTSystem,
            cb,
            ind_cons,
            linear_solver;
        )
        MadNLPTests.test_kkt_system(kkt, cb)
    end

    @testset "Solve LP with NormalKKTSystem" begin
        solver = MadIPM.MPCSolver(
            qp;
            linear_solver=LDLSolver,
            print_level=MadNLP.ERROR,
            kkt_system=MadIPM.NormalKKTSystem,
            rethrow_error=true,
        )
        sol = MadIPM.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end
end

if CUDA.has_cuda()
    include("test_gpu.jl")
end

