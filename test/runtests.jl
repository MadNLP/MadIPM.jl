using Test

using MathOptInterface
using MadNLP
using MadIPM
using MadNLPTests
using MadIPM.Models
import MadIPM.Models: LPData, QPData, LinearModel, QuadraticModel
using NLPModels
using SparseMatricesCOO: SparseMatrixCOO
using SparseArrays, LinearAlgebra
using CUDA

function QuadraticModel(
    c::AbstractVector{T},
    Hrows::AbstractVector{<:Integer},
    Hcols::AbstractVector{<:Integer},
    Hvals::AbstractVector{T};
    Arows::AbstractVector{<:Integer} = Int[],
    Acols::AbstractVector{<:Integer} = Int[],
    Avals::AbstractVector{T} = T[],
    lcon::AbstractVector{T} = T[],
    ucon::AbstractVector{T} = T[],
    lvar::AbstractVector{T} = fill(T(-Inf), length(c)),
    uvar::AbstractVector{T} = fill(T(Inf), length(c)),
    c0::Real = zero(T),
    x0::AbstractVector{T} = zeros(T, length(c)),
    y0::AbstractVector{T} = T[],
    minimize::Bool = true,
    name::String = "QP",
) where {T}
    nvar = length(c)
    ncon = max(length(lcon), length(ucon), isempty(Arows) ? 0 : maximum(Arows))
    A = SparseMatrixCOO(
        ncon,
        nvar,
        Vector{Int}(Arows),
        Vector{Int}(Acols),
        Vector{T}(Avals),
    )
    H = SparseMatrixCOO(
        nvar,
        nvar,
        Vector{Int}(Hrows),
        Vector{Int}(Hcols),
        Vector{T}(Hvals),
    )
    lcon_ = isempty(lcon) ? fill(T(-Inf), ncon) : Vector{T}(lcon)
    ucon_ = isempty(ucon) ? fill(T(Inf), ncon) : Vector{T}(ucon)
    y0_ = length(y0) == ncon ? Vector{T}(y0) : zeros(T, ncon)
    data = QPData(
        A,
        Vector{T}(c),
        H;
        lvar = Vector{T}(lvar),
        uvar = Vector{T}(uvar),
        lcon = lcon_,
        ucon = ucon_,
        c0 = T(c0),
    )
    return QuadraticModel(
        data;
        x0 = Vector{T}(x0),
        y0 = y0_,
        minimize = minimize,
        name = name,
    )
end

function _compare_with_nlp(n, m, ind_fixed, ind_eq; max_ncorr = 0, atol = 1e-5)
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m = m)
    # Solve with MadNLP for reference.
    # Set `bound_relax_factor=1e-10` to get same behavior as in MadQP.
    nlp_solver =
        MadNLP.MadNLPSolver(qp; print_level = MadNLP.ERROR, bound_relax_factor = 1e-10)
    nlp_stats = MadNLP.solve!(nlp_solver)

    qp_solver = MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, max_ncorr = max_ncorr)
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
        _compare_with_nlp(n, m, Int[], Int[]; atol = 1e-4)
    end
    @testset "Equality constraints" begin
        n, m = 20, 15
        # Default Mehrotra-predictor.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol = 1e-5, max_ncorr = 0)
        # Gondzio's multiple correction.
        _compare_with_nlp(n, m, Int[], Int[1, 2, 3, 8]; atol = 1e-5, max_ncorr = 5)
    end
    @testset "Fixed variables" begin
        n, m = 20, 15
        _compare_with_nlp(n, m, Int[1, 2], Int[]; atol = 1e-5)
        _compare_with_nlp(n, m, Int[1, 2], Int[1, 2, 3, 8]; atol = 1e-5)
    end

    # Test inner working in MadIPM
    n, m = 10, 5
    x0 = zeros(n)
    qp = MadNLPTests.DenseDummyQP(x0; m = m)

    @testset "Step rule $rule" for rule in [
        MadIPM.AdaptiveStep(0.99),
        MadIPM.ConservativeStep(0.99),
        MadIPM.MehrotraAdaptiveStep(0.99),
    ]
        qp_solver = MadIPM.MPCSolver(qp; print_level = MadNLP.ERROR, step_rule = rule)
        qp_stats = MadIPM.solve!(qp_solver)
        @test qp_stats.status == MadNLP.SOLVE_SUCCEEDED
    end

    # Compute reference solution
    qp_solver = MadIPM.MPCSolver(
        qp;
        print_level = MadNLP.ERROR,
        regularization = MadIPM.NoRegularization(),
    )
    sol_ref = MadIPM.solve!(qp_solver)

    @testset "K2.5 KKT linear system" begin
        qp_k25 = MadIPM.MPCSolver(
            qp;
            print_level = MadNLP.ERROR,
            kkt_system = MadNLP.ScaledSparseKKTSystem,
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
            linear_solver = LDLSolver,
            print_level = MadNLP.ERROR,
            regularization = reg,
            rethrow_error = true,
        )
        sol = MadIPM.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end

end

# Is (x, y) a KKT point of `qp` in the presolver's convention
# c + Q x - Aᵀy = z_l - z_u with z_l, z_u >= 0?
function presolve_kkt_ok(qp, x, y; tol = 1e-8)
    n, m = NLPModels.get_nvar(qp), NLPModels.get_ncon(qp)
    Ac = MadIPM.Models.operator_sparse_matrix(qp.data.A)
    A = sparse(Ac.rows, Ac.cols, Ac.vals, m, n)
    Qc = MadIPM.Models.operator_sparse_matrix(qp.data.Q)
    Q = Symmetric(sparse(Qc.rows, Qc.cols, Qc.vals, n, n), :L)
    lvar, uvar = NLPModels.get_lvar(qp), NLPModels.get_uvar(qp)
    lcon, ucon = NLPModels.get_lcon(qp), NLPModels.get_ucon(qp)
    all(lvar .- tol .<= x .<= uvar .+ tol) || return false
    r = A * x
    all(lcon .- tol .<= r .<= ucon .+ tol) || return false
    z = qp.data.c .+ Q * x .- A' * y
    for j = 1:n
        at_l = x[j] <= lvar[j] + tol
        at_u = x[j] >= uvar[j] - tol
        (at_l && at_u) && continue
        at_l && (z[j] >= -tol || return false; continue)
        at_u && (z[j] <= tol || return false; continue)
        abs(z[j]) <= tol || return false
    end
    for i = 1:m
        lcon[i] == ucon[i] && continue
        at_l = r[i] <= lcon[i] + tol
        at_u = r[i] >= ucon[i] - tol
        at_l && (y[i] >= -tol || return false; continue)
        at_u && (y[i] <= tol || return false; continue)
        abs(y[i]) <= tol || return false
    end
    return true
end

@testset "Test with simple LP" begin
    qp = simple_lp()

    qp_solver = MadIPM.MPCSolver(
        qp;
        print_level = MadNLP.ERROR,
        regularization = MadIPM.NoRegularization(),
    )
    sol_ref = MadIPM.solve!(qp_solver)

    @testset "Presolve" begin
        # simple_lp() has nothing reducible → unchanged
        m, status = MadIPM.presolve_qp(qp)
        @test status == MadIPM.Models.Presolve.PRESOLVE_UNCHANGED
        @test m === qp

        # model with fixed variable should reduce
        qp_fixed = QuadraticModel(
            [1.0, 1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            lcon = [1.0],
            ucon = [1.0],
            lvar = [0.0, 0.0, 1.0],
            uvar = [Inf, Inf, 1.0],
        )
        red, status = MadIPM.presolve_qp(qp_fixed)
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        @test NLPModels.get_nvar(red) == 2

        # singleton row: bounds transferred onto x1, row removed
        qp_srow = QuadraticModel(
            [1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2],
            Acols = [1, 2, 1],
            Avals = [1.0, 1.0, 2.0],
            lcon = [1.0, 0.0],
            ucon = [1.0, 4.0],
            lvar = [0.0, 0.0],
            uvar = [Inf, Inf],
        )
        red, status = MadIPM.presolve_qp(qp_srow)
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        @test NLPModels.get_ncon(red) == 1
        @test NLPModels.get_nvar(red) == 2
        @test NLPModels.get_uvar(red)[1] == 2.0  # 0 <= 2*x1 <= 4

        # free row (bounds (-Inf, Inf)) removed
        qp_frow = QuadraticModel(
            [1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2, 2],
            Acols = [1, 2, 1, 2],
            Avals = [1.0, 1.0, 1.0, -1.0],
            lcon = [1.0, -Inf],
            ucon = [1.0, Inf],
            lvar = [0.0, 0.0],
            uvar = [Inf, Inf],
        )
        red, status = MadIPM.presolve_qp(qp_frow)
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        @test NLPModels.get_ncon(red) == 1
        @test NLPModels.get_nvar(red) == 2

        # free linear singleton column: x3 free, only in row 2, not in Q.
        # min x1 + x2 + x3  s.t.  x1 + x2 == 1,  x2 + x3 >= 2
        # y2 = c3/a23 = 1 > 0 → row 2 binds at lcon = 2; eliminating (x3, row 2)
        # gives  min x1 + 0*x2 + 2  s.t.  x1 + x2 == 1.
        qp_fsc = QuadraticModel(
            [1.0, 1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2, 2],
            Acols = [1, 2, 2, 3],
            Avals = [1.0, 1.0, 1.0, 1.0],
            lcon = [1.0, 2.0],
            ucon = [1.0, Inf],
            lvar = [0.0, 0.0, -Inf],
            uvar = [Inf, Inf, Inf],
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_fsc,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        red = res.reduced_model
        @test NLPModels.get_nvar(red) == 2
        @test NLPModels.get_ncon(red) == 1
        @test NLPModels.obj(red, [0.0, 1.0]) ≈ 2.0
        # optimal reduced solution: x = (0, 1), equality-row dual 0
        x, y = MadIPM.Models.Presolve.recover_solution(res, [0.0, 1.0], [0.0])
        @test x ≈ [0.0, 1.0, 1.0]  # x3 = (2 - x2)/1
        @test y ≈ [0.0, 1.0]       # y2 = c3/a23; (x, y) is a KKT point of qp_fsc
        @test presolve_kkt_ok(qp_fsc, x, y)

        # forcing row: min x1 + x2  s.t.  x1 + x2 >= 2,  0 <= x <= 1.
        # The bounds allow an activity of at most 2, so both variables are
        # pinned at their upper bound and nothing is left to solve.
        qp_force = QuadraticModel(
            [1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            lcon = [2.0],
            ucon = [Inf],
            lvar = [0.0, 0.0],
            uvar = [1.0, 1.0],
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_force,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_SOLVED
        @test res.objective_value ≈ 2.0
        x, y = MadIPM.Models.Presolve.recover_solution(res, Float64[], Float64[])
        @test x ≈ [1.0, 1.0]
        @test y ≈ [1.0]            # c - Aᵀy = 0: the row carries the whole dual
        @test presolve_kkt_ok(qp_force, x, y)

        # forcing row next to a surviving row; the pinned x2 also appears in
        # row 2, so that row's multiplier feeds back into the forcing-row one.
        # min x1 + x2 + 2 x3 + x4  s.t.  x1 + x2 >= 2,  x2 + x3 + x4 >= 1.5,  0 <= x <= 1
        qp_force2 = QuadraticModel(
            [1.0, 1.0, 2.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2, 2, 2],
            Acols = [1, 2, 2, 3, 4],
            Avals = [1.0, 1.0, 1.0, 1.0, 1.0],
            lcon = [2.0, 1.5],
            ucon = [Inf, Inf],
            lvar = zeros(4),
            uvar = ones(4),
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_force2,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        red = res.reduced_model
        @test NLPModels.get_nvar(red) == 2
        @test NLPModels.get_ncon(red) == 1
        @test NLPModels.get_lcon(red) == [0.5]        # 1.5 - x2
        @test NLPModels.obj(red, [0.0, 0.5]) ≈ 2.5    # 2 + 2*0 + 0.5
        # reduced optimum (x3, x4) = (0, 0.5) with row multiplier 1
        x, y = MadIPM.Models.Presolve.recover_solution(res, [0.0, 0.5], [1.0])
        @test x ≈ [1.0, 1.0, 0.0, 0.5]
        @test y ≈ [1.0, 1.0]
        @test presolve_kkt_ok(qp_force2, x, y)

        # two forcing rows in one pass; the second one sees x2 already pinned
        # min x1 + x2 + x3 + x4  s.t.  x1 + x2 >= 2,  x2 + x3 + x4 >= 3,  0 <= x <= 1
        qp_force3 = QuadraticModel(
            ones(4),
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2, 2, 2],
            Acols = [1, 2, 2, 3, 4],
            Avals = ones(5),
            lcon = [2.0, 3.0],
            ucon = [Inf, Inf],
            lvar = zeros(4),
            uvar = ones(4),
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_force3,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_SOLVED
        @test res.objective_value ≈ 4.0
        x, y = MadIPM.Models.Presolve.recover_solution(res, Float64[], Float64[])
        @test x ≈ ones(4)
        @test presolve_kkt_ok(qp_force3, x, y)

        # forcing row on a QP: min ½(x1² + x2²) - 3 x1  s.t.  x1 + x2 >= 2,  0 <= x <= 1
        qp_forceq = QuadraticModel(
            [-3.0, 0.0],
            [1, 2],
            [1, 2],
            [1.0, 1.0];
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            lcon = [2.0],
            ucon = [Inf],
            lvar = [0.0, 0.0],
            uvar = [1.0, 1.0],
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_forceq,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_SOLVED
        @test res.objective_value ≈ -2.0
        x, y = MadIPM.Models.Presolve.recover_solution(res, Float64[], Float64[])
        @test x ≈ [1.0, 1.0]
        @test y ≈ [1.0]            # reduced costs (-2, 1): the largest ratio wins
        @test presolve_kkt_ok(qp_forceq, x, y)

        # forcing row on a maximization: max -x1 - x2  s.t.  x1 + x2 >= 2,  0 <= x <= 1.
        # Same pins as qp_force; the multiplier keeps the file's c - Aᵀy
        # convention with the original cost, so it flips sign.
        qp_forcemax = QuadraticModel(
            [-1.0, -1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            lcon = [2.0],
            ucon = [Inf],
            lvar = [0.0, 0.0],
            uvar = [1.0, 1.0],
            minimize = false,
        )
        status, res = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_forcemax,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_SOLVED
        @test res.objective_value ≈ -2.0
        x, y = MadIPM.Models.Presolve.recover_solution(res, Float64[], Float64[])
        @test x ≈ [1.0, 1.0]
        @test y ≈ [-1.0]

        # redundant row (implied by the bounds) is dropped; a range row whose
        # lower side can never bind keeps only its upper side.
        # -1 <= x1 + x2 <= 3  and  -5 <= x1 - x2 <= 0.5  with  0 <= x <= 1
        qp_red = QuadraticModel(
            [1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1, 2, 2],
            Acols = [1, 2, 1, 2],
            Avals = [1.0, 1.0, 1.0, -1.0],
            lcon = [-1.0, -5.0],
            ucon = [3.0, 0.5],
            lvar = [0.0, 0.0],
            uvar = [1.0, 1.0],
        )
        red, status = MadIPM.presolve_qp(qp_red)
        @test status == MadIPM.Models.Presolve.PRESOLVE_REDUCED
        @test NLPModels.get_ncon(red) == 1
        @test NLPModels.get_lcon(red) == [-Inf]
        @test NLPModels.get_ucon(red) == [0.5]

        # activity bounds prove infeasibility: x1 + x2 >= 3 with x <= 1
        qp_inf = QuadraticModel(
            [1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            lcon = [3.0],
            ucon = [Inf],
            lvar = [0.0, 0.0],
            uvar = [1.0, 1.0],
        )
        status, _ = MadIPM.Models.Presolve.apply_presolve(
            MadIPM.Models.Presolve.BasicPresolver(),
            qp_inf,
        )
        @test status == MadIPM.Models.Presolve.PRESOLVE_INFEASIBLE
    end

    @testset "Certificate termination" begin
        infeas_lp = QuadraticModel(
            [0.0],
            Int[],
            Int[],
            Float64[];
            Arows = [1],
            Acols = [1],
            Avals = [1.0],
            lcon = [-1.0],
            ucon = [-1.0],
            lvar = [0.0],
            uvar = [Inf],
            x0 = [0.0],
        )
        infeas_solver = MadIPM.MPCSolver(infeas_lp; print_level=MadNLP.ERROR, scaling=false)
        @test infeas_solver.class isa MadIPM.LinearProgram
        MadIPM.initialize!(infeas_solver)
        infeas_solver.y .= 1.0
        infeas_solver.zl_r .= 1.0
        MadNLP.jtprod!(infeas_solver.jacl, infeas_solver.kkt, infeas_solver.y)
        @test MadIPM.has_primal_infeasibility_certificate(infeas_solver)
        MadIPM.update_termination_criteria!(infeas_solver)
        @test infeas_solver.status == MadNLP.INFEASIBLE_PROBLEM_DETECTED

        unbounded_lp = QuadraticModel(
            [-1.0],
            Int[],
            Int[],
            Float64[];
            lvar = [0.0],
            uvar = [Inf],
            x0 = [1.0],
        )
        unbounded_solver = MadIPM.MPCSolver(unbounded_lp; print_level=MadNLP.ERROR, scaling=false)
        @test unbounded_solver.class isa MadIPM.LinearProgram
        MadIPM.initialize!(unbounded_solver)
        MadNLP.primal(unbounded_solver.x) .= 1.0
        MadIPM.evaluate_model!(unbounded_solver)
        @test MadIPM.has_dual_infeasibility_certificate(unbounded_solver)
        MadIPM.update_termination_criteria!(unbounded_solver)
        @test unbounded_solver.status == MadNLP.DIVERGING_ITERATES
    end

    @testset "Standard formulation" begin
        new_qp = MadIPM.standard_form_qp(qp)
        solver = MadIPM.MPCSolver(new_qp; print_level = MadNLP.ERROR)
        sol = MadIPM.solve!(solver)
        @test sol.objective ≈ sol_ref.objective atol=1e-6
    end

    @testset "NormalKKTSystem implementation" begin
        # Test
        linear_solver = MadNLP.LapackCPUSolver
        cb = MadNLP.create_callback(MadNLP.SparseCallback, qp)
        kkt = MadNLP.create_kkt_system(MadIPM.NormalKKTSystem, cb, linear_solver;)
        MadNLPTests.test_kkt_system(kkt, cb)
    end

    @testset "Solve LP with NormalKKTSystem" begin
        solver = MadIPM.MPCSolver(
            qp;
            linear_solver = LDLSolver,
            print_level = MadNLP.ERROR,
            kkt_system = MadIPM.NormalKKTSystem,
            rethrow_error = true,
        )
        sol = MadIPM.solve!(solver)

        @test sol.status == MadNLP.SOLVE_SUCCEEDED
        @test sol.objective ≈ sol_ref.objective atol=1e-6
        @test sol.solution ≈ sol_ref.solution atol=1e-6
        @test sol.constraints ≈ sol_ref.constraints atol=1e-6
        @test sol.multipliers ≈ sol_ref.multipliers atol=1e-6
    end
end

@testset "Fixed variable with MakeParameter" begin
    solver = MadIPM.MPCSolver(
        QuadraticModel(
            [1.0, 1.0, 1.0],
            Int[],
            Int[],
            Float64[];
            lcon = [1.0],
            Arows = [1, 1],
            Acols = [1, 2],
            Avals = [1.0, 1.0],
            ucon = [Inf],
            lvar = [0.0, 0.0, 2.0],
            x0 = [1.0, 1.0, 1.0],
            uvar = [Inf, Inf, 2.0],
        );
        print_level = MadNLP.ERROR,
        fixed_variable_treatment = MadNLP.MakeParameter,
        rethrow_error = true,
    )
    sol = MadIPM.solve!(solver)
    @test sol.status == MadNLP.SOLVE_SUCCEEDED
    @test sol.solution[3] == 2.0
end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end

include("batch/views.jl")
include("batch/solver.jl")

if CUDA.functional()
    include("test_gpu.jl")
    include("batch/gpu.jl")
end
