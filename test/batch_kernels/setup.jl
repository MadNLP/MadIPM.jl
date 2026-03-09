using Test
using LinearAlgebra
using SparseArrays
using MadNLP
using MadIPM
using QuadraticModels
using QuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel

# ──────────────────────────────────────────────────────────────
# Test problems
# ──────────────────────────────────────────────────────────────

# LP: n=2, m=1, no upper bounds (nub=0)
function _setup_simple_lp()
    c = ones(2)
    Hrows = Int[]; Hcols = Int[]; Hvals = Float64[]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    QuadraticModel(c, Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[Inf, Inf],
        c0=0.0, x0=ones(2), name="simpleLP")
end

# QP: n=4, m=2, with finite upper bounds (nub>0)
function _setup_small_qp()
    n, m = 4, 2
    c = [1.0, -2.0, 0.5, 1.0]
    Hrows = [1, 2, 3, 4]; Hcols = [1, 2, 3, 4]; Hvals = [2.0, 1.0, 3.0, 1.5]
    Arows = [1, 1, 2, 2]; Acols = [1, 2, 3, 4]; Avals = [1.0, 1.0, 1.0, 1.0]
    QuadraticModel(c, Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[2.0, 1.5],
        lvar=zeros(n), uvar=fill(Inf, n), x0=ones(n))
end

# QP with only upper bounds (nlb=0, nub>0)
function _setup_upper_only_qp()
    QuadraticModel(
        [1.0, -1.0],
        [1, 2], [1, 2], [1.0, 1.0],
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[-Inf, -Inf], uvar=[5.0, 5.0],
        x0=[2.5, 2.5],
    )
end

# QP with doubly-bounded variables (both lvar and uvar finite)
# Regression test for has_inequalities bug where ind_llb/ind_uub are empty
# but nlb+nub > 0 (variables have bound multipliers on both sides)
function _setup_doubly_bounded_qp()
    QuadraticModel(
        [1.0, -1.0],
        [1, 2], [1, 2], [1.0, 1.0],
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[5.0, 5.0],
        x0=[2.5, 2.5],
    )
end

# Free-variable QP (nlb=0, nub=0, no bound multipliers)
function _setup_free_qp()
    QuadraticModel(
        [1.0, -1.0],
        [1, 2], [1, 2], [2.0, 2.0],
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[-Inf, -Inf], uvar=[Inf, Inf],
        x0=[0.5, 0.5],
    )
end

# QP with large coefficients that trigger non-unit scaling (obj_scale, con_scale < 1)
function _setup_scaled_qp()
    QuadraticModel(
        [500.0, -300.0, 400.0],
        [1, 2, 3], [1, 2, 3], [1.0, 1.0, 1.0],
        Arows=[1, 1, 1], Acols=[1, 2, 3], Avals=[200.0, 150.0, 100.0],
        lcon=[1.0], ucon=[1.0],
        lvar=zeros(3), uvar=fill(Inf, 3),
        x0=ones(3),
    )
end

# QP with large coefficients + inequality constraints (exercises slack scaling)
function _setup_scaled_ineq_qp()
    QuadraticModel(
        [500.0, -300.0, 400.0],
        [1, 2, 3], [1, 2, 3], [1.0, 1.0, 1.0],
        Arows=[1, 1, 1], Acols=[1, 2, 3], Avals=[200.0, 150.0, 100.0],
        lcon=[0.5], ucon=[2.0],
        lvar=zeros(3), uvar=fill(Inf, 3),
        x0=ones(3),
    )
end

# QP with a fixed variable (lvar[1]==uvar[1]) exercising MakeParameter
function _setup_fixed_var_qp()
    QuadraticModel(
        [1.0, -1.0, 0.5],
        [1, 2, 3], [1, 2, 3], [2.0, 1.0, 1.0],
        Arows=[1, 1, 1], Acols=[1, 2, 3], Avals=[1.0, 1.0, 1.0],
        lcon=[3.0], ucon=[3.0],
        lvar=[2.0, 0.0, 0.0], uvar=[2.0, Inf, Inf],
        x0=[2.0, 0.5, 0.5],
    )
end

# QP with all inequality constraints (ns == m, full slack initialization)
function _setup_all_ineq_qp()
    QuadraticModel(
        [1.0, -2.0, 0.5],
        [1, 2, 3], [1, 2, 3], [2.0, 1.0, 1.5],
        Arows=[1, 1, 2, 2], Acols=[1, 2, 2, 3], Avals=[1.0, 1.0, 1.0, 1.0],
        lcon=[0.0, 0.0], ucon=[3.0, 3.0],
        lvar=zeros(3), uvar=fill(Inf, 3),
        x0=ones(3),
    )
end

# QP with mixed bound types (lower-only, upper-only, doubly-bounded, free) + inequality
function _setup_mixed_bounds_qp()
    QuadraticModel(
        [1.0, -1.0, 0.5, -0.5],
        [1, 2, 3, 4], [1, 2, 3, 4], [2.0, 1.0, 1.5, 1.0],
        Arows=[1, 1, 1, 1, 2, 2], Acols=[1, 2, 3, 4, 1, 3],
        Avals=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        lcon=[1.0, 0.5], ucon=[3.0, 1.5],
        lvar=[0.0, -Inf, 0.0, -Inf], uvar=[Inf, 5.0, 10.0, Inf],
        x0=[1.0, 2.0, 5.0, 0.0],
    )
end

# QP with non-diagonal (dense lower-triangular) Hessian
# H = [4  .;  2  3] (lower triangular of symmetric PD matrix)
function _setup_dense_hess_qp()
    QuadraticModel(
        [1.0, -1.0],
        [1, 2, 2], [1, 1, 2], [4.0, 2.0, 3.0],
        Arows=[1, 1], Acols=[1, 2], Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0],
        lvar=[0.0, 0.0], uvar=[Inf, Inf],
        x0=[0.5, 0.5],
    )
end

# Larger QP with non-diagonal Hessian + inequality + mixed bounds
# H = [5 . .; 1 4 .; 0 2 3] (sparse lower-triangular, off-diagonal entries)
function _setup_dense_hess_mixed_qp()
    QuadraticModel(
        [1.0, -2.0, 0.5],
        [1, 2, 2, 3, 3], [1, 1, 2, 2, 3], [5.0, 1.0, 4.0, 2.0, 3.0],
        Arows=[1, 1, 1, 2, 2], Acols=[1, 2, 3, 1, 3],
        Avals=[1.0, 1.0, 1.0, 1.0, 1.0],
        lcon=[1.0, 0.5], ucon=[3.0, 1.5],
        lvar=[0.0, -Inf, 0.0], uvar=[Inf, 5.0, 10.0],
        x0=[1.0, 2.0, 1.0],
    )
end

# All test problems with descriptions
const ALL_TEST_PROBLEMS = [
    ("LP (nlb>0, nub=0)", _setup_simple_lp),
    ("QP (nlb>0, nub>0)", _setup_small_qp),
    ("QP (nlb=0, nub>0)", _setup_upper_only_qp),
    ("QP doubly-bounded", _setup_doubly_bounded_qp),
    ("QP free vars (nlb=0, nub=0)", _setup_free_qp),
    ("QP scaled", _setup_scaled_qp),
    ("QP scaled+ineq", _setup_scaled_ineq_qp),
    ("QP fixed var", _setup_fixed_var_qp),
    ("QP all-ineq", _setup_all_ineq_qp),
    ("QP mixed bounds", _setup_mixed_bounds_qp),
    ("QP dense Hessian", _setup_dense_hess_qp),
    ("QP dense Hessian+mixed", _setup_dense_hess_mixed_qp),
]

# ──────────────────────────────────────────────────────────────
# Paired test problems for batch_size=2 (same structure, different data)
# ──────────────────────────────────────────────────────────────

function _paired_lower_only()
    Hrows = Int[]; Hcols = Int[]; Hvals = Float64[]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp_a = QuadraticModel([1.0, 1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    qp_b = QuadraticModel([2.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.5, 0.5])
    return (qp_a, qp_b)
end

function _paired_doubly_bounded()
    Hrows = [1, 2]; Hcols = [1, 2]; Hvals = [1.0, 1.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [5.0, 5.0]
    qp_a = QuadraticModel([1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[2.5, 2.5])
    qp_b = QuadraticModel([-1.0, 2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[3.0], ucon=[3.0], lvar=lvar, uvar=uvar, x0=[1.5, 1.5])
    return (qp_a, qp_b)
end

function _paired_scaled()
    Hrows = [1, 2, 3]; Hcols = [1, 2, 3]; Hvals = [1.0, 1.0, 1.0]
    Arows = [1, 1, 1]; Acols = [1, 2, 3]; Avals = [200.0, 150.0, 100.0]
    lvar = zeros(3); uvar = fill(Inf, 3)
    qp_a = QuadraticModel([500.0, -300.0, 400.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=ones(3))
    qp_b = QuadraticModel([100.0, -200.0, 150.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5, 0.5])
    return (qp_a, qp_b)
end

function _paired_all_ineq()
    Hrows = [1, 2, 3]; Hcols = [1, 2, 3]; Hvals = [2.0, 1.0, 1.5]
    Arows = [1, 1, 2, 2]; Acols = [1, 2, 2, 3]; Avals = [1.0, 1.0, 1.0, 1.0]
    lvar = zeros(3); uvar = fill(Inf, 3)
    qp_a = QuadraticModel([1.0, -2.0, 0.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[0.0, 0.0], ucon=[3.0, 3.0], lvar=lvar, uvar=uvar, x0=ones(3))
    qp_b = QuadraticModel([2.0, -1.0, 1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[0.5, 0.5], ucon=[4.0, 4.0], lvar=lvar, uvar=uvar, x0=[0.5, 1.5, 1.0])
    return (qp_a, qp_b)
end

function _paired_mixed_bounds()
    Hrows = [1, 2, 3, 4]; Hcols = [1, 2, 3, 4]; Hvals = [2.0, 1.0, 1.5, 1.0]
    Arows = [1, 1, 1, 1, 2, 2]; Acols = [1, 2, 3, 4, 1, 3]
    Avals = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    lvar = [0.0, -Inf, 0.0, -Inf]; uvar = [Inf, 5.0, 10.0, Inf]
    qp_a = QuadraticModel([1.0, -1.0, 0.5, -0.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[3.0, 1.5], lvar=lvar, uvar=uvar,
        x0=[1.0, 2.0, 5.0, 0.0])
    qp_b = QuadraticModel([2.0, -2.0, 1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0, 1.0], ucon=[4.0, 2.0], lvar=lvar, uvar=uvar,
        x0=[0.5, 3.0, 4.0, 0.5])
    return (qp_a, qp_b)
end

function _paired_dense_hess()
    Hrows = [1, 2, 2]; Hcols = [1, 1, 2]; Hvals = [4.0, 2.0, 3.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp_a = QuadraticModel([1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    qp_b = QuadraticModel([-1.0, 2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    return (qp_a, qp_b)
end

function _paired_dense_hess_mixed()
    Hrows = [1, 2, 2, 3, 3]; Hcols = [1, 1, 2, 2, 3]; Hvals = [5.0, 1.0, 4.0, 2.0, 3.0]
    Arows = [1, 1, 1, 2, 2]; Acols = [1, 2, 3, 1, 3]
    Avals = [1.0, 1.0, 1.0, 1.0, 1.0]
    lvar = [0.0, -Inf, 0.0]; uvar = [Inf, 5.0, 10.0]
    qp_a = QuadraticModel([1.0, -2.0, 0.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0, 0.5], ucon=[3.0, 1.5], lvar=lvar, uvar=uvar,
        x0=[1.0, 2.0, 1.0])
    qp_b = QuadraticModel([2.0, -1.0, 1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0, 1.0], ucon=[4.0, 2.0], lvar=lvar, uvar=uvar,
        x0=[0.5, 1.5, 0.5])
    return (qp_a, qp_b)
end

const PAIRED_PROBLEMS = [
    ("paired lower-only LP", _paired_lower_only),
    ("paired doubly-bounded QP", _paired_doubly_bounded),
    ("paired scaled QP", _paired_scaled),
    ("paired all-ineq QP", _paired_all_ineq),
    ("paired mixed-bounds QP", _paired_mixed_bounds),
    ("paired dense-hess QP", _paired_dense_hess),
    ("paired dense-hess+mixed QP", _paired_dense_hess_mixed),
]

# Staggered convergence: easy QP (converges fast) + harder QP (converges slow)
# Same Hessian, Jacobian (ObjRHSBatch shares these). Differ only in c, lcon/ucon, x0.
function _paired_staggered()
    Hrows = [1, 2]; Hcols = [1, 2]; Hvals = [1.0, 1.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    # Easy QP: symmetric objective, tight constraint → fast convergence
    qp_easy = QuadraticModel([1.0, 1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    # Harder QP: asymmetric objective, looser constraint → more iterations
    qp_hard = QuadraticModel([0.01, -0.99], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[0.5], ucon=[0.5], lvar=lvar, uvar=uvar, x0=[5.0, 5.0])
    return (qp_easy, qp_hard)
end

# Quad problems for batch_size=4 tests (all lower-bounded LPs, same structure)
function _quad_lower_only()
    Hrows = Int[]; Hcols = Int[]; Hvals = Float64[]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp1 = QuadraticModel([1.0, 1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    qp2 = QuadraticModel([2.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.5, 0.5])
    qp3 = QuadraticModel([0.5, 1.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.5], ucon=[1.5], lvar=lvar, uvar=uvar, x0=[0.5, 1.0])
    qp4 = QuadraticModel([-1.0, 3.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[0.5], ucon=[0.5], lvar=lvar, uvar=uvar, x0=[0.3, 0.2])
    return (qp1, qp2, qp3, qp4)
end

# Quad problems for batch_size=4 with doubly-bounded QPs
function _quad_doubly_bounded()
    Hrows = [1, 2]; Hcols = [1, 2]; Hvals = [1.0, 1.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [5.0, 5.0]
    qp1 = QuadraticModel([1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[2.5, 2.5])
    qp2 = QuadraticModel([-1.0, 2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[3.0], ucon=[3.0], lvar=lvar, uvar=uvar, x0=[1.5, 1.5])
    qp3 = QuadraticModel([0.5, 0.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    qp4 = QuadraticModel([2.0, -2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[4.0], ucon=[4.0], lvar=lvar, uvar=uvar, x0=[2.0, 2.0])
    return (qp1, qp2, qp3, qp4)
end

# Quad problems for batch_size=4 with non-diagonal Hessian QPs
function _quad_dense_hess()
    Hrows = [1, 2, 2]; Hcols = [1, 1, 2]; Hvals = [4.0, 2.0, 3.0]
    Arows = [1, 1]; Acols = [1, 2]; Avals = [1.0, 1.0]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp1 = QuadraticModel([1.0, -1.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    qp2 = QuadraticModel([-1.0, 2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    qp3 = QuadraticModel([0.5, 0.5], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[1.5], ucon=[1.5], lvar=lvar, uvar=uvar, x0=[0.75, 0.75])
    qp4 = QuadraticModel([2.0, -2.0], Hrows, Hcols, Hvals;
        Arows=Arows, Acols=Acols, Avals=Avals,
        lcon=[0.5], ucon=[0.5], lvar=lvar, uvar=uvar, x0=[0.3, 0.2])
    return (qp1, qp2, qp3, qp4)
end

# ──────────────────────────────────────────────────────────────
# Build initialized sequential solver
# ──────────────────────────────────────────────────────────────
function build_seq(qp; kwargs...)
    solver = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, kwargs...)
    opt = solver.opt

    MadNLP.initialize!(solver.cb, solver.x, solver.xl, solver.xu, solver.y, solver.rhs, solver.ind_ineq;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(solver.jacl, 0.0)
    if opt.scaling
        MadNLP.set_scaling!(solver.cb, solver.x, solver.xl, solver.xu, solver.y, solver.rhs,
            solver.ind_ineq, Float64(opt.nlp_scaling_max_gradient))
    end
    MadNLP.initialize!(solver.kkt)
    MadIPM.init_regularization!(solver, opt.regularization)

    solver.obj_val = MadNLP.eval_f_wrapper(solver, solver.x)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt, solver.x)
    MadNLP.eval_grad_f_wrapper!(solver, solver.f, solver.x)
    MadNLP.eval_cons_wrapper!(solver, solver.c, solver.x)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt, solver.x, solver.y)
    solver.norm_b = norm(solver.rhs, Inf)
    solver.norm_c = norm(MadNLP.primal(solver.f), Inf)

    MadIPM.init_starting_point!(solver)
    solver.mu = opt.mu_init
    solver.cnt.start_time = time()
    solver.best_complementarity = typemax(Float64)
    solver.status = MadNLP.REGULAR
    MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    return solver
end

# ──────────────────────────────────────────────────────────────
# Build initialized batch solver (bs=1)
# ──────────────────────────────────────────────────────────────
function build_batch(qp; kwargs...)
    bnlp = ObjRHSBatchQuadraticModel([qp])
    batch_solver = MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR, kwargs...)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    opt = batch_solver.opt

    MadNLP.initialize!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
        MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
        batch_solver.workspace.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(batch_solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
            MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
            Float64(opt.nlp_scaling_max_gradient),
            batch_solver.workspace.bx)
    end
    MadNLP.initialize!(batch_solver.kkt)
    MadIPM.init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    MadIPM.init_starting_point!(batch_solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0); fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0); fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0); fill!(ws.alpha_d, 0.0)
    batch_solver.batch_cnt.start_time[] = time()
    fill!(batch_solver.batch_cnt.k, 0)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return batch_solver
end

# ──────────────────────────────────────────────────────────────
# Build initialized batch solver with batch_size > 1
# ──────────────────────────────────────────────────────────────
function build_batch_n(qp, n::Int)
    bnlp = ObjRHSBatchQuadraticModel([qp for _ in 1:n])
    batch_solver = MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    opt = batch_solver.opt

    MadNLP.initialize!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
        MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
        batch_solver.workspace.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(batch_solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
            MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
            Float64(opt.nlp_scaling_max_gradient),
            batch_solver.workspace.bx)
    end
    MadNLP.initialize!(batch_solver.kkt)
    MadIPM.init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    MadIPM.init_starting_point!(batch_solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0); fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0); fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0); fill!(ws.alpha_d, 0.0)
    batch_solver.batch_cnt.start_time[] = time()
    fill!(batch_solver.batch_cnt.k, 0)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return batch_solver
end

# ──────────────────────────────────────────────────────────────
# Comparison helper: max absolute difference
# ──────────────────────────────────────────────────────────────
# Safe comparison that handles Inf values (e.g. xu with uvar=Inf)
function cmp(a, b)
    d = 0.0
    for (ai, bi) in zip(a, b)
        if isfinite(ai) && isfinite(bi)
            d = max(d, abs(ai - bi) / max(abs(ai), abs(bi), 1.0))
        elseif ai !== bi
            return Inf
        end
    end
    return d
end

# Extract column 1 from a batch matrix/vector
col1(x::AbstractMatrix) = view(x, :, 1)
col1(x::AbstractVector) = x  # already a vector (scalar workspace)

# ──────────────────────────────────────────────────────────────
# Run first factorize_system to get post-factorization state
# ──────────────────────────────────────────────────────────────
function do_first_factorize!(seq, bat)
    # Sequential
    MadIPM.update_regularization!(seq, seq.opt.regularization)
    MadIPM.set_aug_diagonal_reg!(seq.kkt, seq)
    MadNLP.build_kkt!(seq.kkt)
    MadNLP.factorize_kkt!(seq.kkt)
    # Batch
    MadIPM.update_regularization!(bat, bat.opt.regularization)
    MadIPM.set_aug_diagonal_reg!(bat.kkt, bat)
    MadNLP.build_kkt!(bat.kkt)
    MadNLP.factorize_kkt!(bat.kkt)
end

# ──────────────────────────────────────────────────────────────
# Build initialized batch solver from multiple (potentially different) QPs
# ──────────────────────────────────────────────────────────────
function build_batch_from_qps(qps::Vector; kwargs...)
    bnlp = ObjRHSBatchQuadraticModel(qps)
    batch_solver = MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR, kwargs...)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    opt = batch_solver.opt

    MadNLP.initialize!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
        MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
        batch_solver.workspace.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(batch_solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
            MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
            Float64(opt.nlp_scaling_max_gradient),
            batch_solver.workspace.bx)
    end
    MadNLP.initialize!(batch_solver.kkt)
    MadIPM.init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    MadIPM.init_starting_point!(batch_solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0); fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0); fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0); fill!(ws.alpha_d, 0.0)
    batch_solver.batch_cnt.start_time[] = time()
    fill!(batch_solver.batch_cnt.k, 0)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return batch_solver
end

# ──────────────────────────────────────────────────────────────
# Build initialized batch solver using BatchQuadraticModel (bs=1)
# ──────────────────────────────────────────────────────────────
function build_fullbatch(qp; kwargs...)
    bnlp = BatchQuadraticModel([qp])
    batch_solver = MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR, kwargs...)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    opt = batch_solver.opt

    MadNLP.initialize!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
        MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
        batch_solver.workspace.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(batch_solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
            MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
            Float64(opt.nlp_scaling_max_gradient),
            batch_solver.workspace.bx)
    end
    MadNLP.initialize!(batch_solver.kkt)
    MadIPM.init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    MadIPM.init_starting_point!(batch_solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0); fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0); fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0); fill!(ws.alpha_d, 0.0)
    batch_solver.batch_cnt.start_time[] = time()
    fill!(batch_solver.batch_cnt.k, 0)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return batch_solver
end

# ──────────────────────────────────────────────────────────────
# Build initialized batch solver from multiple QPs using BatchQuadraticModel
# ──────────────────────────────────────────────────────────────
function build_fullbatch_from_qps(qps::Vector; kwargs...)
    bnlp = BatchQuadraticModel(qps)
    batch_solver = MadIPM.UniformBatchMPCSolver(bnlp; print_level=MadNLP.ERROR, kwargs...)
    ws = batch_solver.workspace
    bcb = batch_solver.bcb
    opt = batch_solver.opt

    MadNLP.initialize!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
        MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
        batch_solver.workspace.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(batch_solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, batch_solver.x, batch_solver.xl, batch_solver.xu,
            MadNLP.full(batch_solver.y), MadNLP.full(batch_solver.rhs), bcb.ind_ineq,
            Float64(opt.nlp_scaling_max_gradient),
            batch_solver.workspace.bx)
    end
    MadNLP.initialize!(batch_solver.kkt)
    MadIPM.init_regularization!(batch_solver, opt.regularization)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    MadNLP.eval_f_wrapper(batch_solver, ws.bx)
    MadNLP.eval_jac_wrapper!(batch_solver, batch_solver.kkt)
    MadNLP.eval_grad_f_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_cons_wrapper!(batch_solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(batch_solver, batch_solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(batch_solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(batch_solver.f); dims=1)

    MadIPM.init_starting_point!(batch_solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0); fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0); fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0); fill!(ws.alpha_d, 0.0)
    batch_solver.batch_cnt.start_time[] = time()
    fill!(batch_solver.batch_cnt.k, 0)
    MadNLP.jtprod!(batch_solver.jacl, batch_solver.kkt, batch_solver.y)
    return batch_solver
end

# ──────────────────────────────────────────────────────────────
# Paired problems for BatchQuadraticModel: DIFFERENT H and A values
# (same sparsity pattern, same bound pattern, different everything else)
# ──────────────────────────────────────────────────────────────

# Diagonal Hessian + different A values, equality constraint
function _fullbatch_paired_diagonal()
    Hrows = [1, 2]; Hcols = [1, 2]
    Arows = [1, 1]; Acols = [1, 2]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp_a = QuadraticModel([1.0, -1.0], Hrows, Hcols, [2.0, 1.0];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    qp_b = QuadraticModel([-1.0, 2.0], Hrows, Hcols, [3.0, 2.0];
        Arows=Arows, Acols=Acols, Avals=[2.0, 0.5],
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    return (qp_a, qp_b)
end

# Off-diagonal Hessian + different H and A values, equality constraint
function _fullbatch_paired_dense_hess()
    Hrows = [1, 2, 2]; Hcols = [1, 1, 2]
    Arows = [1, 1]; Acols = [1, 2]
    lvar = [0.0, 0.0]; uvar = [Inf, Inf]
    qp_a = QuadraticModel([1.0, -1.0], Hrows, Hcols, [4.0, 2.0, 3.0];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5])
    qp_b = QuadraticModel([-1.0, 2.0], Hrows, Hcols, [6.0, 1.0, 5.0];
        Arows=Arows, Acols=Acols, Avals=[1.5, 0.5],
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[1.0, 1.0])
    return (qp_a, qp_b)
end

# Different H and A values, doubly-bounded variables
function _fullbatch_paired_doubly_bounded()
    Hrows = [1, 2]; Hcols = [1, 2]
    Arows = [1, 1]; Acols = [1, 2]
    lvar = [0.0, 0.0]; uvar = [5.0, 5.0]
    qp_a = QuadraticModel([1.0, -1.0], Hrows, Hcols, [2.0, 1.0];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0],
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=[2.5, 2.5])
    qp_b = QuadraticModel([-0.5, 1.5], Hrows, Hcols, [3.0, 2.5];
        Arows=Arows, Acols=Acols, Avals=[2.0, 1.0],
        lcon=[3.0], ucon=[3.0], lvar=lvar, uvar=uvar, x0=[1.5, 1.5])
    return (qp_a, qp_b)
end

# Different H and A values, inequality constraints (exercises different scaling per instance)
function _fullbatch_paired_inequality()
    Hrows = [1, 2, 3]; Hcols = [1, 2, 3]
    Arows = [1, 1, 2, 2]; Acols = [1, 2, 2, 3]
    lvar = zeros(3); uvar = fill(Inf, 3)
    qp_a = QuadraticModel([1.0, -2.0, 0.5], Hrows, Hcols, [2.0, 1.0, 1.5];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0, 1.0, 1.0],
        lcon=[0.0, 0.0], ucon=[3.0, 3.0], lvar=lvar, uvar=uvar, x0=ones(3))
    qp_b = QuadraticModel([2.0, -1.0, 1.0], Hrows, Hcols, [3.0, 2.0, 1.0];
        Arows=Arows, Acols=Acols, Avals=[2.0, 0.5, 1.5, 0.5],
        lcon=[0.5, 0.5], ucon=[4.0, 4.0], lvar=lvar, uvar=uvar, x0=[0.5, 1.5, 1.0])
    return (qp_a, qp_b)
end

# Mixed bounds + different H and A values + off-diagonal Hessian
function _fullbatch_paired_mixed()
    Hrows = [1, 2, 2, 3, 3]; Hcols = [1, 1, 2, 2, 3]
    Arows = [1, 1, 1, 2, 2]; Acols = [1, 2, 3, 1, 3]
    lvar = [0.0, -Inf, 0.0]; uvar = [Inf, 5.0, 10.0]
    qp_a = QuadraticModel([1.0, -2.0, 0.5], Hrows, Hcols, [5.0, 1.0, 4.0, 2.0, 3.0];
        Arows=Arows, Acols=Acols, Avals=[1.0, 1.0, 1.0, 1.0, 1.0],
        lcon=[1.0, 0.5], ucon=[3.0, 1.5], lvar=lvar, uvar=uvar,
        x0=[1.0, 2.0, 1.0])
    qp_b = QuadraticModel([2.0, -1.0, 1.0], Hrows, Hcols, [7.0, 2.0, 5.0, 1.0, 4.0];
        Arows=Arows, Acols=Acols, Avals=[2.0, 0.5, 1.0, 1.5, 0.5],
        lcon=[2.0, 1.0], ucon=[4.0, 2.0], lvar=lvar, uvar=uvar,
        x0=[0.5, 1.5, 0.5])
    return (qp_a, qp_b)
end

# Large coefficients triggering scaling + different H and A values
function _fullbatch_paired_scaled()
    Hrows = [1, 2, 3]; Hcols = [1, 2, 3]
    Arows = [1, 1, 1]; Acols = [1, 2, 3]
    lvar = zeros(3); uvar = fill(Inf, 3)
    qp_a = QuadraticModel([500.0, -300.0, 400.0], Hrows, Hcols, [1.0, 1.0, 1.0];
        Arows=Arows, Acols=Acols, Avals=[200.0, 150.0, 100.0],
        lcon=[1.0], ucon=[1.0], lvar=lvar, uvar=uvar, x0=ones(3))
    qp_b = QuadraticModel([100.0, -200.0, 150.0], Hrows, Hcols, [2.0, 3.0, 1.5];
        Arows=Arows, Acols=Acols, Avals=[300.0, 50.0, 50.0],
        lcon=[2.0], ucon=[2.0], lvar=lvar, uvar=uvar, x0=[0.5, 0.5, 0.5])
    return (qp_a, qp_b)
end

const FULLBATCH_PAIRED_PROBLEMS = [
    ("fullbatch diagonal", _fullbatch_paired_diagonal),
    ("fullbatch dense Hessian", _fullbatch_paired_dense_hess),
    ("fullbatch doubly-bounded", _fullbatch_paired_doubly_bounded),
    ("fullbatch inequality", _fullbatch_paired_inequality),
    ("fullbatch mixed bounds", _fullbatch_paired_mixed),
    ("fullbatch scaled", _fullbatch_paired_scaled),
]

# ──────────────────────────────────────────────────────────────
# Extract column n from a batch matrix/vector
# ──────────────────────────────────────────────────────────────
coln(x::AbstractMatrix, n::Int) = view(x, :, n)
coln(x::AbstractVector, ::Int) = x

# ──────────────────────────────────────────────────────────────
# Assertion helpers: compare sequential solver state against batch column
# ──────────────────────────────────────────────────────────────

function assert_termination_match(seq, bat, col; tol=1e-10)
    @test abs(seq.inf_pr - bat.workspace.inf_pr[col]) < tol
    @test abs(seq.inf_du - bat.workspace.inf_du[col]) < tol
    @test abs(seq.inf_compl - bat.workspace.inf_compl[col]) < tol
    @test seq.status == bat.workspace.status[col]
    # dual_obj: used in infeasibility detection
    seq_dobj = MadIPM.dual_objective(seq)
    @test abs(seq_dobj - bat.workspace.dual_obj[col]) < tol
end

function assert_regularization_match(seq, bat, col; tol=1e-10)
    @test abs(seq.del_w - bat.del_w[col]) < tol
    @test abs(seq.del_c - bat.del_c[col]) < tol
end

function assert_prediction_match(seq, bat, col; tol=1e-10)
    @test cmp(MadNLP.full(seq.d), coln(MadNLP.full(bat.d), col)) < tol
    @test abs(seq.mu - bat.workspace.mu_batch[col]) < tol
    @test cmp(seq.correction_lb, coln(MadNLP.full(bat.correction_lb), col)) < tol
    @test cmp(seq.correction_ub, coln(MadNLP.full(bat.correction_ub), col)) < tol
end

function assert_direction_match(seq, bat, col; tol=1e-10)
    @test cmp(MadNLP.full(seq.d), coln(MadNLP.full(bat.d), col)) < tol
end

function assert_step_match(seq, bat, col; tol=1e-10)
    @test abs(seq.alpha_p - bat.workspace.alpha_p[col]) < tol
    @test abs(seq.alpha_d - bat.workspace.alpha_d[col]) < tol
end

function assert_iterate_match(seq, bat, col; tol=1e-10)
    @test cmp(MadNLP.full(seq.x), coln(MadNLP.full(bat.x), col)) < tol
    @test cmp(seq.y, coln(MadNLP.full(bat.y), col)) < tol
    @test cmp(MadNLP.full(seq.zl), coln(MadNLP.full(bat.zl), col)) < tol
    @test cmp(MadNLP.full(seq.zu), coln(MadNLP.full(bat.zu), col)) < tol
    @test cmp(MadNLP.full(seq.xl), coln(MadNLP.full(bat.xl), col)) < tol
    @test cmp(MadNLP.full(seq.xu), coln(MadNLP.full(bat.xu), col)) < tol
end

function assert_model_match(seq, bat, col; tol=1e-10)
    @test abs(seq.obj_val - bat.workspace.obj_val[col]) < tol
    @test cmp(MadNLP.primal(seq.f), coln(MadNLP.primal(bat.f), col)) < tol
    @test cmp(seq.c, coln(MadNLP.full(bat.c), col)) < tol
    @test cmp(seq.jacl, coln(MadNLP.full(bat.jacl), col)) < tol
end

function assert_kkt_diagonals_match(seq, bat, col; tol=1e-10)
    skkt = seq.kkt
    bkkt = bat.kkt
    # l_diag, u_diag, l_lower, u_lower
    if length(skkt.l_diag) > 0
        @test cmp(skkt.l_diag, coln(bkkt.l_diag, col)) < tol
        @test cmp(skkt.l_lower, coln(bkkt.l_lower, col)) < tol
    end
    if length(skkt.u_diag) > 0
        @test cmp(skkt.u_diag, coln(bkkt.u_diag, col)) < tol
        @test cmp(skkt.u_lower, coln(bkkt.u_lower, col)) < tol
    end
    # reg (primal regularization)
    @test cmp(skkt.reg, coln(bkkt.reg, col)) < tol
    # pr_diag (includes reg + bound contributions)
    @test cmp(skkt.pr_diag, coln(MadIPM.pr_diag(bkkt), col)) < tol
    # du_diag (dual regularization)
    @test cmp(skkt.du_diag, coln(MadIPM.du_diag(bkkt), col)) < tol
end

function assert_init_match(seq, bat, col; tol=1e-10)
    # Full state after init_starting_point! + initialization
    @test cmp(MadNLP.full(seq.x), coln(MadNLP.full(bat.x), col)) < tol
    @test cmp(seq.y, coln(MadNLP.full(bat.y), col)) < tol
    @test cmp(MadNLP.full(seq.zl), coln(MadNLP.full(bat.zl), col)) < tol
    @test cmp(MadNLP.full(seq.zu), coln(MadNLP.full(bat.zu), col)) < tol
    @test cmp(MadNLP.full(seq.xl), coln(MadNLP.full(bat.xl), col)) < tol
    @test cmp(MadNLP.full(seq.xu), coln(MadNLP.full(bat.xu), col)) < tol
    @test abs(seq.obj_val - bat.workspace.obj_val[col]) < tol
    @test cmp(MadNLP.primal(seq.f), coln(MadNLP.primal(bat.f), col)) < tol
    @test cmp(seq.c, coln(MadNLP.full(bat.c), col)) < tol
    @test cmp(seq.jacl, coln(MadNLP.full(bat.jacl), col)) < tol
    # Regularization state
    @test abs(seq.del_w - bat.del_w[col]) < tol
    @test abs(seq.del_c - bat.del_c[col]) < tol
    # Barrier parameter
    @test abs(seq.mu - bat.workspace.mu_batch[col]) < tol
    # Normalization constants
    @test abs(seq.norm_b - bat.workspace.norm_b[col]) < tol
    @test abs(seq.norm_c - bat.workspace.norm_c[col]) < tol
end

function assert_correction_match(seq, bat, col; tol=1e-10)
    @test cmp(seq.correction_lb, coln(MadNLP.full(bat.correction_lb), col)) < tol
    @test cmp(seq.correction_ub, coln(MadNLP.full(bat.correction_ub), col)) < tol
end

function assert_kkt_matrix_match(seq, bat, col; tol=1e-10)
    skkt = seq.kkt
    bkkt = bat.kkt
    @test cmp(SparseArrays.nonzeros(skkt.aug_com), coln(bkkt.aug_com_nzvals, col)) < tol
end

function assert_barrier_match(seq, bat, col; mu_affine_seq=nothing, tol=1e-10)
    # mu_curr: complementarity measure (set in update_barrier! → get_complementarity_measure)
    @test abs(seq.mu_curr - bat.workspace.mu_curr[col]) < tol
    # mu_batch (= sigma * mu_curr): final barrier parameter
    @test abs(seq.mu - bat.workspace.mu_batch[col]) < tol
    # mu_affine: affine complementarity measure (if provided)
    if mu_affine_seq !== nothing
        @test abs(mu_affine_seq - bat.workspace.mu_affine[col]) < tol
    end
end

function assert_tau_match(seq_tau, bat, col; tol=1e-10)
    @test abs(seq_tau - bat.workspace.tau[col]) < tol
end

function assert_rhs_match(seq, bat, col; tol=1e-10)
    @test cmp(MadNLP.full(seq.p), coln(MadNLP.full(bat.p), col)) < tol
end

function assert_reduce_rhs_match(seq, bat, col; tol=1e-10)
    # After reduce_rhs!, the primal-dual part of d has been modified.
    # The dual_lb and dual_ub parts contain the original RHS values
    # that will be used in finish_aug_solve!
    sd = MadNLP.full(seq.d)
    bd = coln(MadNLP.full(bat.d), col)
    @test cmp(sd, bd) < tol
end

function assert_finish_aug_solve_match(seq, bat, col; tol=1e-10)
    # After finish_aug_solve!, dzl and dzu are computed from the linear solve result
    @test cmp(MadNLP.dual_lb(seq.d), coln(MadNLP.dual_lb(bat.d), col)) < tol
    @test cmp(MadNLP.dual_ub(seq.d), coln(MadNLP.dual_ub(bat.d), col)) < tol
end

# ──────────────────────────────────────────────────────────────
# Patching: copy sequential solver state into a batch column
# (test-only; prevents FP noise from accumulating across iterations)
# ──────────────────────────────────────────────────────────────

function patch_batch_col_from_seq!(seq, bat, col)
    ws = bat.workspace

    # Iterate: x (includes slacks), xl, xu, y, zl, zu
    coln(MadNLP.full(bat.x), col) .= MadNLP.full(seq.x)
    coln(MadNLP.full(bat.xl), col) .= MadNLP.full(seq.xl)
    coln(MadNLP.full(bat.xu), col) .= MadNLP.full(seq.xu)
    coln(MadNLP.full(bat.y), col) .= seq.y
    coln(MadNLP.full(bat.zl), col) .= MadNLP.full(seq.zl)
    coln(MadNLP.full(bat.zu), col) .= MadNLP.full(seq.zu)

    # Model evaluations: gradient (primal part), constraints, jacl, obj
    coln(MadNLP.primal(bat.f), col) .= MadNLP.primal(seq.f)
    coln(MadNLP.full(bat.c), col) .= seq.c
    coln(MadNLP.full(bat.jacl), col) .= seq.jacl
    ws.obj_val[col] = seq.obj_val

    # Barrier / termination scalars
    ws.mu_batch[col] = seq.mu
    ws.best_complementarity[col] = seq.best_complementarity
end

# ──────────────────────────────────────────────────────────────
# Iteration harness: step sequential + batch solvers in lockstep
# ──────────────────────────────────────────────────────────────

function run_iterations_bs1!(seq, bat, n_iters; tol=1e-10)
    ws = bat.workspace
    for iter in 1:n_iters
        MadIPM.update_termination_criteria!(seq)
        MadIPM.update_termination_criteria!(bat)
        assert_termination_match(seq, bat, 1; tol)

        MadIPM.is_done(seq) && break
        MadIPM.update_active_set!(bat.kkt, ws.status)
        bat.kkt.active_batch_size[] == 0 && break
        MadIPM._update_active_mask!(bat)

        # Factorize
        MadIPM.factorize_system!(seq)
        MadIPM.factorize_system!(bat)
        assert_regularization_match(seq, bat, 1; tol)
        assert_kkt_diagonals_match(seq, bat, 1; tol)
        assert_kkt_matrix_match(seq, bat, 1; tol)

        # Prediction step (decomposed for finer-grained checks)
        # 1. Set predictive RHS
        MadIPM.set_predictive_rhs!(seq, seq.kkt)
        MadIPM.set_predictive_rhs!(bat, bat.kkt)
        assert_rhs_match(seq, bat, 1; tol)
        # 2. Solve system (includes reduce_rhs! → linear solve → finish_aug_solve!)
        MadIPM.solve_system!(seq.d, seq, seq.p)
        MadIPM.solve_system!(bat.d, bat, bat.p)
        assert_direction_match(seq, bat, 1; tol)
        assert_finish_aug_solve_match(seq, bat, 1; tol)
        # 3. Affine step sizes (tau=1) + barrier update
        alpha_aff_p, alpha_aff_d = MadIPM.get_fraction_to_boundary_step(seq, 1.0)
        fill!(ws.tau, one(eltype(ws.tau)))
        MadIPM.get_fraction_to_boundary_step!(bat)
        MadIPM.zero_inactive_step!(bat)
        mu_affine_seq = MadIPM.get_affine_complementarity_measure(seq, alpha_aff_p, alpha_aff_d)
        MadIPM.get_affine_complementarity_measure!(bat, ws.alpha_p, ws.alpha_d)
        MadIPM.get_correction!(seq, seq.correction_lb, seq.correction_ub)
        MadIPM.get_correction!(bat, MadNLP.full(bat.correction_lb), MadNLP.full(bat.correction_ub))
        assert_correction_match(seq, bat, 1; tol)
        seq.mu_curr = MadIPM.update_barrier!(seq.opt.barrier_update, seq, mu_affine_seq)
        MadIPM.update_barrier!(bat.opt.barrier_update, bat, ws.mu_affine)
        assert_prediction_match(seq, bat, 1; tol)
        assert_barrier_match(seq, bat, 1; mu_affine_seq, tol)

        # Mehrotra correction
        MadIPM.set_correction_rhs!(seq, seq.kkt, seq.mu, seq.correction_lb, seq.correction_ub, seq.ind_lb, seq.ind_ub)
        MadIPM.set_correction_rhs!(bat, bat.kkt, ws.mu_batch, MadNLP.full(bat.correction_lb), MadNLP.full(bat.correction_ub), nothing, nothing)
        assert_rhs_match(seq, bat, 1; tol)
        MadIPM.solve_system!(seq.d, seq, seq.p)
        MadIPM.solve_system!(bat.d, bat, bat.p)
        assert_direction_match(seq, bat, 1; tol)
        assert_finish_aug_solve_match(seq, bat, 1; tol)

        # Update step (decomposed to check tau)
        MadIPM.update_step!(seq.opt.step_rule, seq)
        MadIPM.update_step!(bat.opt.step_rule, bat)
        MadIPM.zero_inactive_step!(bat)
        # Check tau: compute sequential tau from step rule
        seq_tau = if seq.opt.step_rule isa MadIPM.ConservativeStep
            seq.opt.step_rule.tau
        elseif seq.opt.step_rule isa MadIPM.AdaptiveStep
            max(1 - seq.mu, seq.opt.step_rule.tau_min)
        else
            1.0  # MehrotraAdaptiveStep uses tau=1.0 internally
        end
        assert_tau_match(seq_tau, bat, 1; tol)
        assert_step_match(seq, bat, 1; tol)

        # Apply step
        MadIPM.apply_step!(seq)
        MadIPM.apply_step!(bat)
        assert_iterate_match(seq, bat, 1; tol)

        # Evaluate model
        MadIPM.evaluate_model!(seq)
        MadIPM.evaluate_model!(bat)
        assert_model_match(seq, bat, 1; tol)
    end
    @test seq.status == bat.workspace.status[1]
    @test seq.status == MadNLP.SOLVE_SUCCEEDED
end

function run_iterations_bs2!(seq1, seq2, bat, n_iters; tol=1e-10, patch=false)
    seq1_done = false
    seq2_done = false

    # Patch initial state to eliminate init FP seed differences
    if patch
        patch_batch_col_from_seq!(seq1, bat, 1)
        patch_batch_col_from_seq!(seq2, bat, 2)
    end

    for iter in 1:n_iters
        if !seq1_done; MadIPM.update_termination_criteria!(seq1); end
        if !seq2_done; MadIPM.update_termination_criteria!(seq2); end
        MadIPM.update_termination_criteria!(bat)

        if !seq1_done; assert_termination_match(seq1, bat, 1; tol); end
        if !seq2_done; assert_termination_match(seq2, bat, 2; tol); end

        seq1_done = seq1_done || MadIPM.is_done(seq1)
        seq2_done = seq2_done || MadIPM.is_done(seq2)

        MadIPM.update_active_set!(bat.kkt, bat.workspace.status)
        bat.kkt.active_batch_size[] == 0 && break
        MadIPM._update_active_mask!(bat)

        (seq1_done && seq2_done) && break

        # Factorize
        if !seq1_done; MadIPM.factorize_system!(seq1); end
        if !seq2_done; MadIPM.factorize_system!(seq2); end
        MadIPM.factorize_system!(bat)
        if !seq1_done
            assert_regularization_match(seq1, bat, 1; tol)
            assert_kkt_diagonals_match(seq1, bat, 1; tol)
        end
        if !seq2_done
            assert_regularization_match(seq2, bat, 2; tol)
            assert_kkt_diagonals_match(seq2, bat, 2; tol)
        end

        # Prediction step
        if !seq1_done; MadIPM.prediction_step!(seq1); end
        if !seq2_done; MadIPM.prediction_step!(seq2); end
        MadIPM.prediction_step!(bat)
        if !seq1_done
            assert_prediction_match(seq1, bat, 1; tol)
            assert_barrier_match(seq1, bat, 1; tol)
        end
        if !seq2_done
            assert_prediction_match(seq2, bat, 2; tol)
            assert_barrier_match(seq2, bat, 2; tol)
        end

        # Mehrotra correction
        if !seq1_done; MadIPM.mehrotra_correction_direction!(seq1); end
        if !seq2_done; MadIPM.mehrotra_correction_direction!(seq2); end
        MadIPM.mehrotra_correction_direction!(bat)
        if !seq1_done; assert_direction_match(seq1, bat, 1; tol); end
        if !seq2_done; assert_direction_match(seq2, bat, 2; tol); end

        # Update step
        if !seq1_done; MadIPM.update_step!(seq1.opt.step_rule, seq1); end
        if !seq2_done; MadIPM.update_step!(seq2.opt.step_rule, seq2); end
        MadIPM.update_step!(bat.opt.step_rule, bat)
        MadIPM.zero_inactive_step!(bat)
        if !seq1_done; assert_step_match(seq1, bat, 1; tol); end
        if !seq2_done; assert_step_match(seq2, bat, 2; tol); end

        # Apply step
        if !seq1_done; MadIPM.apply_step!(seq1); end
        if !seq2_done; MadIPM.apply_step!(seq2); end
        MadIPM.apply_step!(bat)
        if !seq1_done; assert_iterate_match(seq1, bat, 1; tol); end
        if !seq2_done; assert_iterate_match(seq2, bat, 2; tol); end

        # Evaluate model
        if !seq1_done; MadIPM.evaluate_model!(seq1); end
        if !seq2_done; MadIPM.evaluate_model!(seq2); end
        MadIPM.evaluate_model!(bat)
        if !seq1_done; assert_model_match(seq1, bat, 1; tol); end
        if !seq2_done; assert_model_match(seq2, bat, 2; tol); end

        # Patch batch columns from sequential to prevent FP noise accumulation.
        # For QPs the Hessian and Jacobian are constant, so patching iterate +
        # model evals is sufficient to get a clean start for the next iteration.
        if patch
            if !seq1_done; patch_batch_col_from_seq!(seq1, bat, 1); end
            if !seq2_done; patch_batch_col_from_seq!(seq2, bat, 2); end
        end
    end

    @test seq1.status == bat.workspace.status[1]
    @test seq2.status == bat.workspace.status[2]
end

function run_iterations_bsN!(seqs::Vector, bat, n_iters; tol=1e-10)
    N = length(seqs)
    done = falses(N)

    for iter in 1:n_iters
        for i in 1:N
            done[i] || MadIPM.update_termination_criteria!(seqs[i])
        end
        MadIPM.update_termination_criteria!(bat)
        for i in 1:N
            done[i] || assert_termination_match(seqs[i], bat, i; tol)
        end

        for i in 1:N
            done[i] = done[i] || MadIPM.is_done(seqs[i])
        end

        MadIPM.update_active_set!(bat.kkt, bat.workspace.status)
        bat.kkt.active_batch_size[] == 0 && break
        MadIPM._update_active_mask!(bat)
        all(done) && break

        # Factorize
        for i in 1:N; done[i] || MadIPM.factorize_system!(seqs[i]); end
        MadIPM.factorize_system!(bat)
        for i in 1:N
            if !done[i]
                assert_regularization_match(seqs[i], bat, i; tol)
                assert_kkt_diagonals_match(seqs[i], bat, i; tol)
            end
        end

        # Prediction step
        for i in 1:N; done[i] || MadIPM.prediction_step!(seqs[i]); end
        MadIPM.prediction_step!(bat)
        for i in 1:N
            if !done[i]
                assert_prediction_match(seqs[i], bat, i; tol)
                assert_barrier_match(seqs[i], bat, i; tol)
            end
        end

        # Mehrotra correction
        for i in 1:N; done[i] || MadIPM.mehrotra_correction_direction!(seqs[i]); end
        MadIPM.mehrotra_correction_direction!(bat)
        for i in 1:N; done[i] || assert_direction_match(seqs[i], bat, i; tol); end

        # Update step
        for i in 1:N; done[i] || MadIPM.update_step!(seqs[i].opt.step_rule, seqs[i]); end
        MadIPM.update_step!(bat.opt.step_rule, bat)
        MadIPM.zero_inactive_step!(bat)
        for i in 1:N; done[i] || assert_step_match(seqs[i], bat, i; tol); end

        # Apply step
        for i in 1:N; done[i] || MadIPM.apply_step!(seqs[i]); end
        MadIPM.apply_step!(bat)
        for i in 1:N; done[i] || assert_iterate_match(seqs[i], bat, i; tol); end

        # Evaluate model
        for i in 1:N; done[i] || MadIPM.evaluate_model!(seqs[i]); end
        MadIPM.evaluate_model!(bat)
        for i in 1:N; done[i] || assert_model_match(seqs[i], bat, i; tol); end
    end

    for i in 1:N
        @test seqs[i].status == bat.workspace.status[i]
    end
end
