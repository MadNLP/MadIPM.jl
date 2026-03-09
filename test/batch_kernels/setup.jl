using Test
using LinearAlgebra
using SparseArrays
using MadNLP
using MadIPM
using QuadraticModels
using QuadraticModels: ObjRHSBatchQuadraticModel

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

