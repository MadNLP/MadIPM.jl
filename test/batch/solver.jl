using BatchQuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel
using LinearAlgebra
using SparseArrays

_lp() = QuadraticModel(ones(2), Int[], Int[], Float64[];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=ones(2))

_qp() = QuadraticModel([1.0,-2.0,0.5,1.0], [1,2,3,4], [1,2,3,4], [2.0,1.0,3.0,1.5];
    Arows=[1,1,2,2], Acols=[1,2,3,4], Avals=[1.0,1.0,1.0,1.0],
    lcon=[1.0,0.5], ucon=[2.0,1.5], lvar=zeros(4), uvar=fill(Inf,4), x0=ones(4))

_qp_ub() = QuadraticModel([1.0,-1.0], [1,2], [1,2], [1.0,1.0];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[1.0], lvar=[-Inf,-Inf], uvar=[5.0,5.0], x0=[2.5,2.5])

_qp_db() = QuadraticModel([1.0,-1.0], [1,2], [1,2], [1.0,1.0];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[5.0,5.0], x0=[2.5,2.5])

_qp_free() = QuadraticModel([1.0,-1.0], [1,2], [1,2], [1.0,1.0];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[1.0], lvar=[-Inf,-Inf], uvar=[Inf,Inf], x0=[0.5,0.5])

_qp_scaled() = QuadraticModel([1e3,-2e3,5e2,1e3], [1,2,3,4], [1,2,3,4], [2e3,1e3,3e3,1.5e3];
    Arows=[1,1,2,2], Acols=[1,2,3,4], Avals=[1e3,1e3,1e3,1e3],
    lcon=[1e3,5e2], ucon=[2e3,1.5e3], lvar=zeros(4), uvar=fill(Inf,4), x0=ones(4))

_qp_fixed() = QuadraticModel([1.0,1.0,1.0], [1,2,3], [1,2,3], [2.0,1.0,3.0];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[Inf], lvar=[0.0,0.0,2.0], uvar=[Inf,Inf,2.0], x0=[1.0,1.0,1.0])

_qp_mixed() = QuadraticModel([1.0,-1.0,0.5,1.0], [1,2,3,4], [1,2,3,4], [2.0,1.0,3.0,1.5];
    Arows=[1,1,2,2], Acols=[1,2,3,4], Avals=[1.0,1.0,1.0,1.0],
    lcon=[1.0,0.5], ucon=[2.0,1.5], lvar=[0.0,-Inf,0.0,-Inf], uvar=[Inf,5.0,5.0,Inf], x0=ones(4))

_qp_dense() = QuadraticModel([1.0,-1.0], [1,2,2], [1,1,2], [4.0,2.0,3.0];
    Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
    lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=[0.5,0.5])

_qp_A() = QuadraticModel([1.0,-2.0,0.5,1.0], [1,2,3,4], [1,2,3,4], [2.0,1.0,3.0,1.5];
    Arows=[1,1,2,2], Acols=[1,2,3,4], Avals=[1.0,1.0,1.0,1.0],
    lcon=[1.0,0.5], ucon=[2.0,1.5], lvar=zeros(4), uvar=fill(Inf,4), x0=ones(4))
_qp_B() = QuadraticModel([2.0,1.0,-1.0,0.5], [1,2,3,4], [1,2,3,4], [5.0,2.0,1.0,4.0];
    Arows=[1,1,2,2], Acols=[1,2,3,4], Avals=[2.0,1.0,0.5,1.0],
    lcon=[1.0,0.5], ucon=[2.0,1.5], lvar=zeros(4), uvar=fill(Inf,4), x0=ones(4))

const PROBLEMS = [
    "LP"         => _lp,
    "QP mixed"   => _qp_mixed,
    "QP dense"   => _qp_dense,
    "QP"         => _qp,
    "QP ub-only" => _qp_ub,
    "QP doubly"  => _qp_db,
    "QP free"    => _qp_free,
    "QP scaled"  => _qp_scaled,
    "QP fixed"   => _qp_fixed,
]

const PROBLEMS_3 = PROBLEMS[1:3]

_solve(qp; kw...) = MadIPM.solve!(MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, rethrow_error=true, kw...))
_batch(qps; kw...) = MadIPM.madipm_batch(ObjRHSBatchQuadraticModel(qps); print_level=MadNLP.ERROR, rethrow_error=true, kw...)
_fbatch(qps; kw...) = MadIPM.madipm_batch(BatchQuadraticModel(qps); print_level=MadNLP.ERROR, rethrow_error=true, kw...)

function _init_solver(solver)
    ws, bcb, opt = solver.workspace, solver.bcb, solver.opt
    MadNLP.initialize!(bcb, solver.x, solver.xl, solver.xu, MadNLP.full(solver.y),
        MadNLP.full(solver.rhs), bcb.ind_ineq, ws.bx;
        tol=opt.bound_relax_factor, bound_push=opt.bound_push, bound_fac=opt.bound_fac)
    fill!(MadNLP.full(solver.jacl), 0.0)
    if opt.scaling
        MadNLP.set_scaling!(bcb, solver.x, solver.xl, solver.xu, MadNLP.full(solver.y),
            MadNLP.full(solver.rhs), bcb.ind_ineq, Float64(opt.nlp_scaling_max_gradient), ws.bx)
    end
    MadNLP.initialize!(solver.kkt)
    MadIPM.init_regularization!(solver, opt.regularization)
    MadNLP.unpack_x!(ws.bx, bcb, solver.x)
    MadNLP.eval_f_wrapper(solver, ws.bx)
    MadNLP.eval_jac_wrapper!(solver, solver.kkt)
    MadNLP.eval_grad_f_wrapper!(solver, ws.bx)
    MadNLP.eval_cons_wrapper!(solver, ws.bx)
    MadNLP.eval_lag_hess_wrapper!(solver, solver.kkt)
    ws.norm_b .= maximum(abs, MadNLP.full(solver.rhs); dims=1)
    ws.norm_c .= maximum(abs, MadNLP.full(solver.f); dims=1)
    MadIPM.init_starting_point!(solver)
    fill!(ws.mu_batch, opt.mu_init)
    fill!(ws.best_complementarity, typemax(Float64))
    fill!(ws.status, MadNLP.REGULAR)
    fill!(ws.inf_pr, 0.0)
    fill!(ws.inf_du, 0.0)
    fill!(ws.inf_compl, 0.0)
    fill!(ws.dual_obj, 0.0)
    fill!(ws.alpha_p, 0.0)
    fill!(ws.alpha_d, 0.0)
    solver.batch_cnt.start_time[] = time()
    fill!(solver.batch_cnt.k, 0)
    MadNLP.jtprod!(solver.jacl, solver.kkt, solver.y)
    return solver
end

function _build_bat(qp; kwargs...)
    _init_solver(MadIPM.UniformBatchMPCSolver(ObjRHSBatchQuadraticModel([qp]); print_level=MadNLP.ERROR, kwargs...))
end

function _build_bat_n(qp, n::Int; kwargs...)
    _init_solver(MadIPM.UniformBatchMPCSolver(ObjRHSBatchQuadraticModel([qp for _ in 1:n]); print_level=MadNLP.ERROR, kwargs...))
end

function _build_seq(qp; kwargs...)
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
    MadNLP.eval_f_wrapper(solver, solver.x)
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
col1(x::AbstractMatrix) = view(x, :, 1)
col1(x::AbstractVector) = x

function _do_factorize!(seq, bat)
    MadIPM.update_regularization!(seq, seq.opt.regularization)
    MadIPM.set_aug_diagonal_reg!(seq.kkt, seq)
    MadNLP.build_kkt!(seq.kkt)
    MadNLP.factorize_kkt!(seq.kkt)
    MadIPM.update_regularization!(bat, bat.opt.regularization)
    MadIPM.set_aug_diagonal_reg!(bat.kkt, bat)
    MadNLP.build_kkt!(bat.kkt)
    MadNLP.factorize_kkt!(bat.kkt)
end

include("fakels.jl")

@testset "Batch solver" begin

@testset "e2e bs=1: $name" for (name, fn) in PROBLEMS
    s = _solve(fn())
    b = _batch([fn()])
    @test b.status[1] == s.status
    @test isapprox(b.objective[1], s.objective; rtol=1e-6)
    @test isapprox(b.solution[:,1], s.solution; atol=1e-6)
end

@testset "e2e MehrotraAdaptiveStep: $name" for (name, fn) in [PROBLEMS[1], PROBLEMS[3], PROBLEMS[6]]
    qp = fn()
    s = _solve(qp; step_rule=MadIPM.MehrotraAdaptiveStep(0.99))
    b = _batch([qp]; step_rule=MadIPM.MehrotraAdaptiveStep(0.99))
    @test b.status[1] == s.status
    @test isapprox(b.solution[:,1], s.solution; atol=1e-6)
end

@testset "e2e heterogeneous FullBatch" begin
    qps = [_qp_A(), _qp_A(), _qp_B()]
    refs = [_solve(qp) for qp in qps]
    stats = _fbatch(qps)
    for i in 1:3
        @test stats[i].status == MadNLP.SOLVE_SUCCEEDED
        @test stats[i].objective ≈ refs[i].objective atol=1e-6
    end
end

@testset "e2e maximization QP" begin
    qp = QuadraticModel([2.0], [1], [1], [2.0];
        Arows=[1], Acols=[1], Avals=[1.0],
        lcon=[-Inf], ucon=[5.0], lvar=[0.0], uvar=[Inf], x0=[2.5], minimize=false)
    s = _solve(qp)
    b = _batch([qp])
    @test b.status[1] == s.status
    @test isapprox(b.solution[:,1], s.solution; atol=1e-5)
end

@testset "e2e fixed variable (MakeParameter)" begin
    s = _solve(_qp_fixed(); fixed_variable_treatment=MadNLP.MakeParameter)
    b = _batch([_qp_fixed()]; fixed_variable_treatment=MadNLP.MakeParameter)
    @test isapprox(b.solution[:,1], s.solution; atol=1e-6)
end

@testset "kernel parity: $name" for (name, fn) in PROBLEMS_3
    qp = fn()
    seq = _build_seq(qp)
    bat = _build_bat(qp)

    @test cmp(MadNLP.full(seq.x), col1(MadNLP.full(bat.x))) < 1e-10
    @test cmp(seq.y, col1(MadNLP.full(bat.y))) < 1e-10
    @test cmp(MadNLP.full(seq.zl), col1(MadNLP.full(bat.zl))) < 1e-10

    _do_factorize!(seq, bat)

    @test cmp(seq.kkt.reg, col1(bat.kkt.reg)) < 1e-12
    @test cmp(seq.kkt.pr_diag, col1(MadIPM.pr_diag(bat.kkt))) < 1e-12
    @test cmp(seq.kkt.aug_raw.V, bat.kkt.nzVals[:,1]) < 1e-12
    @test cmp(SparseArrays.nonzeros(seq.kkt.aug_com), bat.kkt.aug_com_nzvals[:,1]) < 1e-12

    MadIPM.set_predictive_rhs!(seq, seq.kkt)
    MadIPM.set_predictive_rhs!(bat, bat.kkt)
    @test cmp(MadNLP.full(seq.p), col1(MadNLP.full(bat.p))) < 1e-12

    MadIPM.solve_system!(seq.d, seq, seq.p)
    MadIPM.solve_system!(bat.d, bat, bat.p)
    @test cmp(MadNLP.full(seq.d), col1(MadNLP.full(bat.d))) < 1e-10

    w_s, w_b = seq._w1, bat._w1
    fill!(MadNLP.full(w_s), 0.0)
    fill!(MadNLP.full(w_b), 0.0)
    mul!(w_s, seq.kkt, seq.d)
    mul!(w_b, bat.kkt, bat.d)
    @test cmp(MadNLP.full(w_s), col1(MadNLP.full(w_b))) < 1e-10

    seq_mu = MadIPM.get_complementarity_measure(seq)
    MadIPM.get_complementarity_measure!(bat)
    @test abs(seq_mu - bat.workspace.mu_curr[1]) < 1e-12
end

@testset "regularization" begin
    qp = _qp()
    for (reg1, reg2) in [
        (MadIPM.NoRegularization(), MadIPM.NoRegularization()),
        (MadIPM.FixedRegularization(1e-8,-1e-9), MadIPM.FixedRegularization(1e-8,-1e-9)),
        (MadIPM.AdaptiveRegularization(1e-8,-1e-9,1e-9), MadIPM.AdaptiveRegularization(1e-8,-1e-9,1e-9)),
    ]
        seq = MadIPM.MPCSolver(qp; print_level=MadNLP.ERROR, regularization=reg1)
        bat = MadIPM.UniformBatchMPCSolver(ObjRHSBatchQuadraticModel([qp]); print_level=MadNLP.ERROR, regularization=reg2)
        MadIPM.init_regularization!(seq, reg1)
        MadIPM.init_regularization!(bat, reg2)
        @test seq.del_w == bat.del_w[1]
        for _ in 1:3
            MadIPM.update_regularization!(seq, reg1)
            MadIPM.update_regularization!(bat, reg2)
            @test abs(seq.del_w - bat.del_w[1]) < 1e-15
        end
    end
end

@testset "bs=3 consistency: $name" for (name, fn) in PROBLEMS_3
    bat1 = _build_bat(fn())
    bat3 = _build_bat_n(fn(), 3)
    _do_factorize!(bat1, bat3)
    MadIPM.prediction_step!(bat1)
    MadIPM.prediction_step!(bat3)
    MadIPM.mehrotra_correction_direction!(bat1)
    MadIPM.mehrotra_correction_direction!(bat3)
    MadIPM.update_step!(bat1.opt.step_rule, bat1)
    MadIPM.update_step!(bat3.opt.step_rule, bat3)
    MadIPM.apply_step!(bat1)
    MadIPM.apply_step!(bat3)
    for i in 1:3
        @test cmp(col1(MadNLP.full(bat1.x)), view(MadNLP.full(bat3.x),:,i)) < 1e-10
    end
    @test bat3.workspace.alpha_p[1] == bat3.workspace.alpha_p[2] == bat3.workspace.alpha_p[3]
end

@testset "staggered frozen: $name" for (name, fn) in PROBLEMS_3
    bat = _build_bat_n(fn(), 3)
    ws = bat.workspace
    for _ in 1:5
        MadIPM.mpc_step!(bat)
    end
    snap_x = copy(MadNLP.full(bat.x)[:,2])
    snap_y = copy(MadNLP.full(bat.y)[:,2])
    snap_nzv = copy(bat.kkt.nzVals[:,2])

    ws.status[2] = MadNLP.SOLVE_SUCCEEDED
    MadIPM.update_active_set!(bat)
    MadIPM._update_active_mask!(bat)
    for _ in 1:5
        MadIPM.mpc_step!(bat)
    end
    @test MadNLP.full(bat.x)[:,2] ≈ snap_x atol=1e-14
    @test MadNLP.full(bat.y)[:,2] ≈ snap_y atol=1e-14
    @test bat.kkt.nzVals[:,2] ≈ snap_nzv atol=1e-14
    @test ws.alpha_p[1,1] > 0 || ws.status[1] != MadNLP.REGULAR
end

@testset "termination" begin
    bat = _build_bat(_qp())
    bat.opt.max_iter = 0
    MadIPM.update_termination_criteria!(bat)
    MadIPM.update_termination_status!(bat)
    @test bat.workspace.status[1] == MadNLP.MAXIMUM_ITERATIONS_EXCEEDED

    bat = _build_bat(_qp())
    bat.opt.max_wall_time = 0.0
    bat.batch_cnt.start_time[] = time() - 1.0
    MadIPM.update_termination_criteria!(bat)
    MadIPM.update_termination_status!(bat)
    @test bat.workspace.status[1] == MadNLP.MAXIMUM_WALLTIME_EXCEEDED
end

@testset "compute_term_gpu!" begin
    bat = _build_bat_n(_qp(), 2)
    ws = bat.workspace
    tol = bat.opt.tol
    ds = bat.opt.divergence_scale
    dt = bat.opt.divergence_tol

    function _set!(; pr=1.0, du=1.0, ic=1.0, best=1e10, obj=1.0, dobj=1.0, ls=Int32(0))
        ws.inf_pr .= pr
        ws.inf_du .= du
        ws.inf_compl .= ic
        ws.best_complementarity .= best
        ws.obj_val .= obj
        ws.dual_obj .= dobj
        fill!(ws._ls_error, ls)
    end
    _s(j) = MadNLP.Status(ws._term_gpu[1,j])

    _set!(pr=tol/10, du=tol/10, ic=tol/10)
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.SOLVE_SUCCEEDED

    _set!(ls=Int32(1))
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.INTERNAL_ERROR

    _set!(ic=1e10, best=1e-8, dobj=1e12, obj=1.0)
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.INFEASIBLE_PROBLEM_DETECTED

    _set!(dobj=1.0, obj=-(dt*ds*2))
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.DIVERGING_ITERATES

    _set!(pr=tol/10, du=tol/10, ic=tol/10, ls=Int32(1))
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.INTERNAL_ERROR

    _set!()
    ws.inf_pr[1,1] = tol/10
    ws.inf_du[1,1] = tol/10
    ws.inf_compl[1,1] = tol/10
    ws.obj_val[1,2] = -(dt*ds*2)
    MadIPM.compute_term_gpu!(ws, bat.opt)
    @test _s(1) == MadNLP.SOLVE_SUCCEEDED
    @test _s(2) == MadNLP.DIVERGING_ITERATES
end

@testset "structure mismatch" begin
    qp1 = QuadraticModel([1.0,1.0], [1,2], [1,2], [2.0,2.0];
        Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
        lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[1.0,1.0], x0=[0.5,0.5])
    qp2 = QuadraticModel([1.0,1.0], [1,2], [1,2], [2.0,2.0];
        Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
        lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[0.0,1.0], x0=[0.0,0.5])
    @test_throws AssertionError MadIPM.UniformBatchMPCSolver(BatchQuadraticModel([qp1,qp2]); print_level=MadNLP.ERROR)
    MadIPM.UniformBatchMPCSolver(BatchQuadraticModel([qp1,qp2]); print_level=MadNLP.ERROR, check_batch_structure=false)
end

@testset "residual INTERNAL_ERROR (all fail)" begin
    stats = _batch([_lp() for _ in 1:3]; check_residual=true, tol_linear_solve=0.0)
    for i in 1:3
        @test stats[i].status == MadNLP.INTERNAL_ERROR
    end
end

@testset "residual INTERNAL_ERROR (partial NaN)" begin
    good() = QuadraticModel([1.0,1.0], [1,2], [1,2], [2.0,2.0];
        Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
        lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=[0.5,0.5])
    bad = QuadraticModel([NaN,NaN], [1,2], [1,2], [2.0,2.0];
        Arows=[1,1], Acols=[1,2], Avals=[1.0,1.0],
        lcon=[1.0], ucon=[1.0], lvar=[0.0,0.0], uvar=[Inf,Inf], x0=[0.5,0.5])
    stats = _fbatch([good(), bad, good()])
    @test stats[1].status == MadNLP.SOLVE_SUCCEEDED
    @test stats[3].status == MadNLP.SOLVE_SUCCEEDED
    @test stats[2].status == MadNLP.INTERNAL_ERROR
end

@testset "factorize retry" begin
    function _mk(; fail_positions=Set{Int}(), fail_remaining=0)
        bnlp = BatchQuadraticModel([_qp_A(), _qp_A(), _qp_B()])
        s = MadIPM.UniformBatchMPCSolver(bnlp;
            print_level=MadNLP.ERROR,
            uniformbatch_linear_solver=FailOnDemandLS,
            looped_linear_solver=MadNLP.LDLSolver,
            regularization=MadIPM.FixedRegularization(1e-8,-1e-9),
            rethrow_error=true)
        MadIPM.initialize!(s)
        s.kkt.batch_solver.fail_positions = fail_positions
        s.kkt.batch_solver.fail_remaining = fail_remaining
        return s
    end

    @testset "non-identity view" begin
        ref = _mk()
        ref.workspace.status[2] = MadNLP.SOLVE_SUCCEEDED
        MadIPM.update_active_set!(ref)
        MadIPM._update_active_mask!(ref)
        MadIPM.factorize_system!(ref)
        MadIPM.set_predictive_rhs!(ref, ref.kkt)
        copyto!(MadNLP.full(ref.d), MadNLP.full(ref.p))
        MadNLP.solve_kkt!(ref.kkt, ref)
        ref_d1 = copy(MadNLP.primal(ref.d)[:,1])

        bad = _mk(fail_positions=Set([2]), fail_remaining=1)
        bad.workspace.status[2] = MadNLP.SOLVE_SUCCEEDED
        MadIPM.update_active_set!(bad)
        MadIPM._update_active_mask!(bad)
        MadIPM.factorize_system!(bad)

        @test MadIPM.is_factorized(bad.kkt.batch_solver.solvers[1])
        @test MadIPM.is_factorized(bad.kkt.batch_solver.solvers[2])

        MadIPM.set_predictive_rhs!(bad, bad.kkt)
        copyto!(MadNLP.full(bad.d), MadNLP.full(bad.p))
        MadNLP.solve_kkt!(bad.kkt, bad)
        @test norm(ref_d1 - MadNLP.primal(bad.d)[:,1]) / norm(ref_d1) < 1e-6
    end

    @testset "identity view" begin
        ref = _mk()
        MadIPM.factorize_system!(ref)
        MadIPM.set_predictive_rhs!(ref, ref.kkt)
        copyto!(MadNLP.full(ref.d), MadNLP.full(ref.p))
        MadNLP.solve_kkt!(ref.kkt, ref)
        ref_d1 = copy(MadNLP.primal(ref.d)[:,1])

        bad = _mk(fail_positions=Set([3]), fail_remaining=1)
        MadIPM.factorize_system!(bad)
        @test all(MadIPM.is_factorized(bad.kkt.batch_solver.solvers[j]) for j in 1:3)

        MadIPM.set_predictive_rhs!(bad, bad.kkt)
        copyto!(MadNLP.full(bad.d), MadNLP.full(bad.p))
        MadNLP.solve_kkt!(bad.kkt, bad)
        @test norm(ref_d1 - MadNLP.primal(bad.d)[:,1]) / norm(ref_d1) < 1e-6
    end

    @testset "regularization bumped correctly" begin
        bad = _mk(fail_positions=Set([2]), fail_remaining=1)
        bad.workspace.status[2] = MadNLP.SOLVE_SUCCEEDED
        MadIPM.update_active_set!(bad)
        MadIPM._update_active_mask!(bad)
        MadIPM.factorize_system!(bad)
        @test bad.del_w[1,1] ≈ 1e-8
        @test bad.del_w[1,2] ≈ 1.0
        @test bad.del_w[1,3] ≈ 1e-6
    end
end

@testset "free vars: mu = mu_min" begin
    bat = _build_bat(_qp_free())
    MadIPM.get_complementarity_measure!(bat)
    @test bat.workspace.mu_curr[1] == 0.0
    MadIPM.update_barrier!(MadIPM.Mehrotra(), bat, bat.workspace.mu_affine)
    @test bat.workspace.mu_batch[1] == bat.opt.mu_min
end

@testset "_adjust_boundary_active!" begin
    qp = _qp_mixed()
    seq = _build_seq(qp)
    bat = _build_bat(qp)
    seq.mu = 0.1
    bat.workspace.mu_batch .= 0.1
    bat.workspace.active_mask .= 1.0
    MadNLP.adjust_boundary!(seq.x_lr, seq.xl_r, seq.x_ur, seq.xu_r, seq.mu)
    MadIPM._adjust_boundary_active!(
        MadIPM.lower(bat.x), MadIPM.lower(bat.xl),
        MadIPM.upper(bat.x), MadIPM.upper(bat.xu),
        bat.workspace.mu_batch, bat.workspace.active_mask)
    @test cmp(seq.xl_r, col1(MadIPM.lower(bat.xl))) < 1e-14
    @test cmp(seq.xu_r, col1(MadIPM.upper(bat.xu))) < 1e-14

    bat2 = _build_bat_n(_qp(), 2)
    bat2.workspace.mu_batch .= 1e-12
    bat2.workspace.active_mask .= 0.0
    xl_before = copy(MadIPM.lower(bat2.xl))
    MadIPM._adjust_boundary_active!(
        MadIPM.lower(bat2.x), MadIPM.lower(bat2.xl),
        MadIPM.upper(bat2.x), MadIPM.upper(bat2.xu),
        bat2.workspace.mu_batch, bat2.workspace.active_mask)
    @test MadIPM.lower(bat2.xl) == xl_before
end

@testset "partial active KKT solve preserves rhs" begin
    solver = MadIPM.UniformBatchMPCSolver(BatchQuadraticModel([_lp() for _ in 1:3]); print_level=MadNLP.ERROR)
    MadIPM.initialize!(solver)
    solver.workspace.status[2] = MadNLP.INTERNAL_ERROR
    MadIPM.update_active_set!(solver)
    pd = MadNLP.primal_dual(solver.d)
    pd .= reshape(collect(1.0:length(pd)), size(pd))
    pd_before = copy(pd)
    MadNLP.build_kkt!(solver.kkt)
    MadNLP.factorize_kkt!(solver.kkt)
    MadNLP.solve_kkt!(solver.kkt, solver)
    @test pd[:,2] == pd_before[:,2]
end

end
