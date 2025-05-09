

mutable struct MPCSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
    Iterator <: MadNLP.AbstractIterator{T},
} <: MadNLP.AbstractMadNLPSolver{T}
    nlp::Model
    class::AbstractConicProblem
    cb::CB
    kkt::KKTSystem

    opt::IPMOptions
    cnt::MadNLP.MadNLPCounters
    logger::MadNLP.MadNLPLogger

    n::Int # number of variables (after reformulation)
    m::Int # number of cons
    nlb::Int
    nub::Int

    x::MadNLP.PrimalVector{T, VT, VI} # primal (after reformulation)
    y::VT # dual
    zl::MadNLP.PrimalVector{T, VT, VI} # dual (after reformulation)
    zu::MadNLP.PrimalVector{T, VT, VI} # dual (after reformulation)
    xl::MadNLP.PrimalVector{T, VT, VI} # primal lower bound (after reformulation)
    xu::MadNLP.PrimalVector{T, VT, VI} # primal upper bound (after reformulation)

    obj_val::T
    f::MadNLP.PrimalVector{T, VT, VI}
    c::VT

    jacl::VT

    d::MadNLP.UnreducedKKTVector{T, VT}
    p::MadNLP.UnreducedKKTVector{T, VT}

    # Buffers
    _w1::MadNLP.UnreducedKKTVector{T, VT}
    _w2::MadNLP.UnreducedKKTVector{T, VT}

    correction_lb::VT
    correction_ub::VT
    rhs::VT
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI

    x_lr::MadNLP.SubVector{T,VT,VI}
    x_ur::MadNLP.SubVector{T,VT,VI}
    xl_r::MadNLP.SubVector{T,VT,VI}
    xu_r::MadNLP.SubVector{T,VT,VI}
    zl_r::MadNLP.SubVector{T,VT,VI}
    zu_r::MadNLP.SubVector{T,VT,VI}
    dx_lr::MadNLP.SubVector{T,VT,VI}
    dx_ur::MadNLP.SubVector{T,VT,VI}

    row_scaling::VT
    col_scaling::VT

    iterator::Iterator

    inf_pr::T
    inf_du::T
    inf_compl::T
    norm_b::T
    norm_c::T

    mu::T

    alpha_p::T
    alpha_d::T
    del_w::T
    del_c::T
    status::MadNLP.Status
end

function MPCSolver(nlp::NLPModels.AbstractNLPModel{T,VT}; kwargs...) where {T, VT}
    options = load_options(nlp; kwargs...)

    ipm_opt = options.interior_point
    logger = options.logger
    @assert MadNLP.is_supported(ipm_opt.linear_solver, T)

    cnt = MadNLP.MadNLPCounters(start_time=time())
    cb = MadNLP.create_callback(
        MadNLP.SparseCallback,
        nlp;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
    )

    # generic options
    MadNLP.@trace(logger,"Initializing variables.")

    ind_cons = MadNLP.get_index_constraints(
        NLPModels.get_lvar(nlp),
        NLPModels.get_uvar(nlp),
        NLPModels.get_lcon(nlp),
        NLPModels.get_ucon(nlp);
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment
    )

    ind_lb = ind_cons.ind_lb
    ind_ub = ind_cons.ind_ub

    ns = length(ind_cons.ind_ineq)
    nx = NLPModels.get_nvar(nlp)
    n = nx+ns
    m = NLPModels.get_ncon(nlp)
    nlb = length(ind_lb)
    nub = length(ind_ub)

    MadNLP.@trace(logger,"Initializing KKT system.")
    kkt = MadNLP.create_kkt_system(
        ipm_opt.kkt_system,
        cb,
        ind_cons,
        ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    )

    MadNLP.@trace(logger,"Initializing iterative solver.")
    iterator = ipm_opt.iterator(kkt; cnt = cnt, logger = logger, opt = options.iterative_refinement)

    x = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xl = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    xu = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zl = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    zu = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)
    f = MadNLP.PrimalVector(VT, nx, ns, ind_lb, ind_ub)

    d = MadNLP.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    p = MadNLP.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w1 = MadNLP.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)
    _w2 = MadNLP.UnreducedKKTVector(VT, n, m, nlb, nub, ind_lb, ind_ub)

    # Buffers
    correction_lb = zeros(nlb)
    correction_ub = zeros(nub)
    jacl = VT(undef,n)
    c_trial = VT(undef, m)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    x_lr = view(full(x), ind_cons.ind_lb)
    x_ur = view(full(x), ind_cons.ind_ub)
    xl_r = view(full(xl), ind_cons.ind_lb)
    xu_r = view(full(xu), ind_cons.ind_ub)
    zl_r = view(full(zl), ind_cons.ind_lb)
    zu_r = view(full(zu), ind_cons.ind_ub)
    dx_lr = view(d.xp, ind_cons.ind_lb)
    dx_ur = view(d.xp, ind_cons.ind_ub)

    row_scaling = VT(undef, 0)
    col_scaling = VT(undef, 0)

    cnt.init_time = time() - cnt.start_time

    nnzh = MadNLP.get_nnzh(nlp)
    # check if the problem is a LP or a QP
    class = iszero(nnzh) ? LinearProgram() : QuadraticProgram()

    return MPCSolver(
        nlp, class, cb, kkt,
        ipm_opt, cnt, options.logger,
        n, m, nlb, nub,
        x, y, zl, zu, xl, xu,
        zero(T), f, c,
        jacl,
        d, p,
        _w1, _w2,
        correction_lb, correction_ub,
        rhs,
        ind_cons.ind_ineq, ind_cons.ind_fixed, ind_cons.ind_llb, ind_cons.ind_uub,
        ind_cons.ind_lb, ind_cons.ind_ub,
        x_lr, x_ur, xl_r, xu_r, zl_r, zu_r, dx_lr, dx_ur,
        row_scaling, col_scaling,
        iterator,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T),
        MadNLP.INITIAL,
    )
end

function MadNLP.print_iter(solver::MPCSolver; options...)
    obj_scale = solver.cb.obj_scale[]
    mod(solver.cnt.k,10)==0&& MadNLP.@info(solver.logger,@sprintf(
        "iter    objective    inf_pr   inf_du lg(mu)  ||d||  lg(rg) alpha_du alpha_pr"))
    inf_du = solver.inf_du
    inf_pr = solver.inf_pr
    mu = log10(solver.mu)
    MadNLP.@info(solver.logger,Printf.@sprintf(
        "%4i%s% 10.7e %6.2e %6.2e %5.1f %6.2e %s %6.2e %6.2e",
        solver.cnt.k,
        " ",
        solver.obj_val/obj_scale,
        inf_pr, inf_du, mu,
        solver.cnt.k == 0 ? 0. : norm(MadNLP.primal(solver.d),Inf),
        solver.del_w == 0 ? "   - " : @sprintf("%5.1f",log(10,solver.del_w)),
        solver.alpha_d,solver.alpha_p))
    return
end
