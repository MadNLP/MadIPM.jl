mutable struct MPCSolver{
    T,
    VT <: AbstractVector{T},
    VI <: AbstractVector{Int},
    KKTSystem <: MadNLP.AbstractKKTSystem{T},
    Model <: NLPModels.AbstractNLPModel{T,VT},
    CB <: MadNLP.AbstractCallback{T},
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
    best_complementarity::T
    mu_curr::T
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

    ind_lb = cb.ind_lb
    ind_ub = cb.ind_ub

    ns = length(cb.ind_ineq)
    nx = NLPModels.get_nvar(nlp)
    n = nx+ns
    m = NLPModels.get_ncon(nlp)
    nlb = length(ind_lb)
    nub = length(ind_ub)

    MadNLP.@trace(logger,"Initializing KKT system.")
    kkt = MadNLP.create_kkt_system(
        ipm_opt.kkt_system,
        cb,
        ipm_opt.linear_solver;
        opt_linear_solver=options.linear_solver,
    )

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
    correction_lb = VT(undef, nlb)
    correction_ub = VT(undef, nub)
    jacl = VT(undef,n)
    c_trial = VT(undef, m)
    y = VT(undef, m)
    c = VT(undef, m)
    rhs = VT(undef, m)

    x_lr = view(full(x), cb.ind_lb)
    x_ur = view(full(x), cb.ind_ub)
    xl_r = view(full(xl), cb.ind_lb)
    xu_r = view(full(xu), cb.ind_ub)
    zl_r = view(full(zl), cb.ind_lb)
    zu_r = view(full(zu), cb.ind_ub)
    dx_lr = view(d.xp, cb.ind_lb)
    dx_ur = view(d.xp, cb.ind_ub)

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
        cb.ind_ineq, cb.ind_fixed, cb.ind_llb, cb.ind_uub,
        cb.ind_lb, cb.ind_ub,
        x_lr, x_ur, xl_r, xu_r, zl_r, zu_r, dx_lr, dx_ur,
        zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), typemax(T), zero(T),
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

