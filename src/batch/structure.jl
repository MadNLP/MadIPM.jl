struct UniformBatchWorkspace{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, MI<:AbstractMatrix{Int32}}
    alpha_xl::MT
    alpha_xu::MT
    alpha_zl::MT
    alpha_zu::MT
    alpha_p::MT
    alpha_d::MT
    idx_xl::MI
    idx_xu::MI
    idx_zl::MI
    idx_zu::MI
    tau::MT

    mu_batch::MT
    mu_curr::MT
    mu_affine::MT
    sum_lb::MT
    sum_ub::MT

    obj_val::MT
    norm_b::MT
    norm_c::MT
    inf_pr::MT
    inf_du::MT
    inf_compl::MT
    best_complementarity::MT
    dual_obj::MT
    status::Vector{MadNLP.Status}
    term_converged::Vector{Bool}
    term_infeasible::Vector{Bool}
    term_diverging::Vector{Bool}

    active_mask::MT

    bx::MT
    bf::VT
    bg::MT
    bv::MT
end

function UniformBatchWorkspace(::Type{MT}, ::Type{VT}, nlb::Int, nub::Int, batch_size::Int;
                        nvar_nlp::Int=0, ncon::Int=0) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    _proto = MT(undef, 1, batch_size)
    MI = typeof(similar(_proto, Int32))
    return UniformBatchWorkspace{T, VT, MT, MI}(
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_xl, alpha_xu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_zl, alpha_zu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_p, alpha_d
        similar(_proto, Int32), similar(_proto, Int32),       # idx_xl, idx_xu
        similar(_proto, Int32), similar(_proto, Int32),       # idx_zl, idx_zu
        MT(undef, 1, batch_size),  # tau
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # mu_batch, mu_curr
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # mu_affine, sum_lb
        MT(undef, 1, batch_size),  # sum_ub
        MT(undef, 1, batch_size),  # obj_val
        MT(undef, 1, batch_size),  # norm_b
        MT(undef, 1, batch_size),  # norm_c
        MT(undef, 1, batch_size),  # inf_pr
        MT(undef, 1, batch_size),  # inf_du
        MT(undef, 1, batch_size),  # inf_compl
        MT(undef, 1, batch_size),  # best_complementarity
        MT(undef, 1, batch_size),  # dual_obj
        fill(MadNLP.INITIAL, batch_size),  # status
        falses(batch_size),  # term_converged
        falses(batch_size),  # term_infeasible
        falses(batch_size),  # term_diverging
        fill!(MT(undef, 1, batch_size), one(T)),  # active_mask
        MT(undef, nvar_nlp, batch_size),   # bx
        VT(undef, batch_size),  # bf
        MT(undef, nvar_nlp, batch_size),   # bg
        MT(undef, ncon, batch_size),       # bv
    )
end

mutable struct UniformBatchMPCSolver{T, MT, VT, VI, BM, BCB, MI} <: AbstractBatchMPCSolver{T}
    batch_size::Int

    d::BatchUnreducedKKTVector{T, MT, VT}
    p::BatchUnreducedKKTVector{T, MT, VT}
    _w1::BatchUnreducedKKTVector{T, MT, VT}
    _w2::BatchUnreducedKKTVector{T, MT, VT}

    x::BatchPrimalVector{T, MT, VT}
    xl::BatchPrimalVector{T, MT, VT}
    xu::BatchPrimalVector{T, MT, VT}
    zl::BatchPrimalVector{T, MT, VT}
    zu::BatchPrimalVector{T, MT, VT}
    f::BatchPrimalVector{T, MT, VT}

    y::BatchVector{T, MT, VT}
    c::BatchVector{T, MT, VT}
    jacl::BatchVector{T, MT, VT}
    rhs::BatchVector{T, MT, VT}
    correction_lb::BatchVector{T, MT, VT}
    correction_ub::BatchVector{T, MT, VT}

    workspace::UniformBatchWorkspace{T, VT, MT, MI}

    opt::IPMOptions
    batch_cnt::BatchCounters
    logger::MadNLP.MadNLPLogger
    kkt::AbstractBatchKKTSystem{T}

    del_w::MT
    del_c::MT

    nlp::BM
    bcb::BCB
end

_get_ind_lb(bs::AbstractBatchMPCSolver) = bs.bcb.ind_lb
_get_ind_ub(bs::AbstractBatchMPCSolver) = bs.bcb.ind_ub
_get_ind_llb(bs::AbstractBatchMPCSolver) = bs.bcb.ind_llb
_get_ind_uub(bs::AbstractBatchMPCSolver) = bs.bcb.ind_uub


"""
    UniformBatchMPCSolver(bnlp::AbstractBatchNLPModel; linear_solver, kwargs...)

Construct a batch solver from a `AbstractBatchNLPModel`.
"""
function UniformBatchMPCSolver(
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    MT = typeof(similar(NLPModels.get_x0(bnlp), T, 0, 0)),
    VT = typeof(similar(NLPModels.get_x0(bnlp), T, 0)),
    VI = typeof(similar(NLPModels.get_x0(bnlp), Int, 0)),
    uniformbatch_linear_solver = LoopedBatchLinearSolver,
    kwargs...,
) where {T}
    bmeta = bnlp.meta
    batch_size = bmeta.nbatch
    @assert batch_size > 0 "Need at least one instance in batch"

    nvar_nlp = bmeta.nvar

    opt_batch_ls = MadNLP.default_options(uniformbatch_linear_solver)
    remaining_kwargs = MadNLP.set_options!(opt_batch_ls, kwargs)

    options = load_options(bnlp; remaining_kwargs...)
    ipm_opt = options.interior_point
    logger = options.logger

    batch_cnt = BatchCounters(batch_size)
    bcb = MadNLP.create_callback(
        UniformBatchCallback{T,VT,MT,VI},
        bnlp;
        fixed_variable_treatment=ipm_opt.fixed_variable_treatment,
        equality_treatment=ipm_opt.equality_treatment,
    )

    ind_lb = bcb.ind_lb
    ind_ub = bcb.ind_ub

    ns = length(bcb.ind_ineq)
    nx = bcb.nvar
    n = nx + ns
    m = bcb.ncon
    nlb = length(ind_lb)
    nub = length(ind_ub)

    batch_kkts = MadNLP.create_kkt_system(
        ipm_opt.kkt_system,
        bcb,
        uniformbatch_linear_solver;
        opt_linear_solver = opt_batch_ls,
    )

    batch_x  = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_xl = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_xu = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_zl = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_zu = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)
    batch_f  = BatchPrimalVector(MT, VT, nx, ns, batch_size, ind_lb, ind_ub)

    batch_d  = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    batch_p  = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    batch_w1 = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)
    batch_w2 = BatchUnreducedKKTVector(MT, VT, n, m, nlb, nub, batch_size, ind_lb, ind_ub)

    batch_correction_lb = BatchVector(MT, VT, nlb, batch_size)
    batch_correction_ub = BatchVector(MT, VT, nub, batch_size)
    batch_jacl          = BatchVector(MT, VT, n, batch_size)
    batch_y             = BatchVector(MT, VT, m, batch_size)
    batch_c             = BatchVector(MT, VT, m, batch_size)
    batch_rhs           = BatchVector(MT, VT, m, batch_size)

    workspace = UniformBatchWorkspace(MT, VT, nlb, nub, batch_size;
                               nvar_nlp=nvar_nlp, ncon=m)

    batch_del_w = fill!(MT(undef, 1, batch_size), zero(T))
    batch_del_c = fill!(MT(undef, 1, batch_size), zero(T))

    MI = typeof(workspace.idx_xl)
    return UniformBatchMPCSolver{T, MT, VT, VI, typeof(bnlp), typeof(bcb), MI}(
        batch_size,
        batch_d, batch_p, batch_w1, batch_w2,
        batch_x, batch_xl, batch_xu, batch_zl, batch_zu, batch_f,
        batch_y, batch_c, batch_jacl, batch_rhs,
        batch_correction_lb, batch_correction_ub,
        workspace,
        ipm_opt, batch_cnt, logger,
        batch_kkts,
        batch_del_w, batch_del_c,
        bnlp,
        bcb,
    )
end
