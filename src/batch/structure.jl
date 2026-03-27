struct UniformBatchWorkspace{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}, MI<:AbstractMatrix{Int32}, MI64<:AbstractMatrix{Int64}}
    alpha_xl::MT
    alpha_xu::MT
    alpha_zl::MT
    alpha_zu::MT
    alpha_p::MT
    alpha_d::MT
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

    _term_gpu::MI64
    _term_cpu::Vector{Int64}
    _any_nonregular_gpu::MI64
    _any_nonregular_cpu::Matrix{Int64}
    _norm_gpu_w::MT
    _norm_gpu_p::MT
    _ls_error::MI

    active_mask::MT
    active_mask_cpu::Matrix{T}

    bx::MT
    bf::VT
    bg::MT
    bv::MT
end

function UniformBatchWorkspace(::Type{MT}, ::Type{VT}, n::Int, m::Int, nlb::Int, nub::Int, batch_size::Int;
                        nvar_nlp::Int=0) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    _proto = MT(undef, 1, batch_size)
    MI = typeof(similar(_proto, Int32))
    MI64 = typeof(similar(_proto, Int64))
    return UniformBatchWorkspace{T, VT, MT, MI, MI64}(
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_xl, alpha_xu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_zl, alpha_zu
        MT(undef, 1, batch_size), MT(undef, 1, batch_size),  # alpha_p, alpha_d
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
        similar(_proto, Int64),    # _term_gpu
        zeros(Int64, batch_size),  # _term_cpu
        fill!(similar(_proto, Int64, 1, 1), Int64(Int(MadNLP.REGULAR))),  # _any_nonregular_gpu
        zeros(Int64, 1, 1),        # _any_nonregular_cpu
        MT(undef, 1, batch_size),  # _norm_gpu_w
        MT(undef, 1, batch_size),  # _norm_gpu_p
        fill!(similar(_proto, Int32), zero(Int32)),  # _ls_error
        fill!(MT(undef, 1, batch_size), one(T)),  # active_mask
        ones(T, 1, batch_size),                    # active_mask_cpu
        MT(undef, nvar_nlp, batch_size),   # bx
        VT(undef, batch_size),  # bf
        MT(undef, nvar_nlp, batch_size),   # bg
        MT(undef, m, batch_size),          # bv
    )
end

mutable struct UniformBatchMPCSolver{T, MT, VT, VI, BM, BCB} <: AbstractBatchMPCSolver{T, MT, VT}
    batch_size::Int

    d::BatchUnreducedKKTVector{T, MT}
    p::BatchUnreducedKKTVector{T, MT}
    _w1::BatchUnreducedKKTVector{T, MT}

    x::BatchPrimalVector{T, MT}
    xl::BatchPrimalVector{T, MT}
    xu::BatchPrimalVector{T, MT}
    zl::BatchPrimalVector{T, MT}
    zu::BatchPrimalVector{T, MT}
    f::BatchPrimalVector{T, MT}

    y::BatchVector{T, MT}
    c::BatchVector{T, MT}
    jacl::BatchVector{T, MT}
    rhs::BatchVector{T, MT}
    correction_lb::BatchVector{T, MT}
    correction_ub::BatchVector{T, MT}

    workspace::UniformBatchWorkspace{T, VT, MT}

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

    batch_correction_lb = BatchVector(MT, VT, nlb, batch_size)
    batch_correction_ub = BatchVector(MT, VT, nub, batch_size)
    batch_jacl          = BatchVector(MT, VT, n, batch_size)
    batch_y             = BatchVector(MT, VT, m, batch_size)
    batch_c             = BatchVector(MT, VT, m, batch_size)
    batch_rhs           = BatchVector(MT, VT, m, batch_size)

    workspace = UniformBatchWorkspace(MT, VT, n, m, nlb, nub, batch_size;
                               nvar_nlp=nvar_nlp)

    batch_del_w = fill!(MT(undef, 1, batch_size), zero(T))
    batch_del_c = fill!(MT(undef, 1, batch_size), zero(T))

    return UniformBatchMPCSolver{T, MT, VT, VI, typeof(bnlp), typeof(bcb)}(
        batch_size,
        batch_d, batch_p, batch_w1,
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
