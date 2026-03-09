abstract type AbstractBatchKKTSystem{T} end

"""Batched version of SparseKKTSystem"""
struct SparseUniformBatchKKTSystem{T, LS, VT, MT, VI, VI32, SMT} <: AbstractBatchKKTSystem{T}
    nzVals::MT              # (aug_mat_length × batch_size) COO nonzero values
    aug_I::VI32             # shared row indices
    aug_J::VI32             # shared column indices
    batch_solver::LS        # batched linear solver
    rhs_buffer::VT          # contiguous (n+m)*batch_size for batch solve
    batch_size::Int
    aug_com_nzvals::MT      # (nnz_csc × batch_size) CSC nonzero values
    batch_csc_map::VI       # flattened COO→CSC map for all instances
    n_tot::Int              # n + n_slack (total primal variables)
    m::Int                  # number of constraints
    nnzh::Int               # number of Hessian nonzeros
    # Diagonal and bound data (for _kktmul!)
    reg::MT                 # (n_tot × batch_size) primal regularization
    l_diag::MT              # (nlb × batch_size) lower bound diagonals
    u_diag::MT              # (nub × batch_size) upper bound diagonals
    l_lower::MT             # (nlb × batch_size) lower bound multipliers
    u_lower::MT             # (nub × batch_size) upper bound multipliers
    # Hessian scatter (for mul!)
    hess_scatter::SMT       # (n_tot × n_hess_sym) scatter matrix
    hess_nz_map::VI         # nzVals row indices (with sym duplication)
    hess_var_map::VI        # variable indices for x multiplication
    hess_buffer::MT         # (n_hess_sym × batch_size) workspace
    # J^T scatter (for jtprod! and mul!)
    jt_scatter::SMT         # (n_tot × n_jac_aug) scatter: S[var_idx, k] = 1
    jt_nz_map::VI           # nzVals row indices for Jacobian entries
    jt_con_map::VI          # maps each Jac nonzero to its constraint index (1:m)
    jt_con_map_full::VI     # jt_con_map offset by n_tot (for indexing into full KKT vector)
    jt_buffer::MT           # (n_jac_aug × batch_size) buffer for jtprod
    # J scatter (for mul!)
    j_scatter::SMT          # (m × n_jac_aug) scatter: S[con_idx, k] = 1
    j_var_map::VI           # variable indices for J entries
    j_buffer::MT            # (n_jac_aug × batch_size) buffer for jprod
    # Workspace for mul! (GPU needs full matrices, not SubArray views)
    _mul_w_primal::MT      # (n_tot × batch_size)
    _mul_w_dual::MT        # (m × batch_size)
    # Batch tracking
    batch_map::Vector{Int}              # original index → active position (0 if inactive)
    batch_map_rev::Vector{Int}          # active position → original index
    active_batch_size::Base.RefValue{Int}
end

pr_diag(bkkt::SparseUniformBatchKKTSystem) = view(bkkt.nzVals, 1:bkkt.n_tot, :)
function du_diag(bkkt::SparseUniformBatchKKTSystem)
    du_off = size(bkkt.nzVals, 1) - bkkt.m
    return view(bkkt.nzVals, du_off+1:du_off+bkkt.m, :)
end

function MadNLP.create_kkt_system(
    ::Type{MadNLP.SparseKKTSystem},
    bcb::UniformBatchCallback{T, VT},
    uniformbatch_linear_solver = LoopedBatchLinearSolver;
    opt_linear_solver = MadNLP.default_options(uniformbatch_linear_solver),
) where {T, VT}
    batch_size = bcb.batch_size

    n_slack = length(bcb.ind_ineq)
    n = bcb.nvar
    m = bcb.ncon
    jac_sparsity_I = MadNLP.create_array(bcb, Int32, bcb.nnzj)
    jac_sparsity_J = MadNLP.create_array(bcb, Int32, bcb.nnzj)
    MadNLP._jac_sparsity_wrapper!(bcb, jac_sparsity_I, jac_sparsity_J)

    hess_sparsity_I, hess_sparsity_J = MadNLP.build_hessian_structure(bcb, MadNLP.ExactHessian)

    nlb = length(bcb.ind_lb)
    nub = length(bcb.ind_ub)

    MadNLP.force_lower_triangular!(hess_sparsity_I, hess_sparsity_J)

    ind_ineq = bcb.ind_ineq

    n_slack = length(ind_ineq)
    n_jac = length(jac_sparsity_I)
    n_hess = length(hess_sparsity_I)
    n_tot = n + n_slack

    aug_vec_length = n_tot+m
    aug_mat_length = n_tot+m+n_hess+n_jac+n_slack

    I = MadNLP.create_array(bcb, Int32, aug_mat_length)
    J = MadNLP.create_array(bcb, Int32, aug_mat_length)
    nzVals = similar(bcb.con_buffer, aug_mat_length, batch_size)
    fill!(nzVals, zero(T))
    V = _madnlp_unsafe_column_wrap(nzVals, aug_mat_length, 1, VT)

    offset = n_tot+n_jac+n_slack+n_hess+m

    I[1:n_tot] .= 1:n_tot
    I[n_tot+1:n_tot+n_hess] = hess_sparsity_I
    I[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= (jac_sparsity_I.+n_tot)
    I[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= ind_ineq .+ n_tot
    I[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    J[1:n_tot] .= 1:n_tot
    J[n_tot+1:n_tot+n_hess] = hess_sparsity_J
    J[n_tot+n_hess+1:n_tot+n_hess+n_jac] .= jac_sparsity_J
    J[n_tot+n_hess+n_jac+1:n_tot+n_hess+n_jac+n_slack] .= (n+1:n+n_slack)
    J[n_tot+n_hess+n_jac+n_slack+1:offset] .= (n_tot+1:n_tot+m)

    aug_raw = MadNLP.SparseMatrixCOO(aug_vec_length, aug_vec_length, I, J, V)
    aug_com, aug_csc_map = MadNLP.coo_to_csc(aug_raw)

    nnz_csc = SparseArrays.nnz(aug_com)
    aug_com_nzvals = similar(nzVals, nnz_csc, batch_size)
    fill!(aug_com_nzvals, zero(T))

    csc_offsets = similar(aug_csc_map, 1, batch_size)
    csc_offsets .= (0:batch_size-1)' .* nnz_csc
    batch_csc_map = vec(aug_csc_map .+ csc_offsets)

    batch_ls = uniformbatch_linear_solver(aug_com, aug_com_nzvals, aug_vec_length; opt=opt_linear_solver)

    rhs_buffer = VT(undef, aug_vec_length * batch_size)
    fill!(rhs_buffer, zero(T))

    jac_range = n_tot+n_hess+1:n_tot+n_hess+n_jac+n_slack

    hess_scatter, hess_nz_map, hess_var_map, hess_buffer = _build_hess_scatter(
        I, J, n_tot, n_hess, nzVals, aug_csc_map, batch_size,
    )
    jt_scatter, jt_nz_map, jt_con_map, jt_buffer = _build_scatter(
        I, J, jac_range, n_tot, nzVals, aug_csc_map, batch_size,
    )
    jt_con_map_full = similar(jt_con_map)
    jt_con_map_full .= jt_con_map .+ Int32(n_tot)
    j_scatter, _, j_var_map, j_buffer = _build_jac_scatter(
        I, J, jac_range, n_tot, m, nzVals, aug_csc_map, batch_size,
    )

    reg = similar(nzVals, n_tot, batch_size)
    l_diag = similar(nzVals, nlb, batch_size)
    u_diag = similar(nzVals, nub, batch_size)
    l_lower = similar(nzVals, nlb, batch_size)
    u_lower = similar(nzVals, nub, batch_size)

    _mul_w_primal = similar(nzVals, n_tot, batch_size)
    _mul_w_dual = similar(nzVals, m, batch_size)

    batch_map = collect(1:batch_size)
    batch_map_rev = collect(1:batch_size)
    active_batch_size = Ref(batch_size)

    LS = typeof(batch_ls)
    MT = typeof(nzVals)
    VI = typeof(aug_csc_map)
    VI32 = typeof(I)
    SMT = typeof(jt_scatter)
    return SparseUniformBatchKKTSystem{T, LS, VT, MT, VI, VI32, SMT}(
        nzVals, I, J, batch_ls, rhs_buffer, batch_size,
        aug_com_nzvals, batch_csc_map, n_tot, m, n_hess,
        reg, l_diag, u_diag, l_lower, u_lower,
        hess_scatter, hess_nz_map, hess_var_map, hess_buffer,
        jt_scatter, jt_nz_map, jt_con_map, jt_con_map_full, jt_buffer,
        j_scatter, j_var_map, j_buffer,
        _mul_w_primal, _mul_w_dual,
        batch_map, batch_map_rev, active_batch_size,
    )
end

function update_active_set!(bkkt::SparseUniformBatchKKTSystem, status::Vector{MadNLP.Status})
    active_pos = 0
    for i in 1:bkkt.batch_size
        if status[i] == MadNLP.REGULAR
            active_pos += 1
            bkkt.batch_map[i] = active_pos
            bkkt.batch_map_rev[active_pos] = i
        else
            bkkt.batch_map[i] = 0
        end
    end
    for j in (active_pos + 1):bkkt.batch_size
        bkkt.batch_map_rev[j] = 0
    end
    bkkt.active_batch_size[] = active_pos
end

function MadNLP.factorize_kkt!(bkkt::SparseUniformBatchKKTSystem)
    na = bkkt.active_batch_size[]
    nzvals = bkkt.aug_com_nzvals
    @inbounds for j in 1:na
        i = bkkt.batch_map_rev[j]
        i != j && (view(nzvals, :, j) .= view(nzvals, :, i))
    end
    _active_factorize!(bkkt.batch_solver, na)
    return
end

function MadNLP.solve_linear_system!(bkkt::SparseUniformBatchKKTSystem{T}, rhs::AbstractVector) where T
    na = bkkt.active_batch_size[]
    bs = bkkt.batch_size
    n = length(rhs) ÷ bs

    rhs_mat = reshape(rhs, n, bs)
    @inbounds for j in 1:na
        i = bkkt.batch_map_rev[j]
        i != j && (view(rhs_mat, :, j) .= view(rhs_mat, :, i))
    end
    _active_solve!(bkkt.batch_solver, rhs, na, n)

    @inbounds for j in na:-1:1
        i = bkkt.batch_map_rev[j]
        i != j && (view(rhs_mat, :, i) .= view(rhs_mat, :, j))
    end

    @inbounds for i in 1:bs
        bkkt.batch_map[i] == 0 && (view(rhs_mat, :, i) .= zero(T))
    end
    return rhs
end

function MadNLP.reduce_rhs!(bkkt::SparseUniformBatchKKTSystem, d::BatchUnreducedKKTVector)
    MadNLP.reduce_rhs!(
        xp_lr(d), MadNLP.dual_lb(d), bkkt.l_diag,
        xp_ur(d), MadNLP.dual_ub(d), bkkt.u_diag,
    )
    return
end

function MadNLP.finish_aug_solve!(bkkt::SparseUniformBatchKKTSystem, batch_solver::AbstractBatchMPCSolver)
    d = batch_solver.d
    dzl = MadNLP.dual_lb(d)
    dzu = MadNLP.dual_ub(d)
    dzl .= (.-dzl .+ bkkt.l_lower .* xp_lr(d)) ./ bkkt.l_diag
    dzu .= (dzu .- bkkt.u_lower .* xp_ur(d)) ./ bkkt.u_diag
    return
end

function MadNLP.solve_kkt!(bkkt::SparseUniformBatchKKTSystem, batch_solver::AbstractBatchMPCSolver)
    d = batch_solver.d
    n_pd = d.n + d.m
    bs = bkkt.batch_size
    na = bkkt.active_batch_size[]

    MadNLP.reduce_rhs!(bkkt, d)

    if na == 1
        # only one active: skip packing
        orig_col = bkkt.batch_map_rev[1]
        _active_solve!(bkkt.batch_solver, d.views[orig_col], 1, n_pd)
    else
        rhs = bkkt.rhs_buffer
        pd_view = MadNLP.primal_dual(d)
        copyto!(reshape(rhs, n_pd, bs), pd_view)
        MadNLP.solve_linear_system!(bkkt, rhs)
        copyto!(pd_view, reshape(rhs, n_pd, bs))
    end

    MadNLP.finish_aug_solve!(bkkt, batch_solver)
    return
end

function MadNLP.build_kkt!(bkkt::SparseUniformBatchKKTSystem)
    MadNLP._transfer!(vec(bkkt.aug_com_nzvals), vec(bkkt.nzVals), bkkt.batch_csc_map)
    return
end

function MadNLP.factorize_wrapper!(batch_solver::AbstractBatchMPCSolver)
    MadNLP.@trace(batch_solver.logger, "Factorization started.")
    MadNLP.build_kkt!(batch_solver.kkt)
    batch_solver.batch_cnt.linear_solver_time[] += @elapsed MadNLP.factorize_kkt!(batch_solver.kkt)
    return
end

function MadNLP.jtprod!(res::AbstractMatrix, bkkt::SparseUniformBatchKKTSystem, y::BatchVector)
    @views bkkt.jt_buffer .= bkkt.nzVals[bkkt.jt_nz_map, :] .* MadNLP.full(y)[bkkt.jt_con_map, :]
    mul!(res, bkkt.jt_scatter, bkkt.jt_buffer)
    return res
end

function MadNLP.jtprod!(jacl::BatchVector, bkkt::SparseUniformBatchKKTSystem, y::BatchVector)
    return MadNLP.jtprod!(MadNLP.full(jacl), bkkt, y)
end

function MadNLP.eval_jac_wrapper!(
    batch_solver::AbstractBatchMPCSolver,
    bkkt::SparseUniformBatchKKTSystem,
)
    bcb = batch_solver.bcb
    ws = batch_solver.workspace
    nzVals = bkkt.nzVals
    n_tot = bkkt.n_tot
    nnzj = bcb.nnzj
    n_slack = length(bcb.ind_ineq)

    MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
    jac_free = MadNLP._eval_jac_wrapper!(bcb, ws.bx, bcb.jac_buffer)

    jac_offset = n_tot + bkkt.nnzh
    view(nzVals, jac_offset+1:jac_offset+nnzj, :) .= jac_free

    if n_slack > 0
        view(nzVals, jac_offset+nnzj+1:jac_offset+nnzj+n_slack, :) .= -one(eltype(nzVals))
    end
    return
end

function MadNLP.eval_lag_hess_wrapper!(
    batch_solver::AbstractBatchMPCSolver,
    bkkt::SparseUniformBatchKKTSystem,
)
    bcb = batch_solver.bcb
    ws = batch_solver.workspace
    nzVals = bkkt.nzVals
    n_tot = bkkt.n_tot
    nnzh = bkkt.nnzh

    if nnzh > 0
        hess = view(nzVals, n_tot+1:n_tot+nnzh, :)
        MadNLP.unpack_x!(ws.bx, bcb, batch_solver.x)
        MadNLP._eval_lag_hess_wrapper!(bcb, ws.bx, MadNLP.full(batch_solver.y), ws.bv, hess)
    end
    return
end

function MadNLP.initialize!(bkkt::SparseUniformBatchKKTSystem{T}) where T
    pr_diag(bkkt) .= one(T)
    if bkkt.nnzh > 0
        n_tot = bkkt.n_tot
        view(bkkt.nzVals, n_tot+1:n_tot+bkkt.nnzh, :) .= zero(T)
    end
    du_diag(bkkt) .= zero(T)

    fill!(bkkt.reg, zero(T))
    fill!(bkkt.l_diag, one(T))
    fill!(bkkt.u_diag, one(T))
    fill!(bkkt.l_lower, zero(T))
    fill!(bkkt.u_lower, zero(T))

    fill!(bkkt.aug_com_nzvals, zero(T))
    return
end

@inbounds function _kktmul!(
    w::BatchUnreducedKKTVector,
    x::BatchUnreducedKKTVector,
    reg, du_diag_val, l_lower, u_lower, l_diag, u_diag,
    alpha, beta,
)
    MadNLP.primal(w) .+= alpha .* reg .* MadNLP.primal(x)
    MadNLP.dual(w) .+= alpha .* du_diag_val .* MadNLP.dual(x)
    xp_lr(w) .-= alpha .* MadNLP.dual_lb(x)
    xp_ur(w) .+= alpha .* MadNLP.dual_ub(x)
    MadNLP.dual_lb(w) .= beta .* MadNLP.dual_lb(w) .+ alpha .* (xp_lr(x) .* l_lower .- MadNLP.dual_lb(x) .* l_diag)
    MadNLP.dual_ub(w) .= beta .* MadNLP.dual_ub(w) .+ alpha .* (xp_ur(x) .* u_lower .+ MadNLP.dual_ub(x) .* u_diag)
    return
end

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T},
    bkkt::SparseUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T},
    alpha = one(T),
    beta = zero(T),
) where T
    nzV = bkkt.nzVals
    wp = bkkt._mul_w_primal
    wd = bkkt._mul_w_dual

    # mul!(primal(w), Symmetric(hess_com, :L), primal(x), alpha, beta)
    xv = MadNLP.full(x)
    @views bkkt.hess_buffer .= nzV[bkkt.hess_nz_map, :] .* xv[bkkt.hess_var_map, :]
    mul!(wp, bkkt.hess_scatter, bkkt.hess_buffer)
    MadNLP.primal(w) .= beta .* MadNLP.primal(w) .+ alpha .* wp

    # mul!(primal(w), jac_com', dual(x), alpha, one(T))
    @views bkkt.jt_buffer .= nzV[bkkt.jt_nz_map, :] .* xv[bkkt.jt_con_map_full, :]
    mul!(wp, bkkt.jt_scatter, bkkt.jt_buffer)
    MadNLP.primal(w) .+= alpha .* wp

    # mul!(dual(w), jac_com, primal(x), alpha, beta)
    @views bkkt.j_buffer .= nzV[bkkt.jt_nz_map, :] .* xv[bkkt.j_var_map, :]
    mul!(wd, bkkt.j_scatter, bkkt.j_buffer)
    MadNLP.dual(w) .= beta .* MadNLP.dual(w) .+ alpha .* wd
    _kktmul!(w, x, bkkt.reg, du_diag(bkkt), bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag, alpha, beta)
    return w
end
