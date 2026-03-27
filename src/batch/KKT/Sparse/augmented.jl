"""Batched version of SparseKKTSystem"""
struct SparseUniformBatchKKTSystem{T, LS, MT, VI, VI32, OPT} <: AbstractBatchKKTSystem{T}
    nzVals::MT              # (aug_mat_length × batch_size) COO nonzero values
    aug_I::VI32             # shared row indices
    aug_J::VI32             # shared column indices
    batch_solver::LS        # batched linear solver
    rhs_buffer::MT          # (n+m) × batch_size for batch solve
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
    # Operators for batch SpMV (jtprod! and KKT mul!)
    hess_op::OPT
    jt_op::OPT
    j_op::OPT
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
    bcb::UniformBatchCallback{T, VT, MT, VI},
    uniformbatch_linear_solver = LoopedBatchLinearSolver;
    opt_linear_solver = MadNLP.default_options(uniformbatch_linear_solver),
) where {T, VT, MT, VI}
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

    rhs_buffer = similar(nzVals, aug_vec_length, batch_size)
    fill!(rhs_buffer, zero(T))

    jac_range = n_tot+n_hess+1:n_tot+n_hess+n_jac+n_slack

    hess_op = _build_hess_op(I, J, n_tot, n_hess, nzVals, aug_csc_map)
    jt_op = _build_jt_op(I, J, jac_range, n_tot, nzVals, aug_csc_map)
    j_op = _build_j_op(I, J, jac_range, n_tot, m, nzVals, aug_csc_map)

    reg = similar(nzVals, n_tot, batch_size)
    l_diag = similar(nzVals, nlb, batch_size)
    u_diag = similar(nzVals, nub, batch_size)
    l_lower = similar(nzVals, nlb, batch_size)
    u_lower = similar(nzVals, nub, batch_size)

    batch_map = collect(1:batch_size)
    batch_map_rev = collect(1:batch_size)
    active_batch_size = Ref(batch_size)

    LS = typeof(batch_ls)
    VI32 = typeof(I)
    OPT = typeof(jt_op)
    return SparseUniformBatchKKTSystem{T, LS, MT, VI, VI32, OPT}(
        nzVals, I, J, batch_ls, rhs_buffer, batch_size,
        aug_com_nzvals, batch_csc_map, n_tot, m, n_hess,
        reg, l_diag, u_diag, l_lower, u_lower,
        hess_op, jt_op, j_op,
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
    @inbounds for j in 1:na  # FIXME: refactor to avoid `na` launches
        i = bkkt.batch_map_rev[j]
        i != j && (view(nzvals, :, j) .= view(nzvals, :, i))
    end
    _active_factorize!(bkkt.batch_solver, na)
    return
end

function MadNLP.solve_linear_system!(bkkt::SparseUniformBatchKKTSystem{T}, rhs::AbstractMatrix) where T
    na = bkkt.active_batch_size[]
    bs = bkkt.batch_size
    n = size(rhs, 1)

    @inbounds for j in 1:na
        i = bkkt.batch_map_rev[j]
        i != j && (view(rhs, :, j) .= view(rhs, :, i))
    end
    _active_solve!(bkkt.batch_solver, rhs, na, n)

    @inbounds for j in na:-1:1
        i = bkkt.batch_map_rev[j]
        i != j && (view(rhs, :, i) .= view(rhs, :, j))
    end

    @inbounds for i in 1:bs
        bkkt.batch_map[i] == 0 && (view(rhs, :, i) .= zero(T))
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

    MadNLP.reduce_rhs!(bkkt, d)

    rhs = bkkt.rhs_buffer
    pd_view = MadNLP.primal_dual(d)
    copyto!(rhs, pd_view)
    MadNLP.solve_linear_system!(bkkt, rhs)
    copyto!(pd_view, rhs)

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
    batch_spmv!(res, bkkt.jt_op, MadNLP.full(y))
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

function LinearAlgebra.mul!(
    w::BatchUnreducedKKTVector{T},
    bkkt::SparseUniformBatchKKTSystem{T},
    x::BatchUnreducedKKTVector{T},
    alpha = one(T),
    beta = zero(T),
) where T
    xv = MadNLP.full(x)
    # mul!(primal(w), Symmetric(kkt.hess_com, :L), primal(x), alpha, beta)
    batch_spmv!(MadNLP.primal(w), bkkt.hess_op, xv, alpha, beta)
    # mul!(primal(w), kkt.jac_com', dual(x), alpha, one(T))
    batch_spmv!(MadNLP.primal(w), bkkt.jt_op, xv, alpha, one(T); val_offset=bkkt.n_tot)
    # mul!(dual(w), kkt.jac_com,  primal(x), alpha, beta)
    batch_spmv!(MadNLP.dual(w), bkkt.j_op, xv, alpha, beta)
    _kktmul!(w, x, bkkt.reg, du_diag(bkkt), bkkt.l_lower, bkkt.u_lower, bkkt.l_diag, bkkt.u_diag, alpha, beta)
    return w
end
