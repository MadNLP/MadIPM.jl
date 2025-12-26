_batch_matrix(arr, n, batch_size) = reshape(MadNLP._madnlp_unsafe_wrap(arr, n * batch_size, 1), n, batch_size)

struct BatchStepData{T, VT<:AbstractVector{T}, VTC<:AbstractVector{T}}
    x_lr::VT
    xl_r::VT
    dx_lr::VT
    zl_r::VT
    dzl::VT
    x_ur::VT
    xu_r::VT
    dx_ur::VT
    zu_r::VT
    dzu::VT
    work_lb::VT
    work_ub::VT
    alpha_xl::VT
    alpha_xu::VT
    alpha_zl::VT
    alpha_zu::VT
    alpha_p::VT
    alpha_d::VT
    mu_affine::VT
    mu_curr::VT
    mu_new::VT
    sum_lb::VT
    sum_ub::VT
    mu_curr_cpu::VTC
    mu_new_cpu::VTC
    pr_diag::VT
    buffer1::VT
    buffer2::VT
    scaling_factor::VT
    nlb::Int
    nub::Int
    n_tot::Int
    batch_size::Int
end

function BatchStepData(solver::MadIPM.MPCSolver{T,VT}, batch_size::Int) where {T,VT}
    nlb, nub = solver.nlb, solver.nub
    n_tot = length(solver.kkt.pr_diag)
    VTC = Vector{T}
    BatchStepData{T,VT,VTC}(
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, nlb * batch_size), zero(T)),
        fill!(VT(undef, nub * batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), one(T)),
        nlb, nub, n_tot, batch_size,
    )
end
function pack_diag_data!(batch_solver::UniformBatchSolver)
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub

    for_active_withindex(batch_solver, (i, solver) -> begin
        offset_lb = (i - 1) * nlb + 1
        offset_ub = (i - 1) * nub + 1

        if nlb > 0
            copyto!(MadNLP._madnlp_unsafe_wrap(step.x_lr, nlb, offset_lb), solver.x_lr)
            copyto!(MadNLP._madnlp_unsafe_wrap(step.xl_r, nlb, offset_lb), solver.xl_r)
            copyto!(MadNLP._madnlp_unsafe_wrap(step.zl_r, nlb, offset_lb), solver.zl_r)
        end
        if nub > 0
            copyto!(MadNLP._madnlp_unsafe_wrap(step.x_ur, nub, offset_ub), solver.x_ur)
            copyto!(MadNLP._madnlp_unsafe_wrap(step.xu_r, nub, offset_ub), solver.xu_r)
            copyto!(MadNLP._madnlp_unsafe_wrap(step.zu_r, nub, offset_ub), solver.zu_r)
        end
    end)
end

function batch_compute_reg!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub, n_tot = step.nlb, step.nub, step.n_tot

    solver1 = batch_solver[bkkt.batch_map_rev[1]]
    T = eltype(step.x_lr)
    del_w = T(solver1.del_w)

    kkt1 = solver1.kkt
    ind_lb = kkt1.ind_lb
    ind_ub = kkt1.ind_ub

    if nlb > 0
        x_lr_mat = _batch_matrix(step.x_lr, nlb, active_size)
        xl_r_mat = _batch_matrix(step.xl_r, nlb, active_size)
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)

        work_lb_mat .= xl_r_mat .- x_lr_mat
    end

    if nub > 0
        x_ur_mat = _batch_matrix(step.x_ur, nub, active_size)
        xu_r_mat = _batch_matrix(step.xu_r, nub, active_size)
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)

        work_ub_mat .= x_ur_mat .- xu_r_mat
    end

    pr_diag_mat = _batch_matrix(step.pr_diag, n_tot, active_size)
    fill!(pr_diag_mat, del_w)

    if nlb > 0
        zl_r_mat = _batch_matrix(step.zl_r, nlb, active_size)
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)

        pr_diag_mat[ind_lb, :] .-= zl_r_mat ./ work_lb_mat
    end

    if nub > 0
        zu_r_mat = _batch_matrix(step.zu_r, nub, active_size)
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)

        pr_diag_mat[ind_ub, :] .-= zu_r_mat ./ work_ub_mat
    end
end

function batch_compute_reg!(
    batch_solver::UniformBatchSolver{VS,BK}
) where {VS,KKT<:MadNLP.ScaledSparseKKTSystem,BK<:UniformBatchKKTSystem{KKT}}
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub, n_tot = step.nlb, step.nub, step.n_tot

    solver1 = batch_solver[bkkt.batch_map_rev[1]]
    T = eltype(step.x_lr)
    del_w = T(solver1.del_w)

    kkt1 = solver1.kkt
    ind_lb = kkt1.ind_lb
    ind_ub = kkt1.ind_ub

    if nlb > 0
        x_lr_mat = _batch_matrix(step.x_lr, nlb, active_size)
        xl_r_mat = _batch_matrix(step.xl_r, nlb, active_size)
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)

        work_lb_mat .= x_lr_mat .- xl_r_mat
    end

    if nub > 0
        x_ur_mat = _batch_matrix(step.x_ur, nub, active_size)
        xu_r_mat = _batch_matrix(step.xu_r, nub, active_size)
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)

        work_ub_mat .= xu_r_mat .- x_ur_mat
    end

    buffer1_mat = _batch_matrix(step.buffer1, n_tot, active_size)
    buffer2_mat = _batch_matrix(step.buffer2, n_tot, active_size)
    fill!(buffer1_mat, zero(T))
    fill!(buffer2_mat, zero(T))

    if nub > 0
        zu_r_mat = _batch_matrix(step.zu_r, nub, active_size)
        buffer1_mat[ind_ub, :] .= zu_r_mat
    end
    if nlb > 0
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)
        buffer1_mat[ind_lb, :] .*= work_lb_mat
    end

    if nlb > 0
        zl_r_mat = _batch_matrix(step.zl_r, nlb, active_size)
        buffer2_mat[ind_lb, :] .= zl_r_mat
    end
    if nub > 0
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)
        buffer2_mat[ind_ub, :] .*= work_ub_mat
    end

    pr_diag_mat = _batch_matrix(step.pr_diag, n_tot, active_size)
    pr_diag_mat .= buffer1_mat .+ buffer2_mat

    scaling_mat = _batch_matrix(step.scaling_factor, n_tot, active_size)
    fill!(scaling_mat, one(T))

    if nlb > 0
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)
        scaling_mat[ind_lb, :] .*= sqrt.(work_lb_mat)
    end

    if nub > 0
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)
        scaling_mat[ind_ub, :] .*= sqrt.(work_ub_mat)
    end

    pr_diag_mat .+= del_w .* scaling_mat .^ 2
end

function unpack_diag_data!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    step = batch_solver.step
    nlb, nub, n_tot = step.nlb, step.nub, step.n_tot

    solver1 = batch_solver[bkkt.batch_map_rev[1]]
    T = eltype(step.x_lr)
    del_w = T(solver1.del_w)
    del_c = T(solver1.del_c)

    for_active_withindex(batch_solver, (i, solver) -> begin
        kkt = solver.kkt
        offset_lb = (i - 1) * nlb + 1
        offset_ub = (i - 1) * nub + 1
        offset_n_tot = (i - 1) * n_tot + 1

        fill!(kkt.reg, del_w)
        fill!(kkt.du_diag, del_c)

        if nlb > 0
            copyto!(kkt.l_diag, MadNLP._madnlp_unsafe_wrap(step.work_lb, nlb, offset_lb))
            copyto!(kkt.l_lower, MadNLP._madnlp_unsafe_wrap(step.zl_r, nlb, offset_lb))
        end

        if nub > 0
            copyto!(kkt.u_diag, MadNLP._madnlp_unsafe_wrap(step.work_ub, nub, offset_ub))
            copyto!(kkt.u_lower, MadNLP._madnlp_unsafe_wrap(step.zu_r, nub, offset_ub))
        end

        copyto!(kkt.pr_diag, MadNLP._madnlp_unsafe_wrap(step.pr_diag, n_tot, offset_n_tot))
    end)
end

function unpack_diag_data!(
    batch_solver::UniformBatchSolver{VS,BK}
) where {VS,KKT<:MadNLP.ScaledSparseKKTSystem,BK<:UniformBatchKKTSystem{KKT}}
    bkkt = batch_solver.bkkt
    step = batch_solver.step
    nlb, nub, n_tot = step.nlb, step.nub, step.n_tot

    solver1 = batch_solver[bkkt.batch_map_rev[1]]
    T = eltype(step.x_lr)
    del_w = T(solver1.del_w)
    del_c = T(solver1.del_c)

    for_active_withindex(batch_solver, (i, solver) -> begin
        kkt = solver.kkt
        offset_lb = (i - 1) * nlb + 1
        offset_ub = (i - 1) * nub + 1
        offset_n_tot = (i - 1) * n_tot + 1

        fill!(kkt.reg, del_w)
        fill!(kkt.du_diag, del_c)

        if nlb > 0
            copyto!(kkt.l_diag, MadNLP._madnlp_unsafe_wrap(step.work_lb, nlb, offset_lb))
            copyto!(kkt.l_lower, MadNLP._madnlp_unsafe_wrap(step.zl_r, nlb, offset_lb))
        end

        if nub > 0
            copyto!(kkt.u_diag, MadNLP._madnlp_unsafe_wrap(step.work_ub, nub, offset_ub))
            copyto!(kkt.u_lower, MadNLP._madnlp_unsafe_wrap(step.zu_r, nub, offset_ub))
        end

        copyto!(kkt.pr_diag, MadNLP._madnlp_unsafe_wrap(step.pr_diag, n_tot, offset_n_tot))
        copyto!(kkt.scaling_factor, MadNLP._madnlp_unsafe_wrap(step.scaling_factor, n_tot, offset_n_tot))
    end)
end

function batch_set_aug_diagonal_reg!(batch_solver::UniformBatchSolver)
    pack_diag_data!(batch_solver)
    batch_compute_reg!(batch_solver)
    unpack_diag_data!(batch_solver)
    return
end
