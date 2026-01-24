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
    tau::VT
    mu::VT
    mu_curr_cpu::VTC
    mu_new_cpu::VTC
    alpha_p_cpu::VTC
    alpha_d_cpu::VTC
    mu_cpu::VTC
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
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VT(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VTC(undef, batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
        fill!(VT(undef, n_tot * batch_size), zero(T)),
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

function pack_prediction_step_data!(batch_solver::UniformBatchSolver)
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub

    for_active_withindex(batch_solver, (i, solver) -> begin
        offset_lb = (i - 1) * nlb + 1
        offset_ub = (i - 1) * nub + 1

        for (dest, src, is_l) in [
            (step.x_lr, solver.x_lr, true),
            (step.xl_r, solver.xl_r, true),
            (step.dx_lr, solver.dx_lr, true),
            (step.zl_r, solver.zl_r, true),
            (step.dzl, MadNLP.dual_lb(solver.d), true),
            (step.x_ur, solver.x_ur, false),
            (step.xu_r, solver.xu_r, false),
            (step.dx_ur, solver.dx_ur, false),
            (step.zu_r, solver.zu_r, false),
            (step.dzu, MadNLP.dual_ub(solver.d), false),
        ]
            n = is_l ? nlb : nub
            n == 0 && continue
            offset = is_l ? offset_lb : offset_ub

            copyto!(MadNLP._madnlp_unsafe_wrap(dest, n, offset), src)
        end
    end)
end

function batch_get_fraction_to_boundary_step!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub
    T = eltype(batch_solver)
    inf_val = T(Inf)

    alpha_xl = MadNLP._madnlp_unsafe_wrap(step.alpha_xl, active_size, 1)
    alpha_zl = MadNLP._madnlp_unsafe_wrap(step.alpha_zl, active_size, 1)
    alpha_xu = MadNLP._madnlp_unsafe_wrap(step.alpha_xu, active_size, 1)
    alpha_zu = MadNLP._madnlp_unsafe_wrap(step.alpha_zu, active_size, 1)

    tau_vec = MadNLP._madnlp_unsafe_wrap(step.tau, active_size, 1)

    if nlb > 0
        x_lr_mat = _batch_matrix(step.x_lr, nlb, active_size)
        xl_r_mat = _batch_matrix(step.xl_r, nlb, active_size)
        dx_lr_mat = _batch_matrix(step.dx_lr, nlb, active_size)
        zl_r_mat = _batch_matrix(step.zl_r, nlb, active_size)
        dzl_mat = _batch_matrix(step.dzl, nlb, active_size)

        alpha_xl .= tau_vec .* vec(mapreduce(
            (dx, xl, x, tau) -> ifelse(dx < 0, (xl - x) / dx, inf_val),
            min, dx_lr_mat, xl_r_mat, x_lr_mat; dims=1, init=inf_val))

        alpha_zl .= tau_vec .* vec(mapreduce(
            (dz, z, tau) -> ifelse(dz < 0, -z / dz, inf_val),
            min, dzl_mat, zl_r_mat; dims=1, init=inf_val))
    else
        fill!(alpha_xl, one(T))
        fill!(alpha_zl, one(T))
    end

    if nub > 0
        x_ur_mat = _batch_matrix(step.x_ur, nub, active_size)
        xu_r_mat = _batch_matrix(step.xu_r, nub, active_size)
        dx_ur_mat = _batch_matrix(step.dx_ur, nub, active_size)
        zu_r_mat = _batch_matrix(step.zu_r, nub, active_size)
        dzu_mat = _batch_matrix(step.dzu, nub, active_size)

        alpha_xu .= tau_vec .* vec(mapreduce(
            (dx, xu, x, tau) -> ifelse(dx > 0, (xu - x) / dx, inf_val),
            min, dx_ur_mat, xu_r_mat, x_ur_mat; dims=1, init=inf_val))

        alpha_zu .= tau_vec .* vec(mapreduce(
            (dz, z, tau) -> ifelse((dz < 0) & (z + dz < 0), -z / dz, inf_val),
            min, dzu_mat, zu_r_mat; dims=1, init=inf_val))
    else
        fill!(alpha_xu, one(T))
        fill!(alpha_zu, one(T))
    end

    alpha_p = MadNLP._madnlp_unsafe_wrap(step.alpha_p, active_size, 1)
    alpha_d = MadNLP._madnlp_unsafe_wrap(step.alpha_d, active_size, 1)

    alpha_p .= min.(alpha_xl, alpha_xu)
    alpha_d .= min.(alpha_zl, alpha_zu)
end

function batch_get_affine_complementarity_measure!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub
    T = eltype(step.x_lr)

    sum_lb = MadNLP._madnlp_unsafe_wrap(step.sum_lb, active_size, 1)
    sum_ub = MadNLP._madnlp_unsafe_wrap(step.sum_ub, active_size, 1)
    alpha_p = MadNLP._madnlp_unsafe_wrap(step.alpha_p, active_size, 1)
    alpha_d = MadNLP._madnlp_unsafe_wrap(step.alpha_d, active_size, 1)

    # Reshape alphas to 1×N for broadcasting across rows
    alpha_p_row = reshape(alpha_p, 1, active_size)
    alpha_d_row = reshape(alpha_d, 1, active_size)

    if nlb > 0
        x_lr_mat = _batch_matrix(step.x_lr, nlb, active_size)
        xl_r_mat = _batch_matrix(step.xl_r, nlb, active_size)
        dx_lr_mat = _batch_matrix(step.dx_lr, nlb, active_size)
        zl_r_mat = _batch_matrix(step.zl_r, nlb, active_size)
        dzl_mat = _batch_matrix(step.dzl, nlb, active_size)

        sum_lb .= vec(mapreduce(
            (x, xl, dx, z, dz, ap, ad) -> ((x + ap * dx) - xl) * (z + ad * dz),
            +, x_lr_mat, xl_r_mat, dx_lr_mat, zl_r_mat, dzl_mat, alpha_p_row, alpha_d_row;
            dims=1, init=zero(T)))
    else
        fill!(sum_lb, zero(T))
    end

    if nub > 0
        xu_r_mat = _batch_matrix(step.xu_r, nub, active_size)
        x_ur_mat = _batch_matrix(step.x_ur, nub, active_size)
        dx_ur_mat = _batch_matrix(step.dx_ur, nub, active_size)
        zu_r_mat = _batch_matrix(step.zu_r, nub, active_size)
        dzu_mat = _batch_matrix(step.dzu, nub, active_size)

        sum_ub .= vec(mapreduce(
            (xu, x, dx, z, dz, ap, ad) -> (xu - (x + ap * dx)) * (z + ad * dz),
            +, xu_r_mat, x_ur_mat, dx_ur_mat, zu_r_mat, dzu_mat, alpha_p_row, alpha_d_row;
            dims=1, init=zero(T)))
    else
        fill!(sum_ub, zero(T))
    end

    mu_affine = MadNLP._madnlp_unsafe_wrap(step.mu_affine, active_size, 1)
    mu_affine .= (sum_lb .+ sum_ub) ./ (nlb + nub)
end

function batch_get_correction!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub

    if nlb > 0
        dx_lr_mat = _batch_matrix(step.dx_lr, nlb, active_size)
        dzl_mat = _batch_matrix(step.dzl, nlb, active_size)
        work_lb_mat = _batch_matrix(step.work_lb, nlb, active_size)

        work_lb_mat .= dx_lr_mat .* dzl_mat
    end

    if nub > 0
        dx_ur_mat = _batch_matrix(step.dx_ur, nub, active_size)
        dzu_mat = _batch_matrix(step.dzu, nub, active_size)
        work_ub_mat = _batch_matrix(step.work_ub, nub, active_size)

        work_ub_mat .= dx_ur_mat .* dzu_mat
    end

    for_active_withindex(batch_solver, (i, solver) -> begin
        if nlb > 0
            copyto!(solver.correction_lb, MadNLP._madnlp_unsafe_wrap(step.work_lb, nlb, ((i - 1) * nlb)+1))
        end
        if nub > 0
            copyto!(solver.correction_ub, MadNLP._madnlp_unsafe_wrap(step.work_ub, nub, ((i - 1) * nub)+1))
        end
        return
    end)
end

function batch_get_complementarity_measure!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    nlb, nub = step.nlb, step.nub
    T = eltype(step.x_lr)

    sum_lb = MadNLP._madnlp_unsafe_wrap(step.sum_lb, active_size, 1)
    sum_ub = MadNLP._madnlp_unsafe_wrap(step.sum_ub, active_size, 1)

    if nlb > 0
        x_lr_mat = _batch_matrix(step.x_lr, nlb, active_size)
        xl_r_mat = _batch_matrix(step.xl_r, nlb, active_size)
        zl_r_mat = _batch_matrix(step.zl_r, nlb, active_size)

        sum_lb .= vec(mapreduce(
            (x, xl, z) -> (x - xl) * z,
            +, x_lr_mat, xl_r_mat, zl_r_mat; dims=1, init=zero(T)))
    else
        fill!(sum_lb, zero(T))
    end

    if nub > 0
        x_ur_mat = _batch_matrix(step.x_ur, nub, active_size)
        xu_r_mat = _batch_matrix(step.xu_r, nub, active_size)
        zu_r_mat = _batch_matrix(step.zu_r, nub, active_size)

        sum_ub .= vec(mapreduce(
            (xu, x, z) -> (xu - x) * z,
            +, xu_r_mat, x_ur_mat, zu_r_mat; dims=1, init=zero(T)))
    else
        fill!(sum_ub, zero(T))
    end

    mu_curr = MadNLP._madnlp_unsafe_wrap(step.mu_curr, active_size, 1)
    mu_curr .= (sum_lb .+ sum_ub) ./ (nlb + nub)
end

function batch_update_barrier!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step

    solver1 = batch_solver[bkkt.batch_map_rev[1]]# FIXME: global options
    mu_min = solver1.opt.mu_min
    has_inequalities = (length(solver1.ind_llb) + length(solver1.ind_uub)) > 0

    batch_get_complementarity_measure!(batch_solver)
    mu_curr = MadNLP._madnlp_unsafe_wrap(step.mu_curr, active_size, 1)
    mu = MadNLP._madnlp_unsafe_wrap(step.mu_new, active_size, 1)

    T = eltype(mu)
    if has_inequalities
        mu_affine = MadNLP._madnlp_unsafe_wrap(step.mu_affine, active_size, 1)
        mu .= clamp.((mu_affine ./ mu_curr) .^ 3, T(1e-6), T(10.0))
        mu .= max.(T(mu_min), mu .* mu_curr)
    else
        mu .= max.(T(mu_min), mu_curr)
    end

    mu_curr_cpu = MadNLP._madnlp_unsafe_wrap(step.mu_curr_cpu, active_size, 1)
    mu_cpu = MadNLP._madnlp_unsafe_wrap(step.mu_new_cpu, active_size, 1)
    copyto!(mu_curr_cpu, mu_curr)
    copyto!(mu_cpu, mu)

    for_active_withindex(batch_solver, (i, solver) -> begin
        solver.mu_curr = mu_curr_cpu[i]
        solver.mu = mu_cpu[i]
    end)
end

function batch_prediction_step_size!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    pack_prediction_step_data!(batch_solver)
    fill!(MadNLP._madnlp_unsafe_wrap(step.tau, active_size, 1), one(eltype(batch_solver)))
    batch_get_fraction_to_boundary_step!(batch_solver)
    batch_get_affine_complementarity_measure!(batch_solver)
    batch_get_correction!(batch_solver)
    batch_update_barrier!(batch_solver)  # NOTE: there is only the Mehrotra rule, we ignore solver.opt.barrier_update
    return
end

batch_func(batch_solver::UniformBatchSolver, ::typeof(MadIPM.prediction_step_size!)) =
    batch_prediction_step_size!(batch_solver)

function batch_update_step_size!(batch_solver::UniformBatchSolver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step
    
    rule = batch_solver[bkkt.batch_map_rev[1]].opt.step_rule# FIXME: global options
    pack_prediction_step_data!(batch_solver)
    batch_set_tau!(rule, batch_solver)
    batch_get_fraction_to_boundary_step!(batch_solver)

    alpha_p = MadNLP._madnlp_unsafe_wrap(step.alpha_p, active_size, 1)
    alpha_d = MadNLP._madnlp_unsafe_wrap(step.alpha_d, active_size, 1)

    alpha_p_cpu = MadNLP._madnlp_unsafe_wrap(step.alpha_p_cpu, active_size, 1)
    alpha_d_cpu = MadNLP._madnlp_unsafe_wrap(step.alpha_d_cpu, active_size, 1)

    copyto!(alpha_p_cpu, alpha_p)
    copyto!(alpha_d_cpu, alpha_d)


    for_active_withindex(batch_solver, (i, solver) -> begin
        solver.alpha_p = alpha_p_cpu[i]
        solver.alpha_d = alpha_d_cpu[i]
    end)
end

# TODO: add MehrotraAdaptiveStep
function batch_set_tau!(rule::MadIPM.AdaptiveStep, batch_solver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step

    tau = MadNLP._madnlp_unsafe_wrap(step.tau, active_size, 1)
    mu = MadNLP._madnlp_unsafe_wrap(step.mu, active_size, 1)
    mu_cpu = MadNLP._madnlp_unsafe_wrap(step.mu_cpu, active_size, 1)

    for_active_withindex(batch_solver, (i, solver) -> begin
        mu_cpu[i] = solver.mu
    end)
    copyto!(mu, mu_cpu)
    tau .= max.(1 .- mu, rule.tau_min)
end
function batch_set_tau!(rule::MadIPM.ConservativeStep, batch_solver)
    bkkt = batch_solver.bkkt
    active_size = bkkt.active_batch_size[]
    step = batch_solver.step

    tau = MadNLP._madnlp_unsafe_wrap(step.tau, active_size, 1)
    fill!(tau, rule.tau)
end

batch_func(batch_solver::UniformBatchSolver, ::typeof(MadIPM.update_step_size!)) =
    batch_update_step_size!(batch_solver)