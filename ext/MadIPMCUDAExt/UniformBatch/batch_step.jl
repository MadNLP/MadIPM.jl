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
