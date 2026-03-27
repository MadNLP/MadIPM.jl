@kernel function _set_con_scale_kernel!(con_scale, @Const(jac_I), @Const(jac_buffer))
    k = @index(Global, Linear)
    row = jac_I[k]
    bs = size(jac_buffer, 2)
    @inbounds for j in 1:bs
        val = abs(jac_buffer[k, j])
        # CAS loop for atomic max (CUDA.atomic_max! only supports integers)
        old = con_scale[row, j]
        while val > old
            result = Atomix.@atomicreplace con_scale[row, j] old => val
            old = result.old
            if result.success
                break
            end
        end
    end
end

@inline function _atomic_colreduce!(::typeof(+), out, j, value)
    Atomix.@atomic out[1, j] += value
    return
end

@inline function _atomic_colreduce!(::typeof(min), out, j, value)
    old = out[1, j]
    while value < old
        result = Atomix.@atomicreplace out[1, j] old => value
        old = result.old
        if result.success
            break
        end
    end
    return
end

function MadNLP._set_con_scale_sparse!(
    con_scale::CuMatrix{T},
    jac_I::CuVector{<:Integer},
    jac_buffer::CuMatrix{T},
) where T
    nnzj = length(jac_I)
    if nnzj > 0
        backend = CUDABackend()
        _set_con_scale_kernel!(backend)(con_scale, jac_I, jac_buffer; ndrange=nnzj)
        KernelAbstractions.synchronize(backend)
    end
    return con_scale
end

@kernel function _mehrotra_correction_kernel!(
    alpha_p, alpha_d,
    @Const(mu),
    @Const(val_xl), @Const(idx_xl), @Const(val_xu), @Const(idx_xu),
    @Const(val_zl), @Const(idx_zl), @Const(val_zu), @Const(idx_zu),
    @Const(d_vals), @Const(x_vals), @Const(xl_vals), @Const(xu_vals),
    @Const(zl_vals), @Const(zu_vals),
    @Const(ind_lb), @Const(ind_ub),
    dlb_off, dub_off, gamma_f,
)
    j = @index(Global, Linear)
    T = eltype(alpha_p)

    mu_j = mu[1, j]
    max_ap = alpha_p[1, j]
    max_ad = alpha_d[1, j]

    # primal step
    corrected_p = one(T)
    @inbounds if max_ap < one(T)
        i_xl = idx_xl[1, j]
        i_xu = idx_xu[1, j]
        if val_xl[1, j] <= val_xu[1, j] && i_xl > Int32(0)
            idx = ind_lb[i_xl]
            zl_stepped = zl_vals[idx, j] + max_ad * d_vals[dlb_off + i_xl, j]
            corrected_p = (x_vals[idx, j] - xl_vals[idx, j] - mu_j / zl_stepped) / (-d_vals[idx, j])
        elseif i_xu > Int32(0)
            idx = ind_ub[i_xu]
            zu_stepped = zu_vals[idx, j] + max_ad * d_vals[dub_off + i_xu, j]
            corrected_p = (xu_vals[idx, j] - x_vals[idx, j] - mu_j / zu_stepped) / d_vals[idx, j]
        end
    end
    @inbounds alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

    # dual step
    corrected_d = one(T)
    @inbounds if max_ad < one(T)
        i_zl = idx_zl[1, j]
        i_zu = idx_zu[1, j]
        if val_zl[1, j] <= val_zu[1, j] && i_zl > Int32(0)
            idx = ind_lb[i_zl]
            x_gap = x_vals[idx, j] + max_ap * d_vals[idx, j] - xl_vals[idx, j]
            corrected_d = -(zl_vals[idx, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
        elseif i_zu > Int32(0)
            idx = ind_ub[i_zu]
            x_gap = xu_vals[idx, j] - x_vals[idx, j] - max_ap * d_vals[idx, j]
            corrected_d = -(zu_vals[idx, j] - mu_j / x_gap) / d_vals[dub_off + i_zu, j]
        end
    end
    @inbounds alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
end

@kernel function _gather_compl_kernel!(
    scratch, @Const(x_vals), @Const(xb_vals), @Const(z_vals), @Const(ind),
)
    i, j = @index(Global, NTuple)
    @inbounds begin
        idx = ind[i]
        scratch[i, j] = abs(x_vals[idx, j] - xb_vals[idx, j]) * z_vals[idx, j]
    end
end

function MadIPM.get_inf_compl!(
    inf_compl::CuMatrix, x::MadIPM.BatchPrimalVector, xl::MadIPM.BatchPrimalVector,
    zl::MadIPM.BatchPrimalVector, xu::MadIPM.BatchPrimalVector, zu::MadIPM.BatchPrimalVector,
    scratch_lb::CuMatrix, scratch_ub::CuMatrix, sum_lb, sum_ub, nlb, nub,
)
    T = eltype(inf_compl)
    bs = size(inf_compl, 2)
    backend = CUDABackend()
    if nlb > 0
        _gather_compl_kernel!(backend)(
            scratch_lb, x.values, xl.values, zl.values, x.ind_lb;
            ndrange=(nlb, bs),
        )
        KernelAbstractions.synchronize(backend)
        maximum!(sum_lb, scratch_lb)
    else
        fill!(sum_lb, zero(T))
    end
    if nub > 0
        _gather_compl_kernel!(backend)(
            scratch_ub, xu.values, x.values, zu.values, x.ind_ub;
            ndrange=(nub, bs),
        )
        KernelAbstractions.synchronize(backend)
        maximum!(sum_ub, scratch_ub)
    else
        fill!(sum_ub, zero(T))
    end
    @. inf_compl = max(sum_lb, sum_ub)
    return inf_compl
end

function MadIPM._mehrotra_correct_steps!(
    alpha_p::CuMatrix{T}, alpha_d::CuMatrix{T}, mu,
    val_xl, idx_xl, val_xu, idx_xu,
    val_zl, idx_zl, val_zu, idx_zu,
    d_vals, x_vals, xl_vals, xu_vals, zl_vals, zu_vals,
    ind_lb, ind_ub, dlb_off::Int, dub_off::Int, gamma_f,
) where T
    bs = size(alpha_p, 2)
    if bs > 0
        backend = CUDABackend()
        _mehrotra_correction_kernel!(backend)(
            alpha_p, alpha_d, mu,
            val_xl, idx_xl, val_xu, idx_xu,
            val_zl, idx_zl, val_zu, idx_zu,
            d_vals, x_vals, xl_vals, xu_vals, zl_vals, zu_vals,
            ind_lb, ind_ub,
            Int32(dlb_off), Int32(dub_off), gamma_f;
            ndrange = bs,
        )
        KernelAbstractions.synchronize(backend)
    end
end

@kernel function _gather_mul_kernel!(out, @Const(A), @Const(nz_map), @Const(B), @Const(val_map))
    i, j = @index(Global, NTuple)
    @inbounds out[i, j] = A[nz_map[i], j] * B[val_map[i], j]
end

function MadIPM._gather_mul!(
    out::CuMatrix, A::CuMatrix, nz_map::CuVector, B::CuMatrix, val_map::CuVector,
)
    n, bs = size(out)
    if n > 0
        backend = CUDABackend()
        _gather_mul_kernel!(backend)(out, A, nz_map, B, val_map; ndrange=(n, bs))
        KernelAbstractions.synchronize(backend)
    end
    return out
end
