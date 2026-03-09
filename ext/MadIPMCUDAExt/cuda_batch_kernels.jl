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
