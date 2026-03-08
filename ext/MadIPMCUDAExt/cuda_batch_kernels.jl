@kernel function _batch_set_con_scale_sparse_kernel!(
    con_scale,
    @Const(ptr),
    @Const(inds),
    @Const(jac_buffer),
)
    (index, j) = @index(Global, NTuple)
    @inbounds begin
        rng = ptr[index]:ptr[index+1]-1
        for k in rng
            (row, i) = inds[k]
            con_scale[row, j] = max(con_scale[row, j], abs(jac_buffer[i, j]))
        end
    end
end

function MadNLP._set_con_scale_sparse!(
    con_scale::CuMatrix{T},
    jac_I::CuVector{<:Integer},
    jac_buffer::CuMatrix{T},
) where T
    ind_jac = CuVector{Int}(1:length(jac_I))
    inds = map((i, j) -> (i, j), jac_I, ind_jac)
    !isempty(inds) && sort!(inds)
    ptr = getptr(inds; by = ((x1, x2), (y1, y2)) -> x1 != y1)
    if length(ptr) > 1
        backend = CUDABackend()
        _batch_set_con_scale_sparse_kernel!(backend)(
            con_scale,
            ptr,
            inds,
            jac_buffer;
            ndrange = (length(ptr) - 1, size(con_scale, 2)),
        )
        KernelAbstractions.synchronize(backend)
    end
    return con_scale
end

@kernel function _block_argmin_kernel!(out_val, out_idx, @Const(parent_data), offset, nrows)
    tid = @index(Local, Linear)
    j = @index(Group, Linear)
    gs = @groupsize()[1]
    T = eltype(out_val)

    sval = @localmem T (64,)
    sidx = @localmem Int32 (64,)

    # Strided scan
    local_min = T(Inf)
    local_idx = Int32(0)
    @inbounds begin
        i = Int32(tid)
        while i <= nrows
            v = parent_data[offset + i, j]
            if v < local_min
                local_min = v
                local_idx = i
            end
            i += Int32(gs)
        end
        sval[tid] = local_min
        sidx[tid] = local_idx
    end
    @synchronize()

    # Tree reduction
    @inbounds begin
        stride = Int32(gs) >> Int32(1)
        while stride > Int32(0)
            if Int32(tid) <= stride
                if sval[tid + stride] < sval[tid]
                    sval[tid] = sval[tid + stride]
                    sidx[tid] = sidx[tid + stride]
                end
            end
            @synchronize()
            stride >>= Int32(1)
        end

        if tid == 1
            out_val[1, j] = sval[1]
            out_idx[1, j] = sidx[1]
        end
    end
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

const _MEHROTRA_BLOCK = 64

function MadIPM._argmin_columns!(
    out_val::CuMatrix{T}, out_idx::CuMatrix{Int32},
    parent_data::CuMatrix{T}, offset::Int, nrows::Int; threads_per_column = _MEHROTRA_BLOCK
) where T
    ncols = size(out_val, 2)
    if ncols > 0 && nrows > 0
        backend = CUDABackend()
        _block_argmin_kernel!(backend, threads_per_column)(
            out_val, out_idx, parent_data, Int32(offset), Int32(nrows);
            ndrange = threads_per_column * ncols,
        )
        KernelAbstractions.synchronize(backend)
    end
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
