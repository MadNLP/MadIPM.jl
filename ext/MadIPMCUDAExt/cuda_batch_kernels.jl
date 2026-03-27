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

@kernel function _gather_batch_view_columns_kernel!(dst, @Const(src), @Const(local_to_root))
    i, j = @index(Global, NTuple)
    @inbounds dst[i, j] = src[i, local_to_root[j]]
end

@kernel function _scatter_batch_view_columns_kernel!(dst, @Const(src), @Const(local_to_root))
    i, j = @index(Global, NTuple)
    @inbounds dst[i, local_to_root[j]] = src[i, j]
end

@kernel function _compact_active_columns_inplace_kernel!(dst, @Const(local_to_root))
    i, j = @index(Global, NTuple)
    src_j = local_to_root[j]
    @inbounds if src_j != j
        dst[i, j] = dst[i, src_j]
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

function MadIPM.gather_batch_view_columns!(
    dst::CuMatrix{TD},
    src::CuMatrix{TS},
    batch_view::MadIPM.BatchView,
) where {TD, TS}
    na = MadIPM.local_batch_size(batch_view)
    backend = CUDABackend()
    _gather_batch_view_columns_kernel!(backend)(dst, src, MadIPM.local_to_root_dev(batch_view); ndrange=(size(dst, 1), na))
    return dst
end

function MadIPM.scatter_batch_view_columns!(
    dst::CuMatrix{TD},
    src::CuMatrix{TS},
    batch_view::MadIPM.BatchView,
) where {TD, TS}
    na = MadIPM.local_batch_size(batch_view)
    backend = CUDABackend()
    _scatter_batch_view_columns_kernel!(backend)(dst, src, MadIPM.local_to_root_dev(batch_view); ndrange=(size(src, 1), na))
    return dst
end

function MadIPM.compact_active_columns_inplace!(
    dst::CuMatrix{T},
    batch_view::MadIPM.BatchView,
) where T
    na = MadIPM.local_batch_size(batch_view)
    backend = CUDABackend()
    _compact_active_columns_inplace_kernel!(backend)(dst, MadIPM.local_to_root_dev(batch_view); ndrange=(size(dst, 1), na))
    return dst
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
    end
    return con_scale
end

_ftb_primal_lb_kernel!(alpha_out, dx, x, xb, tau, nrows) = begin
    bs = size(alpha_out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(alpha_out)
    @inbounds if j <= bs
        a = T(Inf); τ = tau[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            d = dx[i, j]
            d < zero(T) && (a = min(a, (-x[i, j] + xb[i, j]) * τ / d))
            i += blockDim().x * gridDim_reduce
        end
        a = CUDA.reduce_block(min, a, T(Inf), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(min, alpha_out, j, a)
    end
    return
end

_ftb_primal_ub_kernel!(alpha_out, dx, x, xb, tau, nrows) = begin
    bs = size(alpha_out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(alpha_out)
    @inbounds if j <= bs
        a = T(Inf); τ = tau[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            d = dx[i, j]
            d > zero(T) && (a = min(a, (-x[i, j] + xb[i, j]) * τ / d))
            i += blockDim().x * gridDim_reduce
        end
        a = CUDA.reduce_block(min, a, T(Inf), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(min, alpha_out, j, a)
    end
    return
end

_ftb_dual_lb_kernel!(alpha_out, dz, z, tau, nrows) = begin
    bs = size(alpha_out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(alpha_out)
    @inbounds if j <= bs
        a = T(Inf); τ = tau[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            d = dz[i, j]
            d < zero(T) && (a = min(a, -z[i, j] * τ / d))
            i += blockDim().x * gridDim_reduce
        end
        a = CUDA.reduce_block(min, a, T(Inf), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(min, alpha_out, j, a)
    end
    return
end

_ftb_dual_ub_kernel!(alpha_out, dz, z, tau, nrows) = begin
    bs = size(alpha_out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(alpha_out)
    @inbounds if j <= bs
        a = T(Inf); τ = tau[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            d = dz[i, j]
            (d < zero(T) && z[i, j] + d < zero(T)) && (a = min(a, -z[i, j] * τ / d))
            i += blockDim().x * gridDim_reduce
        end
        a = CUDA.reduce_block(min, a, T(Inf), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(min, alpha_out, j, a)
    end
    return
end

function _launch_ftb_kernel!(kernel_fn, alpha_out, nrows, srcs...)
    T = eltype(alpha_out)
    bs = size(alpha_out, 2)
    fill!(alpha_out, T(Inf))
    nrows == 0 && return
    kernel = @cuda launch=false kernel_fn(alpha_out, srcs..., nrows)
    config = launch_configuration(kernel.fun)
    threads = (config.threads ÷ 32) * 32
    reduce_blocks = min(cld(nrows, threads), max(1, cld(config.blocks, bs)))
    kernel(alpha_out, srcs..., nrows; threads, blocks = reduce_blocks * bs)
end

function MadIPM._ftb_primal_lb!(alpha_out::AnyCuMatrix, dx::AnyCuMatrix, x::AnyCuMatrix, xb::AnyCuMatrix, tau::AnyCuMatrix)
    _launch_ftb_kernel!(_ftb_primal_lb_kernel!, alpha_out, size(dx, 1), dx, x, xb, tau)
end
function MadIPM._ftb_primal_ub!(alpha_out::AnyCuMatrix, dx::AnyCuMatrix, x::AnyCuMatrix, xb::AnyCuMatrix, tau::AnyCuMatrix)
    _launch_ftb_kernel!(_ftb_primal_ub_kernel!, alpha_out, size(dx, 1), dx, x, xb, tau)
end
function MadIPM._ftb_dual_lb!(alpha_out::AnyCuMatrix, dz::AnyCuMatrix, z::AnyCuMatrix, tau::AnyCuMatrix)
    _launch_ftb_kernel!(_ftb_dual_lb_kernel!, alpha_out, size(dz, 1), dz, z, tau)
end
function MadIPM._ftb_dual_ub!(alpha_out::AnyCuMatrix, dz::AnyCuMatrix, z::AnyCuMatrix, tau::AnyCuMatrix)
    _launch_ftb_kernel!(_ftb_dual_ub_kernel!, alpha_out, size(dz, 1), dz, z, tau)
end

_affine_compl_lb_kernel!(out, x, xl, z, dx, dz, αp, αd, nrows) = begin
    bs = size(out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(out)
    @inbounds if j <= bs
        s = zero(T); ap = αp[1, j]; ad = αd[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            s += (x[i,j] + ap * dx[i,j] - xl[i,j]) * (z[i,j] + ad * dz[i,j])
            i += blockDim().x * gridDim_reduce
        end
        s = CUDA.reduce_block(+, s, zero(T), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(+, out, j, s)
    end
    return
end

_affine_compl_ub_kernel!(out, xu, x, z, dx, dz, αp, αd, nrows) = begin
    bs = size(out, 2)
    blockIdx_reduce, j = fldmod1(blockIdx().x, bs)
    gridDim_reduce = gridDim().x ÷ bs
    T = eltype(out)
    @inbounds if j <= bs
        s = zero(T); ap = αp[1, j]; ad = αd[1, j]
        i = threadIdx().x + (blockIdx_reduce - 1) * blockDim().x
        while i <= nrows
            s += (xu[i,j] - (x[i,j] + ap * dx[i,j])) * (z[i,j] + ad * dz[i,j])
            i += blockDim().x * gridDim_reduce
        end
        s = CUDA.reduce_block(+, s, zero(T), Val(true))
        threadIdx().x == 1 && _atomic_colreduce!(+, out, j, s)
    end
    return
end

function _launch_reduce_kernel!(kernel_fn, out, nrows, srcs...)
    T = eltype(out)
    bs = size(out, 2)
    fill!(out, zero(T))
    nrows == 0 && return
    kernel = @cuda launch=false kernel_fn(out, srcs..., nrows)
    config = launch_configuration(kernel.fun)
    threads = (config.threads ÷ 32) * 32
    reduce_blocks = min(cld(nrows, threads), max(1, cld(config.blocks, bs)))
    kernel(out, srcs..., nrows; threads, blocks = reduce_blocks * bs)
end

function MadIPM._affine_compl_lb!(out::AnyCuMatrix, x::AnyCuMatrix, xl::AnyCuMatrix, z::AnyCuMatrix,
                                   dx::AnyCuMatrix, dz::AnyCuMatrix, αp::AnyCuMatrix, αd::AnyCuMatrix)
    _launch_reduce_kernel!(_affine_compl_lb_kernel!, out, size(x, 1), x, xl, z, dx, dz, αp, αd)
end

function MadIPM._affine_compl_ub!(out::AnyCuMatrix, xu::AnyCuMatrix, x::AnyCuMatrix, z::AnyCuMatrix,
                                   dx::AnyCuMatrix, dz::AnyCuMatrix, αp::AnyCuMatrix, αd::AnyCuMatrix)
    _launch_reduce_kernel!(_affine_compl_ub_kernel!, out, size(x, 1), xu, x, z, dx, dz, αp, αd)
end

@inline function _warp_argmin(val::T, idx::Int32) where T
    offset = Int32(16)
    while offset > Int32(0)
        other_val = CUDA.shfl_down_sync(0xffffffff, val, offset)
        other_idx = CUDA.shfl_down_sync(0xffffffff, idx, offset)
        if other_val < val
            val = other_val
            idx = other_idx
        end
        offset >>= Int32(1)
    end
    return val, idx
end

_mehrotra_step_kernel!(
    alpha_p, alpha_d, mu, gamma_f,
    dx_lr, x_lr, xl_r, nlb::Int32, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub::Int32, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off::Int32, dub_off::Int32,
) = begin
    j = Int32(blockIdx().x)
    lane = Int32(threadIdx().x - Int32(1))  # 0:31
    T = eltype(alpha_p)
    INF = T(Inf)

    # primal lb
    best_xl = INF; i_xl = Int32(0)
    @inbounds begin
        k = lane + Int32(1)
        while k <= nlb
            d = dx_lr[k, j]
            if d < zero(T)
                v = (xl_r[k, j] - x_lr[k, j]) / d
                if v < best_xl
                    best_xl = v; i_xl = k
                end
            end
            k += Int32(32)
        end
    end
    best_xl, i_xl = _warp_argmin(best_xl, i_xl)

    # primal ub
    best_xu = INF; i_xu = Int32(0)
    @inbounds begin
        k = lane + Int32(1)
        while k <= nub
            d = dx_ur[k, j]
            if d > zero(T)
                v = (xu_r[k, j] - x_ur[k, j]) / d
                if v < best_xu
                    best_xu = v; i_xu = k
                end
            end
            k += Int32(32)
        end
    end
    best_xu, i_xu = _warp_argmin(best_xu, i_xu)

    # dual lb
    best_zl = INF; i_zl = Int32(0)
    @inbounds begin
        k = lane + Int32(1)
        while k <= nlb
            d = dzlb[k, j]
            if d < zero(T)
                v = -zl_r[k, j] / d
                if v < best_zl
                    best_zl = v; i_zl = k
                end
            end
            k += Int32(32)
        end
    end
    best_zl, i_zl = _warp_argmin(best_zl, i_zl)

    # dual ub
    best_zu = INF; i_zu = Int32(0)
    @inbounds begin
        k = lane + Int32(1)
        while k <= nub
            d = dzub[k, j]
            if d < zero(T) && zu_r[k, j] + d < zero(T)
                v = -zu_r[k, j] / d
                if v < best_zu
                    best_zu = v; i_zu = k
                end
            end
            k += Int32(32)
        end
    end
    best_zu, i_zu = _warp_argmin(best_zu, i_zu)

    # lane 0: compute corrected steps
    @inbounds if lane == Int32(0)
        mu_j = mu[1, j]
        max_ap = alpha_p[1, j]
        max_ad = alpha_d[1, j]

        # corrected primal
        corrected_p = one(T)
        if max_ap < one(T)
            if best_xl <= best_xu && i_xl > Int32(0)
                zl_stepped = zl_r[i_xl, j] + max_ad * d_vals[dlb_off + i_xl, j]
                corrected_p = (x_lr[i_xl, j] - xl_r[i_xl, j] - mu_j / zl_stepped) / (-dx_lr[i_xl, j])
            elseif i_xu > Int32(0)
                zu_stepped = zu_r[i_xu, j] + max_ad * d_vals[dub_off + i_xu, j]
                corrected_p = (xu_r[i_xu, j] - x_ur[i_xu, j] - mu_j / zu_stepped) / dx_ur[i_xu, j]
            end
        end
        alpha_p[1, j] = max(corrected_p, gamma_f * max_ap)

        # corrected dual
        corrected_d = one(T)
        if max_ad < one(T)
            if best_zl <= best_zu && i_zl > Int32(0)
                x_gap = x_lr[i_zl, j] + max_ap * dx_lr[i_zl, j] - xl_r[i_zl, j]
                corrected_d = -(zl_r[i_zl, j] - mu_j / x_gap) / d_vals[dlb_off + i_zl, j]
            elseif i_zu > Int32(0)
                x_gap = xu_r[i_zu, j] - x_ur[i_zu, j] - max_ap * dx_ur[i_zu, j]
                corrected_d = -(zu_r[i_zu, j] - mu_j / x_gap) / d_vals[dub_off + i_zu, j]
            end
        end
        alpha_d[1, j] = max(corrected_d, gamma_f * max_ad)
    end
    return nothing
end

function MadIPM._mehrotra_step!(
    alpha_p::AnyCuMatrix, alpha_d, mu, gamma_f,
    dx_lr, x_lr, xl_r, nlb, dzlb, zl_r,
    dx_ur, x_ur, xu_r, nub, dzub, zu_r,
    d_vals, ind_lb, ind_ub, dlb_off, dub_off,
)
    CUDA.@cuda threads=32 blocks=bs _mehrotra_step_kernel!(
        alpha_p, alpha_d, mu, gamma_f,
        dx_lr, x_lr, xl_r, Int32(nlb), dzlb, zl_r,
        dx_ur, x_ur, xu_r, Int32(nub), dzub, zu_r,
        d_vals, ind_lb, ind_ub, Int32(dlb_off), Int32(dub_off),
    )
    return
end

@kernel function _reduce_rhs_lb_kernel!(values, @Const(ind_lb), lb_off, @Const(l_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[ind_lb[i], j] -= values[lb_off + i, j] / l_diag[i, j]
end

@kernel function _reduce_rhs_ub_kernel!(values, @Const(ind_ub), ub_off, @Const(u_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[ind_ub[i], j] -= values[ub_off + i, j] / u_diag[i, j]
end

function MadIPM._reduce_rhs_batch!(values::CuMatrix, ind_lb, lb_off, l_diag,
                                                      ind_ub, ub_off, u_diag)
    bs = size(values, 2); backend = CUDABackend()
    nlb = length(ind_lb)
    nlb > 0 && _reduce_rhs_lb_kernel!(backend)(values, ind_lb, lb_off, l_diag; ndrange=(nlb, bs))
    nub = length(ind_ub)
    nub > 0 && _reduce_rhs_ub_kernel!(backend)(values, ind_ub, ub_off, u_diag; ndrange=(nub, bs))
    return
end

@kernel function _finish_aug_solve_lb_kernel!(values, @Const(ind_lb), lb_off, @Const(l_lower), @Const(l_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[lb_off + i, j] = (-values[lb_off + i, j] + l_lower[i, j] * values[ind_lb[i], j]) / l_diag[i, j]
end

@kernel function _finish_aug_solve_ub_kernel!(values, @Const(ind_ub), ub_off, @Const(u_lower), @Const(u_diag))
    i, j = @index(Global, NTuple)
    @inbounds values[ub_off + i, j] = (values[ub_off + i, j] - u_lower[i, j] * values[ind_ub[i], j]) / u_diag[i, j]
end

function MadIPM._finish_aug_solve_batch!(values::CuMatrix, ind_lb, lb_off, l_lower, l_diag,
                                                            ind_ub, ub_off, u_lower, u_diag)
    bs = size(values, 2); backend = CUDABackend()
    nlb = length(ind_lb)
    nlb > 0 && _finish_aug_solve_lb_kernel!(backend)(values, ind_lb, lb_off, l_lower, l_diag; ndrange=(nlb, bs))
    nub = length(ind_ub)
    nub > 0 && _finish_aug_solve_ub_kernel!(backend)(values, ind_ub, ub_off, u_lower, u_diag; ndrange=(nub, bs))
    return
end
