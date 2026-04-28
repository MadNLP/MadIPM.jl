abstract type AbstractBatchMPCSolver{T, MT, VT} end

function _madnlp_unsafe_column_wrap(mat::MT, n, shift, ::Type{VT}) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return unsafe_wrap(VT, pointer(mat, shift), n)
end

function _csc_with_nzval(A::SparseArrays.SparseMatrixCSC, nzval, n)
    return SparseArrays.SparseMatrixCSC(n, n, SparseArrays.getcolptr(A), SparseArrays.rowvals(A), nzval)
end

function zero_inactive_step!(batch_solver::AbstractBatchMPCSolver{T}) where T
    ws = batch_solver.workspace
    ws.alpha_p .*= ws.active_mask
    ws.alpha_d .*= ws.active_mask
end

function _build_batch_op(nzVals, nz_map, val_map, coo_I, nrows)
    rowptr, colidx = BatchQuadraticModels._coo_to_csr(Vector{Int}(coo_I), nrows)
    return BatchQuadraticModels._build_storage_op(
        nzVals,
        rowptr,
        Vector{Int}(nz_map),
        Vector{Int}(val_map),
        colidx,
    )
end

function _build_jt_op(
    aug_I, aug_J, jac_range, n_tot,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    n_jac = length(jac_range)
    coo_I = similar(aug_I, n_jac)
    coo_I .= aug_J[jac_range]
    nz_map = similar(aug_csc_map, n_jac)
    nz_map .= jac_range
    con_map = similar(aug_csc_map, n_jac)
    con_map .= aug_I[jac_range] .- Int32(n_tot)
    return _build_batch_op(nzVals, nz_map, con_map, coo_I, n_tot)
end

function _build_j_op(
    aug_I, aug_J, jac_range, n_tot, m,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    n_jac = length(jac_range)
    coo_I = similar(aug_I, n_jac)
    coo_I .= aug_I[jac_range] .- Int32(n_tot)
    nz_map = similar(aug_csc_map, n_jac)
    nz_map .= jac_range
    var_map = similar(aug_csc_map, n_jac)
    var_map .= aug_J[jac_range]
    return _build_batch_op(nzVals, nz_map, var_map, coo_I, m)
end

function _build_hess_op(
    aug_I, aug_J, n_tot, n_hess,
    nzVals::AbstractMatrix{T}, aug_csc_map,
) where T
    if n_hess == 0
        nz_map = similar(aug_csc_map, 0)
        var_map = similar(aug_csc_map, 0)
        return _build_batch_op(nzVals, nz_map, var_map, similar(aug_I, 0), n_tot)
    end

    hess_range = n_tot+1:n_tot+n_hess
    hess_I = aug_I[hess_range]
    hess_J = aug_J[hess_range]

    offdiag_idx = findall(hess_I .!= hess_J)
    n_hess_sym = n_hess + length(offdiag_idx)

    coo_rows = similar(aug_I, n_hess_sym)
    coo_rows[1:n_hess] .= hess_I
    coo_rows[n_hess+1:end] .= hess_J[offdiag_idx]

    nz_map = similar(aug_csc_map, n_hess_sym)
    nz_map[1:n_hess] .= hess_range
    nz_map[n_hess+1:end] .= n_tot .+ offdiag_idx

    var_map = similar(aug_csc_map, n_hess_sym)
    var_map[1:n_hess] .= hess_J
    var_map[n_hess+1:end] .= hess_I[offdiag_idx]

    return _build_batch_op(nzVals, nz_map, var_map, coo_rows, n_tot)
end

struct BatchVector{T, MT<:AbstractMatrix{T}}
    values::MT
end

MadNLP.full(bv::BatchVector) = bv.values

function BatchVector(
    ::Type{MT}, ::Type{VT},
    len::Int, batch_size::Int,
) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    values = MT(undef, len, batch_size)
    fill!(values, zero(T))
    return BatchVector{T, MT}(values)
end

mutable struct BatchExecutionStats{T, VT<:AbstractVector{T}, MT<:AbstractMatrix{T}}
    status::Vector{MadNLP.Status}  # (bs,)
    solution::MT                   # (nvar_nlp, bs)
    objective::VT                  # (bs,)
    constraints::MT                # (ncon, bs)
    dual_feas::VT                  # (bs,)
    primal_feas::VT                # (bs,)
    multipliers::MT                # (ncon, bs)
    multipliers_L::MT              # (nvar_nlp, bs)
    multipliers_U::MT              # (nvar_nlp, bs)
    iter::Vector{Int}              # (bs,)
    total_time::Vector{Float64}    # (bs,)
end

function BatchExecutionStats(::Type{MT}, ::Type{VT}, nvar_nlp::Int, ncon::Int, batch_size::Int) where {T, MT<:AbstractMatrix{T}, VT<:AbstractVector{T}}
    return BatchExecutionStats{T, VT, MT}(
        fill(MadNLP.INITIAL, batch_size),
        MT(undef, nvar_nlp, batch_size),
        VT(undef, batch_size),
        MT(undef, ncon, batch_size),
        VT(undef, batch_size),
        VT(undef, batch_size),
        MT(undef, ncon, batch_size),
        MT(undef, nvar_nlp, batch_size),
        MT(undef, nvar_nlp, batch_size),
        zeros(Int, batch_size),
        zeros(Float64, batch_size),
    )
end

function Base.getindex(stats::BatchExecutionStats, i::Int)
    return (
        status = stats.status[i],
        solution = view(stats.solution, :, i),
        objective = stats.objective[i],
        constraints = view(stats.constraints, :, i),
        dual_feas = stats.dual_feas[i],
        primal_feas = stats.primal_feas[i],
        multipliers = view(stats.multipliers, :, i),
        multipliers_L = view(stats.multipliers_L, :, i),
        multipliers_U = view(stats.multipliers_U, :, i),
        iter = stats.iter[i],
        total_time = stats.total_time[i],
    )
end

struct BatchCounters
    k::Vector{Int}              # per-instance iteration count
    start_time::Base.RefValue{Float64}
    total_time::Vector{Float64} # per-instance total solve time
    linear_solver_time::Base.RefValue{Float64}
    eval_function_time::Base.RefValue{Float64}
    obj_cnt::Base.RefValue{Int}
    obj_grad_cnt::Base.RefValue{Int}
    con_cnt::Base.RefValue{Int}
end
BatchCounters(batch_size::Int) = BatchCounters(zeros(Int, batch_size), Ref(0.0), zeros(Float64, batch_size), Ref(0.0), Ref(0.0), Ref(0), Ref(0), Ref(0))
