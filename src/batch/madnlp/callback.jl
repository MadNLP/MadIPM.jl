struct UniformBatchCallback{
    T,
    VT<:AbstractVector{T},
    MT<:AbstractMatrix{T},
    VI<:AbstractVector{Int},
    BM<:NLPModels.AbstractBatchNLPModel,
    FH<:MadNLP.AbstractFixedVariableTreatment,
    EH<:MadNLP.AbstractEqualityTreatment,
} <: MadNLP.AbstractCallback{T, VT, FH}
    nlp::BM
    batch_size::Int

    nvar::Int       # per-instance nvar (after fixed variable removal)
    ncon::Int
    nnzj::Int       # per-instance nnzj (after fixed variable removal)
    nnzh::Int       # per-instance nnzh (after fixed variable removal)

    # Per-instance
    con_buffer::MT    # ncon × batch_size
    jac_buffer::MT    # nnzj × batch_size
    grad_buffer::MT   # nvar_nlp × batch_size
    hess_buffer::MT   # nnzh × batch_size

    # Shared
    jac_I::VI
    jac_J::VI
    hess_I::VI
    hess_J::VI

    # Per-instance
    obj_scale::MT     # 1 × batch_size per-instance objective scale
    obj_sign::MT      # 1 × batch_size ±1.0 per instance
    con_scale::MT     # ncon × batch_size
    jac_scale::MT     # nnzj × batch_size

    # Shared
    fixed_handler::FH
    equality_handler::EH
    ind_eq::VI
    ind_ineq::VI
    ind_fixed::VI
    ind_lb::VI
    ind_ub::VI
    ind_llb::VI
    ind_uub::VI
end

function MadNLP.create_sparse_fixed_handler(
    ::Type{MadNLP.MakeParameter},
    bnlp::NLPModels.AbstractBatchNLPModel{T},
    jac_I, jac_J, hess_I, hess_J, hess_buffer,
) where T
    n = NLPModels.get_nvar(bnlp)
    lvar = view(bnlp.meta.lvar, :, 1)
    uvar = view(bnlp.meta.uvar, :, 1)
    nnzj = NLPModels.get_nnzj(bnlp)
    nnzh = NLPModels.get_nnzh(bnlp)

    bs = NLPModels.get_nbatch(bnlp)
    x_full = similar(lvar, n * bs)
    g_full = similar(lvar, n * bs)

    isfixed = (lvar .== uvar)
    isfree = (lvar .< uvar)

    fixed = findall(isfixed)
    nfixed = length(fixed)

    if nfixed == 0
        return MadNLP.NoFixedVariables(), n, nnzj, nnzh
    end

    free = findall(isfree)
    nx = length(free)
    map_full_to_free = similar(jac_I, n); fill!(map_full_to_free, -1)
    map_full_to_free[free] .= 1:nx

    ind_jac_free = findall(@view(isfree[jac_J]))
    ind_hess_free = findall(@view(isfree[hess_I]) .&& @view(isfree[hess_J]))

    nnzh = length(ind_hess_free)
    Hi, Hj = similar(hess_I, nnzh), similar(hess_J, nnzh)
    copyto!(Hi, map_full_to_free[hess_I[ind_hess_free]])
    copyto!(Hj, map_full_to_free[hess_J[ind_hess_free]])
    resize!(hess_I, nnzh)
    resize!(hess_J, nnzh)
    copyto!(hess_I, Hi)
    copyto!(hess_J, Hj)

    nnzj = length(ind_jac_free)
    Ji, Jj = similar(jac_I, nnzj), similar(jac_J, nnzj)
    copyto!(Ji, jac_I[ind_jac_free])
    copyto!(Jj, map_full_to_free[jac_J[ind_jac_free]])
    resize!(jac_I, nnzj)
    resize!(jac_J, nnzj)
    copyto!(jac_I, Ji)
    copyto!(jac_J, Jj)

    fixed_handler = MadNLP.MakeParameter(
        free,
        fixed,
        ind_jac_free,
        ind_hess_free,
        Ref(NaN),
        x_full,
        g_full,
    )

    return fixed_handler, nx, nnzj, nnzh
end

function MadNLP.create_callback(
    ::Type{UniformBatchCallback{T,VT,MT,VI}},
    bnlp::NLPModels.AbstractBatchNLPModel{T};
    fixed_variable_treatment=MadNLP.MakeParameter,
    equality_treatment=MadNLP.EnforceEquality,
    check_batch_structure::Bool=true,
) where {T,VT,MT,VI}
    bmeta = bnlp.meta
    batch_size = bmeta.nbatch
    
    n = bmeta.nvar
    m = bmeta.ncon
    nnzj = bmeta.nnzj
    nnzh = bmeta.nnzh

    x0 = NLPModels.get_x0(bnlp)

    jac_I = similar(x0, Int, nnzj)
    jac_J = similar(x0, Int, nnzj)
    hess_I = similar(x0, Int, nnzh)
    hess_J = similar(x0, Int, nnzh)

    obj_scale = fill!(similar(x0, 1, batch_size), one(T))
    con_scale = fill!(similar(x0, m, batch_size), one(T))
    con_buffer = fill!(similar(x0, m, batch_size), zero(T))
    jac_buffer = fill!(similar(x0, nnzj, batch_size), zero(T))
    hess_buffer = fill!(similar(x0, nnzh, batch_size), zero(T))

    if nnzj > 0
        NLPModels.jac_structure!(bnlp, jac_I, jac_J)
    end
    if nnzh > 0
        NLPModels.hess_structure!(bnlp, hess_I, hess_J)
    end

    if check_batch_structure
        row_sums = vcat(
            sum(bmeta.lvar .== bmeta.uvar; dims=2),
            sum(isfinite.(bmeta.lvar); dims=2),
            sum(isfinite.(bmeta.uvar); dims=2),
            sum(bmeta.lcon .== bmeta.ucon; dims=2),
            sum(isfinite.(bmeta.lcon); dims=2),
            sum(isfinite.(bmeta.ucon); dims=2),
        )
        @assert all((row_sums .== 0) .| (row_sums .== batch_size)) "Batch fixed/bound/equality structure must match across instances"
    end

    lvar = view(bmeta.lvar, :, 1)
    uvar = view(bmeta.uvar, :, 1)
    lcon = view(bmeta.lcon, :, 1)
    ucon = view(bmeta.ucon, :, 1)

    fixed_handler, nvar, nnzj, nnzh = MadNLP.create_sparse_fixed_handler(
        fixed_variable_treatment,
        bnlp,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        nothing,  # hess_buffer not used
    )
    equality_handler = equality_treatment()

    # Allocate with reduced sizes (after fixed var removal)
    jac_scale = similar(x0, nnzj, batch_size); fill!(jac_scale, one(T))
    grad_buffer = fill!(similar(x0, nvar, batch_size), zero(T))

    # Get fixed variables
    ind_fixed = findall(lvar .== uvar)
    if length(ind_fixed) > 0 && fixed_variable_treatment == MadNLP.MakeParameter
        ind_free = findall(lvar .< uvar)
        # Remove fixed variables from problem's formulation
        lvar = lvar[ind_free]
        uvar = uvar[ind_free]
    end

    indexes = MadNLP._parse_indexes(lvar, uvar, lcon, ucon, equality_treatment)

    return UniformBatchCallback{T, VT, MT, VI, typeof(bnlp), typeof(fixed_handler), typeof(equality_handler)}(
        bnlp,
        batch_size,
        nvar,
        m,
        nnzj,
        nnzh,
        con_buffer,
        jac_buffer,
        grad_buffer,
        hess_buffer,
        jac_I,
        jac_J,
        hess_I,
        hess_J,
        obj_scale,
        fill!(similar(x0, 1, batch_size), bmeta.minimize ? one(T) : -one(T)),  # obj_sign
        con_scale,
        jac_scale,
        fixed_handler,
        equality_handler,
        indexes.ind_eq,
        indexes.ind_ineq,
        ind_fixed,
        indexes.ind_lb,
        indexes.ind_ub,
        indexes.ind_llb,
        indexes.ind_uub,
    )
end



function MadNLP.n_variables(bcb::UniformBatchCallback{T, VT, MT, VI, BM, FH}) where {T, VT, MT, VI, BM, FH<:MadNLP.MakeParameter}
    return length(bcb.fixed_handler.free)
end
MadNLP.n_variables(bcb::UniformBatchCallback) = bcb.nvar

function MadNLP._jac_sparsity_wrapper!(bcb::UniformBatchCallback, I::AbstractVector, J::AbstractVector)
    copyto!(I, bcb.jac_I)
    copyto!(J, bcb.jac_J)
    return
end

function MadNLP._hess_sparsity_wrapper!(bcb::UniformBatchCallback, I::AbstractVector, J::AbstractVector)
    copyto!(I, bcb.hess_I)
    copyto!(J, bcb.hess_J)
    return
end

function MadNLP.build_hessian_structure(bcb::UniformBatchCallback, ::Type{<:MadNLP.ExactHessian})
    hess_I = MadNLP.create_array(bcb, Int32, bcb.nnzh)
    hess_J = MadNLP.create_array(bcb, Int32, bcb.nnzh)
    MadNLP._hess_sparsity_wrapper!(bcb, hess_I, hess_J)
    return hess_I, hess_J
end
