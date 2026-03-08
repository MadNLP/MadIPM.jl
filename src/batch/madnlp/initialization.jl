function MadNLP.set_initial_bounds!(xl::AbstractMatrix{T}, xu::AbstractMatrix{T}, tol) where T
    if tol > zero(T)
        xl .= xl .- max.(one(T), abs.(xl)) .* tol
        xu .= xu .+ max.(one(T), abs.(xu)) .* tol
    end
end

function MadNLP.initialize!(
    bcb::UniformBatchCallback{T},
    x,
    xl,
    xu,
    y,
    rhs,
    ind_ineq;
    tol=1e-8,
    bound_push=1e-2,
    bound_fac=1e-2,
) where T
    x0   = MadNLP.variable(x)
    lvar = MadNLP.variable(xl)
    uvar = MadNLP.variable(xu)

    x0   .= MadNLP.get_x0(bcb)
    lvar .= MadNLP.get_lvar(bcb)
    uvar .= MadNLP.get_uvar(bcb)
    y .= MadNLP.get_y0(bcb)
    lcon = copy(MadNLP.get_lcon(bcb))
    ucon = copy(MadNLP.get_ucon(bcb))

    MadNLP._treat_equality_initialize!(bcb.equality_handler, lcon, ucon, tol)
    MadNLP._treat_fixed_variable_initialize!(bcb, x0, lvar, uvar)  # this is no-op, recall bcb <: MadNLP.AbstractCallback. will need to be adjust for batchdensecallback

    MadNLP.set_initial_bounds!(lvar, uvar, tol)
    x0 .= MadNLP._initialize_variables!.(x0, lvar, uvar, bound_push, bound_fac)

    x_full = MadNLP._update_x!(bcb, x0)
    MadNLP._eval_cons_wrapper!(bcb, x_full, bcb.con_buffer)

    MadNLP.slack(xl) .= view(lcon, ind_ineq, :)
    MadNLP.slack(xu) .= view(ucon, ind_ineq, :)
    rhs .= (lcon .== ucon) .* lcon
    copyto!(MadNLP.slack(x), view(bcb.con_buffer, ind_ineq, :))

    MadNLP.set_initial_bounds!(MadNLP.slack(xl), MadNLP.slack(xu), tol)
    MadNLP.slack(x) .= MadNLP._initialize_variables!.(MadNLP.slack(x), MadNLP.slack(xl), MadNLP.slack(xu), bound_push, bound_fac)

    return
end

function MadNLP.set_con_scale_sparse!(
    con_scale::MT,
    jac_I,
    jac_buffer,
    max_gradient,
) where {T,MT<:AbstractMatrix{T}}
    fill!(con_scale, one(T))
    MadNLP._set_con_scale_sparse!(con_scale, jac_I, jac_buffer)
    con_scale .= min.(one(T), max_gradient ./ con_scale)
    return con_scale
end

function MadNLP._set_con_scale_sparse!(con_scale::MT, jac_I, jac_buffer) where {T,MT<:AbstractMatrix{T}}
    nnzj = length(jac_I)
    bs = size(jac_buffer, 2)
    @inbounds for k in 1:nnzj
        row = jac_I[k]
        for j in 1:bs
            con_scale[row, j] = max(con_scale[row, j], abs(jac_buffer[k, j]))
        end
    end
    return con_scale
end

function MadNLP.set_jac_scale_sparse!(jac_scale::MT, con_scale, jac_I) where {T,MT<:AbstractMatrix{T}}
    return copyto!(jac_scale, @view(con_scale[jac_I, :]))
end

function MadNLP.set_obj_scale!(obj_scale, F::MT, max_gradient) where {T,MT<:AbstractMatrix{T}}
    return obj_scale .= min.(one(T), max_gradient ./ maximum(abs, F; dims=1))
end

function MadNLP.set_scaling!(
    cb::UniformBatchCallback,
    x, xl, xu, y, rhs, ind_ineq, nlp_scaling_max_gradient,
)
    x0 = MadNLP.variable(x)
    x_full = MadNLP._update_x!(cb, x0)

    jac_free = MadNLP._eval_jac_wrapper!(cb, x_full, cb.jac_buffer)
    MadNLP.set_con_scale_sparse!(cb.con_scale, cb.jac_I, jac_free, nlp_scaling_max_gradient)
    MadNLP.set_jac_scale_sparse!(cb.jac_scale, cb.con_scale, cb.jac_I)

    MadNLP._eval_grad_f_wrapper!(cb, x_full, cb.grad_buffer)
    MadNLP.set_obj_scale!(cb.obj_scale, cb.grad_buffer, nlp_scaling_max_gradient)

    con_scale_slk = @view(cb.con_scale[ind_ineq, :])
    y ./= cb.con_scale
    rhs .*= cb.con_scale
    MadNLP.slack(x)  .*= con_scale_slk
    MadNLP.slack(xl) .*= con_scale_slk
    MadNLP.slack(xu) .*= con_scale_slk
    return
end
