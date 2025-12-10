abstract type AbstractConicProblem end

struct LinearProgram <: AbstractConicProblem end
struct QuadraticProgram <: AbstractConicProblem end

#=
    Barrier update
=#

abstract type AbstractBarrierUpdate end
struct Mehrotra <: AbstractBarrierUpdate end

#=
    Step rule for next iterate
=#

abstract type AbstractStepRule end

@kwdef struct ConservativeStep{T} <: AbstractStepRule
    tau::T = T(0.995)
end

@kwdef struct AdaptiveStep{T} <: AbstractStepRule
    tau_min::T = T(0.99)
end

@kwdef struct MehrotraAdaptiveStep{T} <: AbstractStepRule
    gamma_f::T = T(0.99)
end

#=
    Primal-dual regularization for KKT system
=#

abstract type AbstractRegularization end

struct NoRegularization <: AbstractRegularization end

mutable struct FixedRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
end

mutable struct AdaptiveRegularization{T} <: AbstractRegularization
    delta_p::T
    delta_d::T
    delta_min::T
end

#=
    Utils for linear solvers
=#

function is_factorized(::MadNLP.AbstractLinearSolver)
    return true # assume the system is factorized by default
end
function is_factorized(lin_solver::MadNLP.LDLSolver)
    return LDLFactorizations.factorized(lin_solver.inner)
end
function is_factorized(lin_solver::MadNLP.CHOLMODSolver)
    return issuccess(lin_solver.inner)
end


#=
    Options
=#

@kwdef mutable struct IPMOptions <: MadNLP.AbstractOptions
    tol::Float64
    kkt_system::Type
    linear_solver::Type
    # Output options
    output_file::String = ""
    print_level::MadNLP.LogLevels = MadNLP.INFO
    file_print_level::MadNLP.LogLevels = MadNLP.INFO
    rethrow_error::Bool = false
    # Termination options
    max_iter::Int = 3000
    max_wall_time::Float64 = 1e6
    divergence_tol::Float64 = 1e4
    kappa_d::Float64 = 1e-5
    fixed_variable_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxBound : MadNLP.MakeParameter
    equality_treatment::Type = kkt_system <: MadNLP.SparseCondensedKKTSystem ? MadNLP.RelaxEquality : MadNLP.EnforceEquality
    # initialization options
    scaling::Bool = true
    bound_push::Float64 = 1e-2
    bound_fac::Float64 = 1e-2
    bound_relax_factor::Float64 = 1e-12
    # Regularization
    regularization::AbstractRegularization = FixedRegularization(1e-10, 1e-10)
    # Step
    step_rule::AbstractStepRule = AdaptiveStep(0.99)
    # Barrier
    barrier_update::AbstractBarrierUpdate = Mehrotra()
    max_ncorr::Int = 0
    s_max::Float64 = 100.0
    mu_init::Float64 = 1e-1
    mu_min::Float64 = 1e-12
    mu_superlinear_decrease_power::Float64 = 1.5
    tau_min::Float64 = 0.99
    # Linear solve
    tol_linear_solve::Float64 = 1e-8
    check_residual::Bool = false
end

# smart option presets
function IPMOptions(
    nlp::NLPModels.AbstractNLPModel{T};
    kkt_system =  MadNLP.SparseKKTSystem,
    linear_solver =  MadNLP.default_sparse_solver(nlp),
    tol = 1e-8,
) where T
    return IPMOptions(
        tol = tol,
        kkt_system = kkt_system,
        linear_solver = linear_solver,
    )
end

function load_options(nlp; options...)
    primary_opt, options = MadNLP._get_primary_options(options)

    # Initiate interior-point options
    opt_ipm = IPMOptions(nlp; primary_opt...)
    linear_solver_options = MadNLP.set_options!(opt_ipm, options)
    # Initiate linear-solver options
    opt_linear_solver = MadNLP.default_options(opt_ipm.linear_solver)
    remaining_options = MadNLP.set_options!(opt_linear_solver, linear_solver_options)

    # Initiate logger
    logger = MadNLP.MadNLPLogger(
        print_level=opt_ipm.print_level,
        file_print_level=opt_ipm.file_print_level,
        file = opt_ipm.output_file == "" ? nothing : open(opt_ipm.output_file,"w+"),
    )
    MadNLP.@trace(logger,"Logger is initialized.")

    # Print remaning options (unsupported)
    if !isempty(remaining_options)
        MadNLP.print_ignored_options(logger, remaining_options)
    end
    return (
        interior_point=opt_ipm,
        linear_solver=opt_linear_solver,
        logger=logger,
    )
end

function update_solution!(stats::MadNLP.MadNLPExecutionStats{T}, solver) where T
    MadNLP.update!(stats,solver)
    return
end

function coo_to_csr(
    n_rows,
    n_cols,
    Ai::AbstractVector{Ti},
    Aj::AbstractVector{Ti},
    Ax::AbstractVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    nnz = length(Ai)
    Bp = zeros(Ti, n_rows+1)
    Bj = zeros(Ti, nnz)
    Bx = zeros(Tv, nnz)

    nnz = length(Ai)
    @inbounds for n in 1:nnz
        Bp[Ai[n]] += 1
    end

    # cumsum the nnz per row to get Bp
    cumsum = 1
    @inbounds for i in 1:n_rows
        tmp = Bp[i]
        Bp[i] = cumsum
        cumsum += tmp
    end
    Bp[n_rows+1] = nnz + 1

    @inbounds for n in 1:nnz
        i = Ai[n]
        dest = Bp[i]
        Bj[dest] = Aj[n]
        Bx[dest] = Ax[n]
        Bp[i] += 1
    end

    last = 1
    @inbounds for i in 1:n_rows+1
        tmp = Bp[i]
        Bp[i] = last
        last = tmp
    end

    return (Bp, Bj, Bx)
end

function coo_to_csr(A::MadNLP.SparseMatrixCOO)
    return coo_to_csr(
        A.m, A.n, A.I, A.J, A.V,
    )
end

function build_normal_system(
    n_rows,
    n_cols,
    Jtp::AbstractVector{Ti},
    Jtj::AbstractVector{Ti},
) where {Ti}
    Cp = zeros(Ti, n_rows + 1)
    xb = zeros(UInt8, n_cols)

    # Count nonzeros per rows
    nnz = 0
    @inbounds for i in 1:n_rows
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            xb[j] = UInt8(1)
        end
        # JᵀJ is symmetric, store only lower triangular part
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                k = Jtj[c]
                if xb[k] == 1
                    nnz += 1
                    Cp[i] += 1
                    break
                end
            end
        end
        # Reset to 0
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(0)
        end
    end
    # cumsum the nnz per row to get Bp
    cumsum = 1
    @inbounds for i in 1:n_rows
        tmp = Cp[i]
        Cp[i] = cumsum
        cumsum += tmp
    end
    Cp[n_rows+1] = nnz + 1

    Cj = zeros(Ti, nnz)
    cnt = 0
    @inbounds for i in 1:n_rows
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            xb[j] = UInt8(1)
        end
        # JᵀJ is symmetric, store only lower triangular part
        for j in i:n_rows
            for c in Jtp[j]:Jtp[j+1]-1
                k = Jtj[c]
                if xb[k] == 1
                    cnt += 1
                    Cj[cnt] = j
                    break
                end
            end
        end
        for c in Jtp[i]:Jtp[i+1]-1
            xb[Jtj[c]] = UInt8(0)
        end
    end

    return (Cp, Cj)
end

function assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::AbstractVector{Ti},
    Jtj::AbstractVector{Ti},
    Jtx::AbstractVector{Tv},
    Cp::AbstractVector{Ti},
    Cj::AbstractVector{Ti},
    Cx::AbstractVector{Tv},
    Dx::AbstractVector{Tv},
) where {Ti, Tv}
    buffer = zeros(Tv, n_cols)
    @inbounds for i in 1:n_rows
        # Read row i
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            buffer[j] = Jtx[c] * Dx[j]
        end
        for c in Cp[i]:Cp[i+1]-1
            j = Cj[c]
            Cx[c] = Tv(0)
            for d in Jtp[j]:Jtp[j+1]-1
                k = Jtj[d]
                Cx[c] += buffer[k] * Jtx[d]
            end
        end
        # Reset buffer
        for c in Jtp[i]:Jtp[i+1]-1
            j = Jtj[c]
            buffer[j] = Tv(0)
        end
    end
end

sparse_csc_format(::Type{<:Array}) = SparseArrays.SparseMatrixCSC
_colptr(A::SparseArrays.SparseArrays.SparseMatrixCSC) = A.colptr
_rowval(A::SparseArrays.SparseMatrixCSC) = A.rowval
_nzval(A::SparseArrays.SparseMatrixCSC) = A.nzval

#=
    QuadraticModels wrapper
=#

"""
    presolved_qp, flag = presolve_qp(qp::QuadraticModel)

Run basic presolve routines implemented in `QuadraticModels.presolve` and return
a new QuadraticModel if flag is `true`.

If `flag` is `false`, the initial `qp` is returned.
"""
function presolve_qp(qp::QuadraticModels.QuadraticModel)
    # Use routine implemented in QuadraticModels
    res = QuadraticModels.presolve(qp)
    qp_presolved = res.solver_specific[:presolvedQM]
    if qp_presolved != nothing
        new_qp = QuadraticModels.QuadraticModel(
            qp_presolved.meta,
            qp_presolved.counters,
            qp_presolved.data,
        )
        resize!(new_qp.data.v, NLPModels.get_nvar(new_qp))
        return new_qp, true
    else
        # unbounded, infeasible or nvarps == 0
        return qp, false
    end
end

"""
    standard_form_qp(qp::QuadraticModel)

Reformulate a QP into standard form.

The function takes as input a generic QP of the form:
```
min_{x}  c'x
  s.t.   xl <= x <= xu
         bl <= Ax <= bu

```
and reformulates it by:
- introducing slack variables `s` such that `s = Ax` for inequality constraints,
- rewriting upper bounds (on `x` and `s`) as equality constraints using nonnegative slack variables `w = (wx, ws)`.

The resulting problem is equivalent to:
```
min_{x,s,w}  c'x
  s.t.       s = Ax
             x + wx = xu  (for upper-bounded x)
             s + ws = bu  (for upper-bounded Ax)
             xl <= x
             bl <= s
             0 <= w
```
Equality constraints are preserved as-is.
"""
function standard_form_qp(qp::QuadraticModels.QuadraticModel)
    n = NLPModels.get_nvar(qp)
    m = NLPModels.get_ncon(qp)

    lvar, uvar = NLPModels.get_lvar(qp), NLPModels.get_uvar(qp)
    lcon, ucon = NLPModels.get_lcon(qp), NLPModels.get_ucon(qp)

    # Count ineq. constraints
    ind_ineq = Int[]
    for i in 1:m
        (lcon[i] < ucon[i]) && push!(ind_ineq, i)
    end
    ns = length(ind_ineq)

    # Count upper-bounds in x
    ind_rng = Int[]
    ind_only_ub = Int[]
    ind_fixed = Int[]
    xu = Float64[]
    for i in 1:n
        if (lvar[i] == uvar[i])
            # Fixed variable
            push!(ind_fixed, i)
        elseif (-Inf < lvar[i] < uvar[i] < Inf)
            # Range bounds
            push!(ind_rng, i)
            push!(xu, uvar[i])
        elseif (uvar[i] < Inf)
            # Only upper bounds
            push!(ind_only_ub, i)
        end
    end

    # Count upper-bounds in future slack s
    for (k, i) in enumerate(ind_ineq)
        if (-Inf < lcon[i] < ucon[i] < Inf)
            # Range bounds
            push!(ind_rng, k + n)
            push!(xu, ucon[i])
        elseif (ucon[i] < Inf)
            # Only upper bounds
            push!(ind_only_ub, k + n)
        end
    end

    # Build problem
    nw = length(ind_rng)
    nvar = n + ns + nw
    ncon = m + nw

    # Build A and H
    Hs = SparseMatrixCOO(nvar, nvar, qp.data.H.rows, qp.data.H.cols, qp.data.H.vals)

    Ai, Aj, Ax = SparseArrays.findnz(qp.data.A)
    Bi, Bj, Bx = similar(Ai, ns+2*nw), similar(Aj, ns+2nw), similar(Ax, ns+2*nw)

    cnt = 1
    # Slack contribution Ax - s = 0
    for (k, i) in enumerate(ind_ineq)
        Bi[cnt] = i
        Bj[cnt] = n + k
        Bx[cnt] = -1.0
        cnt += 1
    end
    # Range reformulation x + w = xu
    for (k, i) in enumerate(ind_rng)
        Bi[cnt] = m + k
        Bj[cnt] = i
        Bx[cnt] = 1.0
        cnt += 1
        Bi[cnt] = m + k
        Bj[cnt] = k + n + ns
        Bx[cnt] = 1.0
        cnt += 1
    end

    As = SparseMatrixCOO(ncon, nvar, [Ai; Bi], [Aj; Bj], [Ax; Bx])

    # Build constraints' lower and upper bounds
    lcon_ = zeros(ncon)
    ucon_ = zeros(ncon)

    for i in 1:m
        if lcon[i] < ucon[i]
            # inequality constraints
            lcon_[i] = 0.0
            ucon_[i] = 0.0
        else
            # equality constraints
            lcon_[i] = lcon[i]
            ucon_[i] = ucon[i]
        end
    end
    for (k, i) in enumerate(ind_rng)
        lcon_[m + k] = xu[k]
        ucon_[m + k] = xu[k]
    end

    lvar_ = [lvar; lcon[ind_ineq]; zeros(nw)]
    uvar_ = [uvar; ucon[ind_ineq]; fill(Inf, nw)]
    # The upper bounds in range constraints have been moved in a separate constraint
    uvar_[ind_rng] .= Inf
    # Keep fixed variables in the formulation
    uvar_[ind_fixed] .= uvar[ind_fixed]

    data = QuadraticModels.QPData(
        qp.data.c0,
        [qp.data.c; zeros(ns + nw)],
        [qp.data.v; zeros(ns + nw)],
        Hs,
        As,
    )

    return QuadraticModels.QuadraticModel(
        NLPModels.NLPModelMeta(
            nvar;
            ncon=ncon,
            lvar=lvar_,
            uvar=uvar_,
            lcon=lcon_,
            ucon=ucon_,
            x0=[qp.meta.x0; zeros(ns + nw)],
            y0=[qp.meta.y0; zeros(nw)],
            nnzj=qp.meta.nnzj + ns + 2*nw,
            lin_nnzj=qp.meta.nnzj + ns + 2*nw,
            lin=[qp.meta.lin; (m+1:m+nw)],
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        NLPModels.Counters(),
        data,
    )
end

