# Adapted from NLPModelsJuMP

import QuadraticModels: SparseMatrixCOO

const MOI = MathOptInterface

const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction{Float64}
const VAF = MOI.VectorAffineFunction{Float64}
const SQF = MOI.ScalarQuadraticFunction{Float64}
const AF = Union{SAF, VAF}
const LinQuad = Union{VI, SAF, SQF}

const _SCALAR_SETS = Union{
  MOI.EqualTo{Float64},
  MOI.GreaterThan{Float64},
  MOI.LessThan{Float64},
  MOI.Interval{Float64},
}

function parse_variable(model)
    # Number of variables and bounds constraints
    vars = MOI.get(model, MOI.ListOfVariableIndices())
    nvar = length(vars)
    lvar = zeros(nvar)
    uvar = zeros(nvar)
    # Initial solution
    x0 = zeros(nvar)
    has_start = MOI.VariablePrimalStart() in MOI.get(model, MOI.ListOfVariableAttributesSet())

    index_map = MOI.Utilities.IndexMap()
    for (i, vi) in enumerate(vars)
        index_map[vi] = MOI.VariableIndex(i)
    end

    for (i, vi) in enumerate(vars)
        lvar[i], uvar[i] = MOI.Utilities.get_bounds(model, Float64, vi)
        if has_start
            val = MOI.get(model, MOI.VariablePrimalStart(), vi)
            if val !== nothing
                x0[i] = val
            end
        end
    end

    return index_map, nvar, lvar, uvar, x0
end

function parse_constraints(moimodel, index_map)
    _index(v::MOI.VariableIndex) = index_map[v].value
    # Variables associated to linear constraints
    nlin = 0
    linrows = Int[]
    lincols = Int[]
    linvals = Float64[]
    lin_lcon = Float64[]
    lin_ucon = Float64[]

    contypes = MOI.get(moimodel, MOI.ListOfConstraintTypesPresent())
    for (F, S) in contypes
        @assert F <: AF || F<: VI
        conindices = MOI.get(moimodel, MOI.ListOfConstraintIndices{F, S}())
        for cidx in conindices
            fun = MOI.get(moimodel, MOI.ConstraintFunction(), cidx)
            if F == VI
                index_map[cidx] = MOI.ConstraintIndex{F, S}(_index(fun))
                continue
            else
                index_map[cidx] = MOI.ConstraintIndex{F, S}(nlin)
            end
            set = MOI.get(moimodel, MOI.ConstraintSet(), cidx)
            if F <: SAF
                for term in fun.terms
                    push!(linrows, nlin + 1)
                    push!(lincols, _index(term.variable))
                    push!(linvals, term.coefficient)
                end
                # LB
                if typeof(set) in (MOI.Interval{Float64}, MOI.GreaterThan{Float64})
                    push!(lin_lcon, -fun.constant + set.lower)
                elseif typeof(set) == MOI.EqualTo{Float64}
                    push!(lin_lcon, -fun.constant + set.value)
                else
                    push!(lin_lcon, -Inf)
                end
                # UB
                if typeof(set) in (MOI.Interval{Float64}, MOI.LessThan{Float64})
                    push!(lin_ucon, -fun.constant + set.upper)
                elseif typeof(set) == MOI.EqualTo{Float64}
                    push!(lin_ucon, -fun.constant + set.value)
                else
                    push!(lin_ucon, Inf)
                end
                nlin += 1
            end
        end
    end
    return linrows, lincols, linvals, lin_lcon, lin_ucon
end

function parse_objective(moimodel, index_map, nvar)
    _index(v::MOI.VariableIndex) = index_map[v].value

    # Variables associated to linear and quadratic objective
    constant = 0.0
    vect = zeros(Float64, nvar)
    rows = Int[]
    cols = Int[]
    vals = Float64[]

    fobj = MOI.get(moimodel, MOI.ObjectiveFunction{LinQuad}())


    # Single Variable
    if typeof(fobj) == VI
        vect[_index(fobj)] = 1.0
    end
    # Linear objective
    if typeof(fobj) == SAF
        constant = fobj.constant
        for term in fobj.terms
            vect[_index(term.variable)] += term.coefficient
        end
    end
    # Quadratic objective
    if typeof(fobj) == SQF
        # Ensure that all coefficients are unique
        MOI.Utilities.canonicalize!(fobj)
        constant = fobj.constant
        for term in fobj.affine_terms
            vect[_index(term.variable)] += term.coefficient
        end
        for term in fobj.quadratic_terms
            i = _index(term.variable_1)
            j = _index(term.variable_2)
            if i ≥ j
                push!(rows, i)
                push!(cols, j)
            else
                push!(rows, j)
                push!(cols, i)
            end
            push!(vals, term.coefficient)
        end
    end
    return rows, cols, vals, vect, constant
end

function qp_model(moimodel::MOI.ModelLike)
    index_map, nvar, lvar, uvar, x0 = parse_variable(moimodel)
    Ai, Aj, Ax, lb, ub = parse_constraints(moimodel, index_map)
    Qi, Qj, Qx, c, d = parse_objective(moimodel, index_map, nvar)

    nnzQ = length(Qi)
    for k in 1:nnzQ
        if Qi[k] < Qj[k]
            tmp = Qj[k]
            Qj[k] = Qi[k]
            Qi[k] = tmp
        end
    end

    minimize = MOI.get(moimodel, MOI.ObjectiveSense()) == MOI.MIN_SENSE

    ncon = length(lb)

    A = SparseMatrixCOO(ncon, nvar, Ai, Aj, Ax)
    Q = SparseMatrixCOO(nvar, nvar, Qi, Qj, Qx)

    data = QuadraticModels.QPData(
        d,
        c,
        zeros(length(c)),
        Q,
        A,
    )
    return QuadraticModels.QuadraticModel(
        NLPModels.NLPModelMeta(
            nvar;
            ncon=ncon,
            lvar=lvar,
            uvar=uvar,
            lcon=lb,
            ucon=ub,
            x0=x0,
            y0=zeros(ncon),
            nnzj=length(Ai),
            lin_nnzj=length(Ai),
            lin=collect(1:ncon),
            nnzh=length(Qi),
            minimize=minimize,
        ),
        NLPModels.Counters(),
        data,
    ), index_map
end
