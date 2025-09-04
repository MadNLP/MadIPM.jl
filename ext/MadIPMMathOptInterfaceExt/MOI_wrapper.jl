
mutable struct Optimizer <: MOI.AbstractOptimizer
    options::Dict{String, Any}
    silent::Bool
    solver::Union{Nothing, MadIPM.MPCSolver}
    qp::Union{Nothing, QuadraticModel}
    array_type::Type{<:AbstractVector{Float64}}
    stats::Union{
        Nothing,
        MadNLP.MadNLPExecutionStats{Float64, <:AbstractVector{Float64}},
    }
    function Optimizer()
        return new(Dict{String, Any}(), false, nothing, nothing, Vector{Float64}, nothing)
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "MadIPM"

MOI.is_empty(optimizer::Optimizer) = isnothing(optimizer.solver) && isnothing(optimizer.qp)

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.qp = nothing
    optimizer.stats = nothing
    return
end

###
### MOI.RawOptimizerAttribute
###

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    if param.name == "array_type"
        optimizer.array_type = value
    else
        optimizer.options[param.name] = value
    end
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    return optimizer.options[param.name]
end

###
### MOI.Silent
###

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

###
### MOI.AbstractModelAttribute
###

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{<:Union{VI, SAF, SQF}},
    },
)
    return true
end


###
### MOI.AbstractVariableAttribute
###

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex})
    return true
end

###
### `supports_constraint`
###

MOI.supports_constraint(::Optimizer, ::Type{VI}, ::Type{<:ALS}) = true
MOI.supports_constraint(::Optimizer, ::Type{SAF}, ::Type{<:ALS}) = true
MOI.supports_constraint(::Optimizer, ::Type{VAF}, ::Type{<:VLS}) = true

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    dest.qp, index_map = qp_model(src)
    if dest.array_type != Vector{Float64}
        VT = dest.array_type
        T = eltype(VT)
        dest.qp = convert(QuadraticModel{T, VT}, dest.qp)
    end
    return index_map
end

function MOI.optimize!(model::Optimizer)
    options = Dict{Symbol, Any}(
        Symbol(key) => model.options[key] for key in keys(model.options) if key != "solver"
    )
    if model.silent
        options[:print_level] = MadNLP.ERROR
    else
        options[:print_level] = MadNLP.INFO
    end
    model.solver = MadIPM.MPCSolver(model.qp; options...)
    model.stats = MadIPM.solve!(model.solver)
    return
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.stats.counters.total_time
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    return string(optimizer.stats.status)
end

struct RawStatus <: MOI.AbstractModelAttribute
    name::Symbol
end

MOI.is_set_by_optimize(::RawStatus) = true

function MOI.get(optimizer::Optimizer, attr::RawStatus)
    return getfield(optimizer.stats, attr.name)
end

const TERMINATION_STATUS = Dict{MadNLP.Status,MOI.TerminationStatusCode}(
    MadNLP.SOLVE_SUCCEEDED => MOI.OPTIMAL,
    MadNLP.SOLVED_TO_ACCEPTABLE_LEVEL => MOI.ALMOST_OPTIMAL,
    MadNLP.SEARCH_DIRECTION_BECOMES_TOO_SMALL => MOI.SLOW_PROGRESS,
    MadNLP.DIVERGING_ITERATES => MOI.INFEASIBLE_OR_UNBOUNDED,
    MadNLP.INFEASIBLE_PROBLEM_DETECTED => MOI.INFEASIBLE,
    MadNLP.MAXIMUM_ITERATIONS_EXCEEDED => MOI.ITERATION_LIMIT,
    MadNLP.MAXIMUM_WALLTIME_EXCEEDED => MOI.TIME_LIMIT,
    MadNLP.INITIAL => MOI.OPTIMIZE_NOT_CALLED,
    MadNLP.RESTORATION_FAILED => MOI.NUMERICAL_ERROR,
    MadNLP.INVALID_NUMBER_DETECTED => MOI.INVALID_MODEL,
    MadNLP.ERROR_IN_STEP_COMPUTATION => MOI.NUMERICAL_ERROR,
    MadNLP.NOT_ENOUGH_DEGREES_OF_FREEDOM => MOI.INVALID_MODEL,
    MadNLP.USER_REQUESTED_STOP => MOI.INTERRUPTED,
    MadNLP.INTERNAL_ERROR => MOI.OTHER_ERROR,
    MadNLP.INVALID_NUMBER_OBJECTIVE => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_GRADIENT => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_CONSTRAINTS => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_JACOBIAN => MOI.INVALID_MODEL,
    MadNLP.INVALID_NUMBER_HESSIAN_LAGRANGIAN => MOI.INVALID_MODEL,
)

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.stats)
        return MOI.OPTIMIZE_NOT_CALLED
    end
    return TERMINATION_STATUS[optimizer.stats.status]
end

function MOI.get(optimizer::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.objective
end

function MOI.get(optimizer::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index > MOI.get(optimizer, MOI.ResultCount())
        return MOI.NO_SOLUTION
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif MOI.get(optimizer, MOI.TerminationStatus()) == MOI.INFEASIBLE
        return MOI.INFEASIBLE_POINT
    else
        # TODO
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(::Optimizer, ::MOI.DualStatus)
    # TODO
    return MOI.NO_SOLUTION
end

function MOI.get(optimizer::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(optimizer, attr)
    return optimizer.stats.solution[vi.value]
end

MOI.get(optimizer::Optimizer, ::MOI.ResultCount) = 1
