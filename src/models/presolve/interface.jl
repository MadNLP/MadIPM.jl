"""
    PresolveStatus

Outcome of [`apply_presolve`](@ref): `PRESOLVE_UNCHANGED` (nothing to reduce),
`PRESOLVE_REDUCED`, `PRESOLVE_INFEASIBLE`, `PRESOLVE_UNBOUNDED`,
`PRESOLVE_UNBOUNDED_OR_INFEASIBLE`, or `PRESOLVE_SOLVED` (every variable
eliminated; `recover_solution(result, T[], T[])` yields the optimum).
"""
@enum PresolveStatus::UInt8 begin
  PRESOLVE_UNCHANGED               = 0
  PRESOLVE_REDUCED                 = 1
  PRESOLVE_INFEASIBLE              = 2
  PRESOLVE_UNBOUNDED               = 3
  PRESOLVE_UNBOUNDED_OR_INFEASIBLE = 4
  PRESOLVE_SOLVED                  = 5
end

"""Abstract supertype for presolver configurations."""
abstract type AbstractPresolver end

"""No-op presolver. `apply_presolve(NoPresolver(), model)` always returns `(PRESOLVE_UNCHANGED, NoPresolveResult(model))`."""
struct NoPresolver <: AbstractPresolver end

"""
Opaque handle returned by [`apply_presolve`](@ref); implements
[`recover_solution`](@ref) and, except for `PRESOLVE_SOLVED`, carries a
`reduced_model` field to hand to the downstream solver.
"""
abstract type AbstractPresolveResult end

"""
    apply_presolve(presolver, model) -> (status::PresolveStatus, result)

Apply `presolver` to a [`LinearModel`](@ref) or [`QuadraticModel`](@ref).
`result` is an [`AbstractPresolveResult`](@ref), or `nothing` when presolve
proves the problem infeasible/unbounded.
"""
function apply_presolve end

"""
    recover_solution(result, x_reduced, y_reduced) -> (x, y)

Reconstruct an original-problem primal/dual solution from a reduced-problem one.
The eliminated primal entries are filled with the values determined during
presolve (typically a fixed bound); the eliminated dual rows are zero.
"""
function recover_solution end


# --- Trivial NoPresolver implementation -------------------------------------

"""Result for [`NoPresolver`](@ref). Aliases the original model — no copy."""
struct NoPresolveResult{M<:ScalarModel} <: AbstractPresolveResult
  reduced_model::M
end

apply_presolve(::NoPresolver, model::ScalarModel) =
  (PRESOLVE_UNCHANGED, NoPresolveResult(model))

recover_solution(::NoPresolveResult, x::AbstractVector, y::AbstractVector) = (x, y)
