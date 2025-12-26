## for each KKT system:
get_rhs_size(kkt::MadNLP.AbstractReducedKKTSystem, vec::MadNLP.UnreducedKKTVector) = length(get_rhs(typeof(kkt), vec))
get_rhs(::Type{<:MadNLP.AbstractReducedKKTSystem}, vec::MadNLP.UnreducedKKTVector) = MadNLP.primal_dual(vec)

# TODO: better modularize; below combine MadIPM.solve_system! and MadNLP.solve!(::AbstractReducedKKTSystem)
pre_solve!(solver::MadIPM.MPCSolver{T,VT,VI,KKTSystem}) where {
    T,VT,VI,KKTSystem<:MadNLP.AbstractReducedKKTSystem,
} = begin
    copyto!(MadNLP.full(solver.d), MadNLP.full(solver.p))
    MadNLP.reduce_rhs!(solver.kkt, solver.d)
    return
end
post_solve!(solver::MadIPM.MPCSolver{T,VT,VI,KKTSystem}) where {
    T,VT,VI,KKTSystem<:MadNLP.AbstractReducedKKTSystem,
} = begin
    MadNLP.finish_aug_solve!(solver.kkt, solver.d)
    MadIPM.check_residual!(solver.d, solver, solver.p)
end

## dummy solver to make sure batch solve uses batch solver only
struct NoLinearSolver{T} <: MadNLP.AbstractLinearSolver{T} end
NoLinearSolver(A; kwargs...) = NoLinearSolver{Float64}()
MadNLP.default_options(::Type{NoLinearSolver}) = nothing
MadNLP.set_options!(::Nothing, x) = x
MadNLP.is_supported(::Type{NoLinearSolver}, ::Type{T}) where {T<:AbstractFloat} = true


include("kkt.jl")
include("structure.jl")
include("solver.jl")