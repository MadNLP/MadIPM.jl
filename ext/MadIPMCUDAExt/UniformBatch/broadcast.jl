struct ActiveSolversStyle <: Base.Broadcast.BroadcastStyle end
struct ActiveSolversIterator{S}
    batch_solver::UniformBatchSolver{S}
end

Base.Broadcast.BroadcastStyle(::Type{<:UniformBatchSolver}) = ActiveSolversStyle()
Base.Broadcast.BroadcastStyle(::ActiveSolversStyle, ::Base.Broadcast.DefaultArrayStyle{0}) = ActiveSolversStyle()
Base.Broadcast.instantiate(bc::Base.Broadcast.Broadcasted{ActiveSolversStyle}) = bc
Base.broadcastable(batch_solver::UniformBatchSolver) = ActiveSolversIterator(batch_solver)
Base.axes(iter::ActiveSolversIterator) = (Base.OneTo(iter.batch_solver.bkkt.active_batch_size[]),)
Base.ndims(::Type{<:ActiveSolversIterator}) = 1
Base.getindex(iter::ActiveSolversIterator, i::Int) = iter.batch_solver.solvers[iter.batch_solver.bkkt.batch_map_rev[i]]

# FIXME:  this works but does not seem to be the correct way to do it
@inline function Base.Broadcast.materialize(bc::Base.Broadcast.Broadcasted{ActiveSolversStyle})
    iter = bc.args[1]
    bkkt = iter.batch_solver.bkkt
    active_i = 1
    while (solver_idx = bkkt.batch_map_rev[active_i]) != 0
        bc.f(iter.batch_solver.solvers[solver_idx])
        active_i += 1
    end
    return nothing
end