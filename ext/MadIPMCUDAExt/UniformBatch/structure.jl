abstract type AbstractBatchSolver end

struct UniformBatchSolver{VS,BK,SD} <: AbstractBatchSolver
    solvers::VS
    bkkt::BK
    step::SD

    function UniformBatchSolver(solvers::Vector{Solver}; linear_solver::Type, kwargs...) where {Solver<:MadIPM.MPCSolver}
        batch_size = length(solvers)
        solver1 = first(solvers)
        nlp1 = solver1.nlp
        kkt1 = solver1.kkt
        vec1 = solver1.d

        kkts = Vector{typeof(kkt1)}(undef, batch_size)
        vecs = Vector{typeof(vec1)}(undef, batch_size)
        for i in 1:batch_size
            solver_i = solvers[i]
            kkts[i] = solver_i.kkt
            vecs[i] = solver_i.d
        end

        options = MadIPM.load_options(nlp1; linear_solver=linear_solver, kwargs...)
        bkkt = UniformBatchKKTSystem(kkts, vecs, linear_solver, opt_linear_solver=options.linear_solver)
        step = BatchStepData(solver1, batch_size)
        return new{Vector{Solver},typeof(bkkt),typeof(step)}(solvers, bkkt, step)
    end
end

all_done(batch_solver::UniformBatchSolver) = all_done(batch_solver.bkkt)
is_active(batch_solver::UniformBatchSolver, i) = is_active(batch_solver.bkkt, i)
Base.length(batch_solver::UniformBatchSolver) = length(batch_solver.solvers)
Base.iterate(batch_solver::UniformBatchSolver, i=1) = iterate(batch_solver.solvers, i)
Base.getindex(batch_solver::UniformBatchSolver, i) = batch_solver.solvers[i]

update_batch!(batch_solver::UniformBatchSolver) = begin
    needs_update = false
    for (i, solver) in enumerate(batch_solver)
        if is_active(batch_solver, i) && MadIPM.is_done(solver)
            needs_update = true
            batch_solver.bkkt.is_active[i] = false
        end
    end
    needs_update && update_batch!(batch_solver.bkkt)
    return
end

Base.eltype(batch_solver::UniformBatchSolver) = eltype(batch_solver.step.x_lr)