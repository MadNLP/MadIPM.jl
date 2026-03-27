# FIXME: threads/polyester version

struct LoopedBatchLinearSolver{T, VT, LS<:MadNLP.AbstractLinearSolver{T}} <: MadNLP.AbstractLinearSolver{T}
    solvers::Vector{LS}
    batch_size::Int
end

@kwdef mutable struct LoopedBatchLinearSolverOptions <: MadNLP.AbstractOptions
    looped_linear_solver::Type = MadNLP.MumpsSolver
end

MadNLP.default_options(::Type{LoopedBatchLinearSolver}) = LoopedBatchLinearSolverOptions()

function LoopedBatchLinearSolver(
    aug_com,
    nzvals_mat::AbstractMatrix{T},
    n::Int;
    opt::LoopedBatchLinearSolverOptions = LoopedBatchLinearSolverOptions(),
) where T
    linear_solver = opt.looped_linear_solver
    per_instance_opt = MadNLP.default_options(linear_solver)
    batch_size = size(nzvals_mat, 2)
    nnz_csc = size(nzvals_mat, 1)
    VT = typeof(similar(nzvals_mat, T, 0))
    individual_solvers = map(1:batch_size) do i
        nzval_i = _madnlp_unsafe_column_wrap(nzvals_mat, nnz_csc, (i - 1) * nnz_csc + 1, VT)
        csc_i = _csc_with_nzval(aug_com, nzval_i, n)
        linear_solver(csc_i; opt=per_instance_opt)
    end
    LoopedBatchLinearSolver{T, VT, eltype(individual_solvers)}(individual_solvers, batch_size)
end

function is_factorized(batch_linear_solver::LoopedBatchLinearSolver)
    return all(is_factorized(s) for s in batch_linear_solver.solvers)
end

function _active_factorize!(s::LoopedBatchLinearSolver, na::Int)
    for j in 1:na
        MadNLP.factorize!(s.solvers[j])
    end
    return
end

function _active_solve!(s::LoopedBatchLinearSolver{T, VT}, rhs::AbstractMatrix{T}, na::Int, n::Int) where {T, VT}
    for j in 1:na
        rhs_j = _madnlp_unsafe_column_wrap(rhs, n, (j-1)*n + 1, VT)
        MadNLP.solve_linear_system!(s.solvers[j], rhs_j)
    end
    return
end
