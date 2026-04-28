using LDLFactorizations

mutable struct FailOnDemandLS{T,VT,LS<:MadNLP.AbstractLinearSolver{T}} <: MadNLP.AbstractLinearSolver{T}
    solvers::Vector{LS}
    batch_size::Int
    fail_positions::Set{Int}
    fail_remaining::Int
end

@kwdef mutable struct FailOnDemandLSOptions <: MadNLP.AbstractOptions
    looped_linear_solver::Type = MadNLP.LDLSolver
end

MadNLP.default_options(::Type{FailOnDemandLS}) = FailOnDemandLSOptions()

function FailOnDemandLS(aug_com, nzvals_mat::AbstractMatrix{T}, n::Int;
        opt=FailOnDemandLSOptions()) where T
    bs = size(nzvals_mat, 2)
    nnz_csc = size(nzvals_mat, 1)
    VT = typeof(similar(nzvals_mat, T, 0))
    solvers = map(1:bs) do i
        nzval_i = MadIPM._madnlp_unsafe_column_wrap(nzvals_mat, nnz_csc, (i-1)*nnz_csc+1, VT)
        csc_i = MadIPM._csc_with_nzval(aug_com, nzval_i, n)
        opt.looped_linear_solver(csc_i; opt=MadNLP.default_options(opt.looped_linear_solver))
    end
    FailOnDemandLS{T,VT,eltype(solvers)}(solvers, bs, Set{Int}(), 0)
end

MadIPM.is_factorized(s::FailOnDemandLS) = all(MadIPM.is_factorized(sj) for sj in s.solvers)

function MadIPM.is_factorized!(buf::Vector{Int32}, s::FailOnDemandLS, v::MadIPM.BatchView)
    nf = 0
    @inbounds for j in 1:v.n
        if !MadIPM.is_factorized(s.solvers[j])
            nf += 1
            buf[nf] = j
        end
    end
    return nf
end

function MadIPM.factorize_active!(s::FailOnDemandLS, v::MadIPM.BatchView)
    fail = s.fail_remaining > 0
    if fail
        s.fail_remaining -= 1
    end
    @inbounds for j in 1:v.n
        if fail && j in s.fail_positions
            nz = s.solvers[j].tril.nzval
            saved = nz[1]
            nz[1] = 0.0
            MadNLP.factorize!(s.solvers[j])
            nz[1] = saved
        else
            MadNLP.factorize!(s.solvers[j])
        end
    end
end

function MadIPM.solve_active!(s::FailOnDemandLS{T,VT}, rhs::AbstractMatrix{T}, v::MadIPM.BatchView) where {T,VT}
    n = size(rhs, 1)
    @inbounds for j in 1:v.n
        rhs_j = MadIPM._madnlp_unsafe_column_wrap(rhs, n, (j-1)*n+1, VT)
        MadNLP.solve_linear_system!(s.solvers[j], rhs_j)
    end
end
