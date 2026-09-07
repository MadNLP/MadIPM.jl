const RUIZ_DEFAULT_MAXITER = 10  # matches MC77's ICNTL(7) default

abstract type AbstractScaling{R,C} end

struct ScalingConvergenceError <: Exception
    max_iter::Int
    eps::Float64
end

Base.showerror(io::IO, err::ScalingConvergenceError) = print(
    io,
    "scaling did not converge after ",
    err.max_iter,
    " iterations at eps=",
    err.eps,
)

storage_vector(A::SparseMatrixCSC{T}, n, value::T) where {T} =
    fill!(similar(nonzeros(A), T, n), value)

scaling_vectors(A) = storage_vector(A, size(A, 1), one(eltype(A))),
storage_vector(A, size(A, 2), one(eltype(A)))
