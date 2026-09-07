struct SparseCOO{T,Ti<:Integer,V<:AbstractVector{T},Vi<:AbstractVector{Ti}} <:
       SparseArrays.AbstractSparseMatrix{T,Ti}
    m::Int
    n::Int
    rowval::Vi
    colval::Vi
    nzval::V
end

SparseCOO(
    m::Integer,
    n::Integer,
    rowval::Vi,
    colval::Vi,
    nzval::V,
) where {T,Ti,V<:AbstractVector{T},Vi<:AbstractVector{Ti}} =
    SparseCOO{T,Ti,V,Vi}(m, n, rowval, colval, nzval)

function SparseCOO(A::SparseArrays.SparseMatrixCSC)
    rowval, colval, nzval = SparseArrays.findnz(A)
    return SparseCOO(size(A, 1), size(A, 2), rowval, colval, nzval)
end

SparseCOO(A::SparseMatrixCOO) = SparseCOO(A.m, A.n, A.rows, A.cols, A.vals)

Base.size(A::SparseCOO) = (A.m, A.n)
SparseArrays.nnz(A::SparseCOO) = length(A.nzval)
SparseArrays.nonzeros(A::SparseCOO) = A.nzval
Base.copy(A::SparseCOO) = SparseCOO(A.m, A.n, copy(A.rowval), copy(A.colval), copy(A.nzval))

function Base.getindex(A::SparseCOO{T}, i::Integer, j::Integer) where {T}
    @inbounds for k in eachindex(A.nzval)
        A.rowval[k] == i && A.colval[k] == j && return A.nzval[k]
    end
    return zero(T)
end

storage_vector(A::SparseCOO{T}, n, value::T) where {T} =
    fill!(similar(A.nzval, T, n), value)
