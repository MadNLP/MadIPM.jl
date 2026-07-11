struct SparseOperator{T,S,O} <: AbstractSparseOperator{T}
  source::S
  op::O
end

Base.size(A::SparseOperator) = size(A.source)
SparseArrays.nnz(A::SparseOperator) = nnz(A.source)

operator_sparse_matrix(A::SparseOperator) = A.source

SparseArrays.findnz(A::SparseOperator) = SparseArrays.findnz(A.source)
_sparse_structure(A::SparseOperator) = _sparse_structure(A.source)
_sparse_values(A::SparseOperator)    = _sparse_values(A.source)
_copy_sparse_structure!(A::SparseOperator, rows::AbstractVector, cols::AbstractVector) =
  _copy_sparse_structure!(A.source, rows, cols)
_copy_sparse_values!(A::SparseOperator, vals::AbstractVector) =
  _copy_sparse_values!(A.source, vals)

function sparse_operator(A::Union{SparseMatrixCOO{T}, SparseMatrixCSC{T}};
                         transa::Char = 'N', symmetric::Bool = false) where {T}
  csc = A isa SparseMatrixCSC ? A : sparse(A.rows, A.cols, A.vals, size(A)...)
  op  = symmetric ? Symmetric(csc, :L) : csc
  wrapped = SparseOperator{T, typeof(A), typeof(op)}(A, op)
  return transa == 'N' ? wrapped : transpose(wrapped)
end

sparse_operator(A::AbstractSparseOperator; transa::Char = 'N', symmetric::Bool = false) =
  transa == 'N' ? A : transpose(A)

function Adapt.adapt_structure(to, A::SparseOperator{T}) where {T}
  source = Adapt.adapt(to, A.source)
  op = Adapt.adapt(to, A.op)
  return SparseOperator{T, typeof(source), typeof(op)}(source, op)
end

LinearAlgebra.mul!(Y::AbstractMatrix{T}, A::SparseOperator{T}, X::AbstractMatrix{T}, α::Number, β::Number) where {T} =
  mul!(Y, A.op, X, α, β)
LinearAlgebra.mul!(y::AbstractVector{T}, A::SparseOperator{T}, x::AbstractVector{T}, α::Number, β::Number) where {T} =
  mul!(y, A.op, x, α, β)

_mul_jt!(jtv, A::SparseOperator, v) = mul!(jtv, transpose(A.op), v)
