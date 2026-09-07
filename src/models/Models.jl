module Models

using Adapt
using LinearAlgebra, SparseArrays
using NLPModels
using SparseMatricesCOO

abstract type AbstractSparseOperator{T} <: AbstractMatrix{T} end

include("utils.jl")
include("sparse_operator.jl")
include("scalar_models.jl")
include("batch_operator.jl")
include("batch_models.jl")

module Presolve
using LinearAlgebra
using SparseArrays
using SparseMatricesCOO: SparseMatrixCOO
import ..Models:
    LinearModel,
    QuadraticModel,
    LPData,
    QPData,
    ScalarModel,
    operator_sparse_matrix,
    sparse_operator

include("presolve/interface.jl")
include("presolve/basic.jl")
end # module Presolve

module Scaling
using LinearAlgebra
using SparseArrays
import SparseArrays: getcolptr
using SparseMatricesCOO: SparseMatrixCOO
import ..Models: LinearModel, QuadraticModel, LPData, QPData, operator_sparse_matrix

include("scaling/utils.jl")
include("scaling/sparse_coo.jl")
include("scaling/reductions.jl")
include("scaling/ruiz.jl")
include("scaling/scale_model.jl")
end # module Scaling

export ScalarModel, LinearModel, QuadraticModel, LPData, QPData
export BatchQuadraticModel,
    ObjRHSBatchQuadraticModel, UniformBatchQuadraticModel, batch_model
export BatchLinearModel, ObjRHSBatchLinearModel, UniformBatchLinearModel
export BatchSparseOperator, batch_spmv!, sync_batch_operator!
export batch_mapreduce!, batch_maximum!

end # module Models
