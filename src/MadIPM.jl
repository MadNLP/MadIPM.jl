module MadIPM

# Hello world

using Printf
using LinearAlgebra
import SparseArrays
import MadNLP
import MadNLP: full, LDLFactorizations
import NLPModels
import QuadraticModels
import QuadraticModels: QPData, QuadraticModel, SparseMatrixCOO

import BatchQuadraticModels
import BatchQuadraticModels:
  ObjRHSBatchQuadraticModel,
  BatchSparseOp,
  batch_spmv!,
  batch_mapreduce!,
  batch_maximum!

include("utils.jl")
include("structure.jl")
include("kernels.jl")
include("KKT/normalkkt.jl")
include("linear_solver.jl")
include("solver.jl")

include("batch/utils.jl")
include("batch/views.jl")
include("batch/madnlp/rhs.jl")
include("batch/madnlp/callback.jl")
include("batch/KKT/KKT.jl")
include("batch/structure.jl")
include("batch/madnlp/linear_solver.jl")
include("batch/madnlp/kernels.jl")
include("batch/madnlp/initialization.jl")
include("batch/madnlp/nlpmodels.jl")
include("batch/madipm/kernels.jl")
include("batch/madipm/solver.jl")

export MPCSolver, madipm, madipm_batch

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
