module MadIPM

using Printf
using LinearAlgebra
import SparseArrays
import MadNLP
import MadNLP: full, LDLFactorizations
import NLPModels
import QuadraticModels
import QuadraticModels: SparseMatrixCOO

include("utils.jl")
include("structure.jl")
include("kernels.jl")
include("KKT/normalkkt.jl")
include("linear_solver.jl")
include("solver.jl")

export MPCSolver, madipm

MadNLP.madsuite(::Val{:madipm}, args...; kwargs...) = madipm(args...; kwargs...)

end # module MadIPM
