#=
    Adapted from NLPModelsJuMP.jl

    The following source code is subject to the following LICENSE:

    Copyright (c) 2018-2019: Abel Soares Siqueira and Dominique Orban
    NLPModelsJuMP.jl is licensed under the MPL version 2.0.

    Full license available at this URL:
    https://github.com/JuliaSmoothOptimizers/NLPModelsJuMP.jl/blob/main/LICENSE.md

=#

module MadIPMMathOptInterfaceExt

using MathOptInterface
using QuadraticModels
import NLPModels
import MadNLP
import MadIPM

const MOI = MathOptInterface

include("parse_moi.jl")
include("MOI_wrapper.jl")

function __init__()
    @eval MadIPM const Optimizer = $Optimizer
end

end

