module TestMOI

using Test

import MathOptInterface as MOI
import MadIPM

function test_runtests()
    excludes = [
        "test_model_copy_to_UnsupportedAttribute",
        r"^test_linear_integration$",
        # Currently not supported
        "test_linear_complex",
        "test_quadratic",
        "test_conic",
        # MadNLP.SymbolicException() with MUMPS
        "test_solve_VariableIndex_ConstraintDual_MIN_SENSE",
        "test_solve_VariableIndex_ConstraintDual_MAX_SENSE",
    ]
    model = MOI.instantiate(MadIPM.Optimizer, with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true) # comment this to enable output
    config = MOI.Test.Config(
        atol = 1e-6,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(model, config, exclude=excludes)
    return
end

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

end  # module

TestMOI.runtests()
