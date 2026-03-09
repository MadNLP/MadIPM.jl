@testset "Batch kernel tests" begin
    include("setup.jl")
    include("test_rhs.jl")
    include("test_kkt.jl")
    include("test_complementarity.jl")
    include("test_step.jl")
    include("test_regularization.jl")
    include("test_solver_steps.jl")
    include("test_batch_multi.jl")
    include("test_termination.jl")
    include("test_iteration_match.jl")
end
