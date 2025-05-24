using DelimitedFiles
using MadNLP
using MadQP
using QPSReader
using QuadraticModels

include("common.jl")
include("excluded_problems.jl")
include("cuda_wrapper.jl")
include("qp_gpu.jl")

function run_benchmark(src, probs; reformulate::Bool=false, test_reader::Bool=false)
    nprobs = length(probs)
    results = zeros(nprobs, 9)
    for (k, prob) in enumerate(probs)
        @info "$prob -- $k / $nprobs"
        qpdat = try
            import_mps(joinpath(src, prob))
        catch e
            @warn "Failed to import $prob: $e"
            continue
        end
        @info "The problem $prob was imported."

        if !test_reader
            # Instantiate QuadraticModel
            qp = QuadraticModel(qpdat)
            presolved_qp, flag = presolve_qp(qp)
            !flag && continue  # problem already solved, unbounded or infeasible
            scaled_qp = scale_qp(presolved_qp)
            qp_cpu = reformulate ? standard_form_qp(scaled_qp) : scaled_qp

            # Transfer data to the GPU
            qp_gpu = transfer_to_gpu(qp_cpu)

            try
                solver = MadQP.MPCSolver(
                    qp_gpu;
                    max_iter=300,
                    tol=1e-7,
                    linear_solver=MadNLPGPU.CUDSSSolver,
                    cudss_algorithm=MadNLP.LDL,
                    print_level=MadNLP.INFO,
                    max_ncorr=3,
                    bound_push=1.0,
                    scaling=true,
                    step_rule=MadQP.AdaptiveStep(0.995),
                    regularization=MadQP.FixedRegularization(1e-8, -1e-8),
                    rethrow_error=true,
                )
                res = MadQP.solve!(solver)
                results[k, 1] = Int(qp_gpu.meta.nvar)
                results[k, 2] = Int(qp_gpu.meta.ncon)
                results[k, 3] = Int(qp_gpu.meta.nnzj)
                results[k, 4] = Int(qp_gpu.meta.nnzh)
                results[k, 5] = Int(res.status)
                results[k, 6] = res.iter
                results[k, 7] = res.objective
                results[k, 8] = res.counters.total_time
                results[k, 9] = res.counters.linear_solver_time
            catch ex
                @warn "Failed to solve $prob: $ex"
                results[k, 8] = -1
                continue
            end
        end
    end
    return results
end

# src = fetch_netlib()
# mps_files = filter(x -> endswith(x, ".SIF") && !(x in excluded_netlib), readdir(src))
# name_results = "benchmark-netlib-gpu.txt"

# src = fetch_mm()
# mps_files = filter(x -> endswith(x, ".SIF") && !(x in excluded_mm), readdir(src))
# name_results = "benchmark-mm-gpu.txt"

src = "/home/amontoison/Argonne/miplib"
mps_files = filter(x -> endswith(x, ".mps.gz") && !(x in excluded_miplib), readdir(src))
name_results = "benchmark-miplib-gpu.txt"

# src = "/home/amontoison/Argonne/large-scale-LPs"
# mps_files = filter(x -> endswith(x, ".mps.gz") || endswith(x, ".mps"), readdir(src))
# name_results = "benchmark-fp-gpu.txt"

# variant = "medium"
# src = "/home/amontoison/Argonne/LP_instances/$variant-problem-instances"
# mps_files = filter(x -> endswith(x, ".mps.gz"), readdir(src))
# name_results = "benchmark-$variant-pdlp-gpu.txt"

reformulate = false
test_reader = false
results = run_benchmark(src, mps_files; reformulate, test_reader)
path_results = joinpath(@__DIR__, "tables", name_results)
writedlm(path_results, [mps_files results])
