using DelimitedFiles
using MadNLP
using MadIPM
using MadNLPHSL
using QPSReader
using QuadraticModels
using SparseArrays

include("common.jl")
include("excluded_problems.jl")

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
            qp = QuadraticModel(qpdat)
            presolved_qp, flag = MadIPM.presolve_qp(qp)
            !flag && continue  # problem already solved, unbounded or infeasible
            scaled_qp = scale_qp(presolved_qp)
            qp_cpu = reformulate ? MadIPM.standard_form_qp(scaled_qp) : scaled_qp

            try
                solver = MadIPM.MPCSolver(
                    qp_cpu;
                    max_iter=300,
                    linear_solver=Ma57Solver,
                    print_level=MadNLP.INFO,
                    rethrow_error=true,
                )
                res = MadIPM.solve!(solver)
                results[k, 1] = Int(qp_cpu.meta.nvar)
                results[k, 2] = Int(qp_cpu.meta.ncon)
                results[k, 3] = Int(qp_cpu.meta.nnzj)
                results[k, 4] = Int(qp_cpu.meta.nnzh)
                results[k, 5] = Int(res.status)
                results[k, 6] = res.iter
                results[k, 7] = res.objective
                results[k, 8] = res.counters.total_time
                results[k, 9] = res.counters.linear_solver_time
            catch ex
                results[k, 8] = -1
                @warn "Failed to solve $prob: $ex"
                continue
            end
        end
    end
    return results
end

# src = fetch_netlib()
# name_results = "benchmark-netlib-cpu.txt"
# mps_files = filter(x -> endswith(x, ".SIF") && !(x in excluded_netlib), readdir(src))

# src = fetch_mm()
# name_results = "benchmark-mm-cpu.txt"
# mps_files = filter(x -> endswith(x, ".SIF") && !(x in excluded_mm), readdir(src))

src = joinpath(@__DIR__, "instances", "miplib2010")
mps_files = readdlm(joinpath(@__DIR__, "miplib_problems.txt"))[:]
name_results = "benchmark-miplib-cpu.txt"

reformulate = true
test_reader = false
results = run_benchmark(src, mps_files; reformulate, test_reader)
path_results = joinpath(@__DIR__, "tables", name_results)
writedlm(path_results, [mps_files results])
