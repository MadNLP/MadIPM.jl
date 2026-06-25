
include("common.jl")

CUDA.device!(1)

const NETLIB_PATH = fetch_netlib()
const MIPLIB_INSTANCES = "miplib_problems.txt"
const WARMUP_INSTANCE = "ADLITTLE.SIF"

function select_netlib_instance()
    cases = filter(x -> endswith(x, ".SIF"), readdir(NETLIB_PATH))
    selected = String[]
    for case in cases
        try
            qp = readqps(joinpath(NETLIB_PATH, case))
            push!(selected, case)
        catch ex
            println("Fail to load $(case)")
        end
    end
    return selected
end

function select_miplib_instance()
    return readdlm(joinpath(@__DIR__, MIPLIB_INSTANCES))[:]
end

@memoize function load_netlib_instance(case)
    qpdat = readqps(joinpath(NETLIB_PATH, case))
    return QuadraticModel(qpdat)
end

@memoize function load_miplib_instance(case)
    return MIPLIB.miplib2010(case)
end

function benchmark_lps(cases, batches, load_instance; bench_options...)
    m = 5 + 2*length(batches)
    shift1 = 5
    shift2 = shift1 + length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        # Load instance
        qp = load_instance(case)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        # Launch on CPU
        try
            cpu_solver = MadIPM.MPCSolver(
                qp;
                print_level=MadNLP.ERROR,
                tol=1e-6,
                max_iter=300,
                regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
                linear_solver=Ma27Solver,
            )
            stats = MadIPM.solve!(cpu_solver)
            results[k, 4] = stats.iter
            results[k, 5] = stats.counters.total_time
        catch ex
            println("$(case) fails with message $(ex)")
            results[k, 4] = -1
            results[k, 5] = -1
        end
        # Launch on GPU (batch)
        qps = build_qps(qp, batches[end]; bench_options...)
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            try
                cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:batch])
                gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
                gpu_solver = MadIPM.UniformBatchMPCSolver(
                    gpu_bnlp;
                    print_level=MadNLP.ERROR,
                    max_iter=300,
                    tol=1e-6,
                    regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
                    uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                    cudss_algorithm = MadNLP.LDL,
                    cudss_pivot_epsilon=1e-8,
                )
                stats = MadIPM.solve!(gpu_solver)
                results[k, shift1+l] = sum(stats.iter) / batch
                results[k, shift2+l] = sum(stats.total_time) / batch
            catch ex
                println("$(case) fails with message $(ex)")
                results[k, shift1+l] = -1
                results[k, shift2+l] = -1
                refresh_memory()
            end
        end
    end

    return [cases results]
end

function main()
    @info "Warmup"
    bench = :miplib
    _warmup(load_netlib_instance(WARMUP_INSTANCE))
    batches = [1, 2, 4, 8]
    if bench == :netlib
        cases = select_netlib_instance()
        results = benchmark_lps(cases, batches, load_netlib_instance)
        writedlm(joinpath("results", "2-benchmark-netlib.csv"), results)
    elseif bench == :miplib
        cases = select_miplib_instance()
        results = benchmark_lps(cases, batches, load_miplib_instance)
        writedlm(joinpath("results", "2-benchmark-miplib.csv"), results)
    end
    return
end

main()
