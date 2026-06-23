
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
        # Launch on CPU
        qp = load_instance(case)
        cpu_solver = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.ERROR,
            max_iter=500,
            regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
            linear_solver=Ma57Solver,
        )
        stats = MadIPM.solve!(cpu_solver)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        results[k, 4] = stats.iter
        results[k, 5] = stats.counters.total_time
        # Launch on GPU (batch)
        qps = build_qps(qp, batches[end]; bench_options...)
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:batch])
            gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
            gpu_solver = MadIPM.UniformBatchMPCSolver(
                gpu_bnlp;
                print_level=MadNLP.ERROR,
                max_iter=500,
                regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
                uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_algorithm = MadNLP.LDL,
                cudss_pivot_epsilon=1e-8,
            )
            stats = MadIPM.solve!(gpu_solver)
            results[k, shift1+l] = sum(stats.iter) / batch
            results[k, shift2+l] = sum(stats.total_time) / batch
        end
    end

    return [cases results]
end

function main()
    @info "Warmup"
    bench = :netlib
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
