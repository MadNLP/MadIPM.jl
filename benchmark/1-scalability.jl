
include("common.jl")

CUDA.device!(1)

const NETLIB_PATH = fetch_netlib()
const NNZJ_THRESHOLD = 5_000
const WARMUP_INSTANCE = "ADLITTLE.SIF"

@memoize function load_instance(case)
    qpdat = readqps(joinpath(NETLIB_PATH, case))
    return QuadraticModel(qpdat)
end

function select_netlib()
    cases = filter(x -> endswith(x, ".SIF"), readdir(NETLIB_PATH))
    selected = String[]
    for case in cases
        try
            qp = load_instance(case)
            # Select only medium-sized instances
            if NLPModels.get_nnzj(qp) <= NNZJ_THRESHOLD
                push!(selected, case)
            end
        catch ex
            println("Fail to load $(case)")
        end
    end
    return selected
end

function benchmark_scalability(cases, batches)

    m = 5 + length(batches)
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
            tol=1e-6,
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
        qps = build_qps(qp, batches[end]; shift_c=false)
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:batch])
            gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
            gpu_solver = MadIPM.UniformBatchMPCSolver(
                gpu_bnlp;
                print_level=MadNLP.ERROR,
                tol=1e-6,
                max_iter=500,
                regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
                uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_algorithm = MadNLP.LDL,
                cudss_pivot_epsilon=1e-8,
            )
            stats = MadIPM.solve!(gpu_solver)
            results[k, 5+l] = sum(stats.total_time) / batch
        end
    end

    return [cases results]
end

function main()
    @info "Warmup"
    warmup(WARMUP_INSTANCE)
    # 1 -> 4096
    batches = [2^i for i in 0:12]
    cases = select_netlib()
    @info "#instances: $(length(cases))"
    results = benchmark_scalability(cases, batches)
    writedlm(joinpath("results", "1-scalability-netlib.csv"), results)
end

main()

