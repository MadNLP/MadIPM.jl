
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
    return [
        "30n20b8.mps.gz",
        "aflow40b.mps.gz",
        "ash608gpia-3col.mps.gz",
        "biella1.mps.gz",
        "binkar10_1.mps.gz",
        "bnatt350.mps.gz",
        "core2536-691.mps.gz",
        "cov1075.mps.gz",
        "eil33-2.mps.gz",
        "eilB101.mps.gz",
        "enlight13.mps.gz",
        "enlight14.mps.gz",
        "glass4.mps.gz",
        "gmu-35-40.mps.gz",
        "iis-100-0-cov.mps.gz",
        "iis-bupa-cov.mps.gz",
        "iis-pima-cov.mps.gz",
        "lectsched-4-obj.mps.gz",
        "m100n500k4r1.mps.gz",
        "macrophage.mps.gz",
        "map18.mps.gz",               # slow 4
        "map20.mps.gz",               # slow 4
        "mik-250-1-100-1.mps.gz",
        "mine-166-5.mps.gz",
        "mine-90-10.mps.gz",
        "n3div36.mps.gz",
        "neos-1109824.mps.gz",
        "neos13.mps.gz",
        "neos18.mps.gz",
        "neos-934278.mps.gz",
        "noswot.mps.gz",
        "pg5_34.mps.gz",
        "pw-myciel4.mps.gz",
        "qiu.mps.gz",
        "rail507.mps.gz",
        "reblock67.mps.gz",
        "rmatr100-p10.mps.gz",
        "rmatr100-p5.mps.gz",
        "rmine6.mps.gz",
        "rmine14.mps.gz",
        "sp98ic.mps.gz",
        "tanglegram1.mps.gz",
        "tanglegram2.mps.gz",
        "timtab1.mps.gz",
    ]
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
