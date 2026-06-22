
include("common.jl")

function build_madipm_batch(qps; options...)
    return MadIPM.UniformBatchMPCSolver(
        qps;
        options...
    )
end

function build_madipm_cpu(qp; options...)
    return MadIPM.MPCSolver(
        qp;
        options...
    )
end

function select_netlib()
    netlib_path = fetch_netlib()
    cases = filter(x -> endswith(x, ".SIF"), readdir(netlib_path))
    selected = String[]
    for case in cases
        try
            qp = readqps(joinpath(netlib_path, case))
            # Select only medium-sized instances
            if NLPModels.get_nnzj(qp) <= 10_000
                push!(selected, case)
            end
        catch ex
            println("Fail to load $(case)")
        end
    end
    return selected
end

function warmup(qp)
    # Warmup CPU
    cpu_solver = build_madipm_cpu(
        qp;
        print_level=MadNLP.ERROR,
        max_iter=1,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
        linear_solver=Ma57Solver,
    )
    MadIPM.solve!(cpu_solver)

    # Warmup GPU
    qps = build_qps(qp, 2)
    gpu_solver = build_madipm_batch(
        qps;
        print_level=MadNLP.ERROR,
        max_iter=1,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
        uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
        cudss_algorithm = MadNLP.LDL,
    )
    stats = MadIPM.solve!(gpu_solver)
    return
end

function benchmark_scalability(cases, batches)

    m = 5 + length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        refresh_memory()
        # Launch on CPU
        qp = readqps(joinpath(fetch_netlib(), case))
        cpu_solver = build_madipm_cpu(
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
        results[k, 4] = stats.iterations
        results[k, 5] = stats.total_time
        # Launch on GPU (batch)
        qps = build_qps(qp, batches[end]; shift_c=false)
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            gpu_solver = build_madipm_batch(
                qps[1:batch];
                print_level=MadNLP.ERROR,
                max_iter=500,
                regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
                uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_algorithm = MadNLP.LDL,
                cudss_pivot_epsilon=1e-8,
            )
            stats = MadIPM.solve!(gpu_solver)
            results[k, 5+l] = stats.total_time
        end
    end

    return [cases results]
end

function main()
    batches = [2^i for i in 0:10]
    case = select_netlib()
    results = benchmark_scalability(cases)
    writedlm(joinpath("results", "1-scalability-netlib.txt"), results)
end

