
include("common.jl")

const NETLIB_PATH = fetch_netlib()
const NNZJ_THRESHOLD = 5_000
const WARMUP_INSTANCE = "ADLITTLE.SIF"

@memoize function load_instance(case)
    qpdat = readqps(joinpath(NETLIB_PATH, case))
    return qps_model(qpdat)
end

function warmup(instance)
    qp = load_instance(instance)
    _warmup(qp)
    return
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

function benchmark_scalability(cases, batches; options...)
    shift = 5
    m = shift + 4*length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        # Launch on CPU
        qp = load_instance(case)
        cpu_solver = MadIPM.MPCSolver(
            qp;
            linear_solver=Ma57Solver,
            options...
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
            gpu_bnlp = to_gpu(cpu_bnlp)
            gpu_solver = MadIPM.UniformBatchMPCSolver(
                gpu_bnlp;
                uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_algorithm = MadNLP.LDL,
                cudss_pivot_epsilon=1e-8,
                options...
            )
            stats = MadIPM.solve!(gpu_solver)
            has_converged = findall(isequal(MadNLP.SOLVE_SUCCEEDED), stats.status)

            results[k, shift+4*(l-1)+1] = length(has_converged)
            results[k, shift+4*(l-1)+2] = sum(stats.iter) / batch
            results[k, shift+4*(l-1)+3] = sum(gpu_solver.batch_cnt.init_time) / batch
            results[k, shift+4*(l-1)+4] = sum(stats.total_time) / batch
        end
    end

    return [cases results]
end

function parse_args(args::Vector{String})
    # Default options
    max_batch = 12
    device = nothing
    tol = 1e-6
    for arg in args
        if startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--max-batch=")
            max_batch = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--device=")
            device = parse(Int, split(arg, "=")[2])
        end
    end
    return (
        max_batch=max_batch,
        tol=tol,
        device=device,
    )
end

function @main(args::Vector{String})
    pargs = parse_args(args)

    # Set-up device
    if !isnothing(pargs.device)
        CUDA.device!(pargs.device)
    end

    @info "Warmup"
    warmup(WARMUP_INSTANCE)

    batches = [2^i for i in 0:pargs.max_batch]
    cases = select_netlib()
    @info "#instances: $(length(cases))"
    results = benchmark_scalability(
        cases,
        batches;
        print_level=MadNLP.ERROR,
        tol=pargs.tol,
        max_iter=500,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
    )
    mkpath("results")
    writedlm(joinpath("results", "1-scalability-netlib.csv"), results)
end


