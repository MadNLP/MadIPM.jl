
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
    _warmup(qp; linear_solver=Ma57Solver)
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

# Columns: nvar, ncon, nnzj of the loaded instance; then, per batch size b,
#   CPU on instances 1:b: converged count, mean iter, summed init time, summed
#       solve time (one CPU, sequentially), max init time, max solve time
#       (b CPUs with perfect scaling);
#   GPU solving the same instances 1:b as one batch: converged count, mean iter,
#       init time, solve time.
# Both sides solve the same presolved, scaled, standard-form instances; all
# times are wall clock.
function benchmark_scalability(cases, batches; options...)
    shift = 3
    m = shift + 10*length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        qp = load_instance(case)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        # Test pure scalability, do not change cost vector here.
        qps = build_qps(qp, batches[end]; shift_c=false)
        # CPU: every instance of the largest batch, sequentially
        cpu = timed_cpu_sequential(qps; linear_solver=Ma57Solver, options...)
        for (l, batch) in enumerate(batches)
            results[k, shift+10*(l-1) .+ (1:6)] .= cpu_summary(cpu..., batch)
            # GPU: the same instances as one batch
            refresh_memory()
            gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps[1:batch]))
            _, stats, t_init, t_solve = timed_gpu_solve(gpu_bnlp; cudss_pivot_epsilon=1e-8, options...)
            results[k, shift+10*(l-1) .+ (7:10)] .= gpu_summary(stats, t_init, t_solve)
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


