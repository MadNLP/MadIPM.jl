
include("common.jl")

const NETLIB_PATH = fetch_netlib()
const NNZJ_THRESHOLD = 5_000
const WARMUP_INSTANCE = "ADLITTLE.SIF"

@memoize function load_instance(case)
    qpdat = readqps(joinpath(NETLIB_PATH, case))
    return qps_model(qpdat)
end

function warmup(instance; cpu_solver)
    qp = load_instance(instance)
    _warmup(qp; linear_solver=cpu_solver)
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
# times are wall clock. The CPU solves at most `cpu_max_batch` instances and
# stops early once `cpu_time_budget` seconds are used up; CPU blocks of larger
# batches are -1 (not measured, never extrapolated).
function benchmark_scalability(cases, batches; cpu_solver, cpu_max_batch=batches[end],
                               cpu_time_budget=Inf, options...)
    shift = 3
    m = shift + 10*length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        progress(case)
        refresh_memory()
        qp = load_instance(case)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        # Test pure scalability, do not change cost vector here.
        qps = build_qps(qp, batches[end]; shift_c=false)
        # CPU: the first instances of the largest batch, sequentially
        cpu = RUN_CPU ? timed_cpu_sequential(qps[1:min(length(qps), cpu_max_batch)];
            linear_solver=cpu_solver, time_budget=cpu_time_budget, options...) : nothing
        n_cpu = cpu === nothing ? 0 : length(cpu[1])
        for (l, batch) in enumerate(batches)
            cpu_cols = shift+10*(l-1) .+ (1:6)
            gpu_cols = shift+10*(l-1) .+ (7:10)
            results[k, cpu_cols] .= batch <= n_cpu ? cpu_summary(cpu..., batch) : -1
            # GPU: the same instances as one batch
            if RUN_GPU
                refresh_memory()
                gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps[1:batch]))
                _, stats, t_init, t_solve = timed_gpu_solve(gpu_bnlp; cudss_pivot_epsilon=1e-8, options...)
                results[k, gpu_cols] .= gpu_summary(stats, t_init, t_solve)
            else
                results[k, gpu_cols] .= -1
            end
        end
    end

    return [cases results]
end

function parse_args(args::Vector{String})
    # Default options
    max_batch = 12
    device = nothing
    tol = 1e-6
    cpu_solver = "auto"
    cpu_max_batch = nothing   # log2 of the largest batch the CPU solves sequentially
    cpu_time_budget = Inf     # seconds of sequential CPU solves per instance
    blas_threads = 1          # BLAS threads for the CPU baseline
    for arg in args
        if startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--max-batch=")
            max_batch = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--device=")
            device = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--cpu-solver=")
            cpu_solver = String(split(arg, "=")[2])
        elseif startswith(arg, "--cpu-max-batch=")
            cpu_max_batch = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--cpu-time-budget=")
            cpu_time_budget = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--blas-threads=")
            blas_threads = parse(Int, split(arg, "=")[2])
        end
    end
    return (
        max_batch=max_batch,
        tol=tol,
        device=device,
        cpu_solver=cpu_solver,
        cpu_max_batch=something(cpu_max_batch, max_batch),
        cpu_time_budget=cpu_time_budget,
        blas_threads=blas_threads,
    )
end

function @main(args::Vector{String})
    pargs = parse_args(args)

    # Set-up device
    if !isnothing(pargs.device)
        CUDA.device!(pargs.device)
    end
    cpu_solver = cpu_linear_solver(pargs.cpu_solver; preferred=Ma57Solver)
    BLAS.set_num_threads(pargs.blas_threads)
    progress("CPU linear solver: $(cpu_solver), BLAS threads: $(BLAS.get_num_threads()), " *
             "CPU max batch: $(2^pargs.cpu_max_batch), CPU time budget: $(pargs.cpu_time_budget)s")

    progress("Warmup")
    warmup(WARMUP_INSTANCE; cpu_solver=cpu_solver)

    batches = [2^i for i in 0:pargs.max_batch]
    cases = select_netlib()
    progress("#instances: $(length(cases))")
    results = benchmark_scalability(
        cases,
        batches;
        cpu_solver=cpu_solver,
        cpu_max_batch=2^pargs.cpu_max_batch,
        cpu_time_budget=pargs.cpu_time_budget,
        print_level=MadNLP.ERROR,
        tol=pargs.tol,
        max_iter=500,
        regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
    )
    mkpath("results")
    writedlm(results_path("1-scalability-netlib"), results)
end


