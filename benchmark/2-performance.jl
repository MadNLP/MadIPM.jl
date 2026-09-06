
include("common.jl")

# MIPLIB (and its instance downloads) only for --benchmark=miplib
any(==("--benchmark=miplib"), ARGS) && @eval using MIPLIB


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
        "sp98ic.mps.gz",
        "tanglegram1.mps.gz",
        "tanglegram2.mps.gz",
        "timtab1.mps.gz",
    ]
end

@memoize function load_netlib_instance(case)
    qpdat = readqps(joinpath(NETLIB_PATH, case))
    return qps_model(qpdat)
end

@memoize function load_miplib_instance(case)
    return qps_model(MIPLIB.miplib2010_data(case))
end

# Columns: nvar, ncon, nnzj of the loaded instance; then, per batch size b,
#   CPU on instances 1:b: converged count, mean iter, summed init time, summed
#       solve time (one CPU, sequentially), max init time, max solve time
#       (b CPUs with perfect scaling);
#   GPU solving the same instances 1:b as one batch: converged count, mean iter,
#       init time, solve time.
# Both sides solve the same presolved, scaled, standard-form instances, cost
# perturbations included; all times are wall clock. The CPU solves at most
# `cpu_max_batch` instances and stops early once `cpu_time_budget` seconds are
# used up; CPU blocks of larger batches are -1 (not measured, never
# extrapolated). Failures are marked -1.
function benchmark_lps(cases, batches, load_instance; cpu_solver, cpu_max_batch=batches[end],
                       cpu_time_budget=Inf, options...)
    shift = 3
    m = shift + 10*length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        progress(case)
        refresh_memory()
        # Load instance
        qp = load_instance(case)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        qps = try
            build_qps(qp, batches[end])
        catch ex
            println("$(case) fails in presolve with message $(ex)")
            results[k, shift+1:end] .= -1
            continue
        end
        # CPU: the first instances of the largest batch, sequentially
        cpu = RUN_CPU ? try
            timed_cpu_sequential(qps[1:min(length(qps), cpu_max_batch)];
                linear_solver=cpu_solver, time_budget=cpu_time_budget, options...)
        catch ex
            println("$(case) fails on CPU with message $(ex)")
            nothing
        end : nothing
        n_cpu = cpu === nothing ? 0 : length(cpu[1])
        for (l, batch) in enumerate(batches)
            cpu_cols = shift+10*(l-1) .+ (1:6)
            gpu_cols = shift+10*(l-1) .+ (7:10)
            if cpu === nothing || batch > n_cpu
                results[k, cpu_cols] .= -1
            else
                results[k, cpu_cols] .= cpu_summary(cpu..., batch)
            end
            # GPU: the same instances as one batch
            if !RUN_GPU
                results[k, gpu_cols] .= -1
                continue
            end
            try
                refresh_memory()
                gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps[1:batch]))
                _, stats, t_init, t_solve = timed_gpu_solve(gpu_bnlp; cudss_pivot_epsilon=1e-8, options...)
                results[k, gpu_cols] .= gpu_summary(stats, t_init, t_solve)
            catch ex
                println("$(case) fails on GPU with message $(ex)")
                results[k, gpu_cols] .= -1
            end
        end
    end
    return [cases results]
end

function parse_args(args::Vector{String})
    # Default options
    max_batch = 7
    device = nothing
    tol = 1e-6
    benchmark = :netlib
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
        elseif startswith(arg, "--benchmark=")
            benchmark = Symbol(split(arg, "=")[2])
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
        benchmark=benchmark,
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
    cpu_solver = cpu_linear_solver(pargs.cpu_solver; preferred=Ma27Solver)
    BLAS.set_num_threads(pargs.blas_threads)
    progress("CPU linear solver: $(cpu_solver), BLAS threads: $(BLAS.get_num_threads()), " *
             "CPU max batch: $(2^pargs.cpu_max_batch), CPU time budget: $(pargs.cpu_time_budget)s")

    progress("Warmup")
    _warmup(load_netlib_instance(WARMUP_INSTANCE); linear_solver=cpu_solver)

    batches = [2^i for i in 0:pargs.max_batch]
    if pargs.benchmark == :netlib
        cases = select_netlib_instance()
        mkpath("results")
        results = benchmark_lps(
            cases,
            batches,
            load_netlib_instance;
            cpu_solver=cpu_solver,
            cpu_max_batch=2^pargs.cpu_max_batch,
            cpu_time_budget=pargs.cpu_time_budget,
            print_level=MadNLP.ERROR,
            tol=pargs.tol,
            max_iter=300,
            regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
        )
        writedlm(results_path("2-benchmark-netlib"), results)
    elseif pargs.benchmark == :miplib
        cases = select_miplib_instance()
        mkpath("results")
        results = benchmark_lps(
            cases,
            batches,
            load_miplib_instance;
            cpu_solver=cpu_solver,
            cpu_max_batch=2^pargs.cpu_max_batch,
            cpu_time_budget=pargs.cpu_time_budget,
            print_level=MadNLP.ERROR,
            tol=pargs.tol,
            max_iter=300,
            regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
        )
        writedlm(results_path("2-benchmark-miplib"), results)
    end
    return
end

