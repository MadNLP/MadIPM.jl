
include("common.jl")

using JuMP
using PowerModels
using Statistics

PowerModels.silence()

const WARMUP_INSTANCE = "case300.m"
const MATPOWER_DATA = get(ENV, "MATPOWER_DATA", joinpath(@__DIR__, "matpower", "data"))

function select_dcopf_instances()
    return [
        "case89pegase.m",
        "case118.m",
        "case_ACTIVSg200.m",
        "case_ACTIVSg500.m",
        "case1354pegase.m",
        "case1888rte.m",
        "case1951rte.m",
        "case_ACTIVSg2000.m",
        "case2848rte.m",
        "case2868rte.m",
        "case2869pegase.m",
        "case6468rte.m",
        "case6470rte.m",
        "case6495rte.m",
        "case6515rte.m",
        "case9241pegase.m",
        "case_ACTIVSg10k.m",
        "case13659pegase.m",
        "case_ACTIVSg25k.m",
    ]
end

function dcopf_model(data)
    # Add zeros to turn linear objective functions into quadratic ones
    # so that additional parameter checks are not required
    PowerModels.standardize_cost_terms!(data, order=2)
    # Adds reasonable rate_a values to branches without them
    PowerModels.calc_thermal_limits!(data)

    # use build_ref to filter out inactive components
    ref = PowerModels.build_ref(data)[:it][:pm][:nw][0]
    # Note: ref contains all the relevant system parameters needed to build the OPF model
    # When we introduce constraints and variable bounds below, we use the parameters in ref.

    # Collect loads
    busid = [i for (i, bus) in ref[:bus]]
    nbus = length(busid)
    loads = zeros(nbus)
    k = 1
    for (i, bus) in ref[:bus]
        bus_loads = [ref[:load][l] for l in ref[:bus_loads][i]]
        loads[k] = sum(load["pd"] for load in bus_loads; init=0.0)
        k += 1
    end


    ###############################################################################
    # 1. Building the Optimal Power Flow Model
    ###############################################################################
    # Initialize a JuMP Optimization Model
    #-------------------------------------
    model = Model()

    # Add voltage angles va for each bus
    @variable(model, va[i in keys(ref[:bus])])
    # note: [i in keys(ref[:bus])] adds one `va` variable for each bus in the network
    # Add active power generation variable pg for each generator (including limits)
    @variable(model, ref[:gen][i]["pmin"] <= pg[i in keys(ref[:gen])] <= ref[:gen][i]["pmax"])
    # Add power flow variables p to represent the active power flow for each branch
    @variable(model, -ref[:branch][l]["rate_a"] <= p[(l,i,j) in ref[:arcs_from]] <= ref[:branch][l]["rate_a"])

    # Build JuMP expressions for the value of p[(l,i,j)] and p[(l,j,i)] on the branches
    p_expr = Dict([((l,i,j), 1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]])
    p_expr = merge(p_expr, Dict([((l,j,i), -1.0*p[(l,i,j)]) for (l,i,j) in ref[:arcs_from]]))
    # note: this is used to make the definition of nodal power balance simpler

    # Add power flow variables p_dc to represent the active power flow for each HVDC line
    @variable(model, p_dc[a in ref[:arcs_dc]])
    @variable(model, sigmap[i in keys(ref[:bus])] >= 0)
    @variable(model, sigman[i in keys(ref[:bus])] >= 0)

    # Encore loads as fixed variables
    # N.B.: keep the load variables at the end to find the corresponding indexes
    # more easilly once converted to MadIPM.Models format
    @variable(model, _loads[i=1:nbus] == loads[i])

    for (l,dcline) in ref[:dcline]
        f_idx = (l, dcline["f_bus"], dcline["t_bus"])
        t_idx = (l, dcline["t_bus"], dcline["f_bus"])

        JuMP.set_lower_bound(p_dc[f_idx], dcline["pminf"])
        JuMP.set_upper_bound(p_dc[f_idx], dcline["pmaxf"])

        JuMP.set_lower_bound(p_dc[t_idx], dcline["pmint"])
        JuMP.set_upper_bound(p_dc[t_idx], dcline["pmaxt"])
    end


    # index representing which side the HVDC line is starting
    from_idx = Dict(arc[1] => arc for arc in ref[:arcs_from_dc])

    # Minimize the cost of active power generation and cost of HVDC line usage
    # assumes costs are given as *linear* functions
    @objective(model, Min,
        sum(gen["cost"][2]*pg[i] + gen["cost"][3] for (i,gen) in ref[:gen]) +
        sum(dcline["cost"][2]*p_dc[from_idx[i]] + dcline["cost"][3] for (i,dcline) in ref[:dcline]) +
        1e6 * sum(sigmap[i] + sigman[i] for i in keys(ref[:bus]))
    )

    # Fix the voltage angle to zero at the reference bus
    for (i,bus) in ref[:ref_buses]
        @constraint(model, va[i] == 0)
    end


    # Nodal power balance constraints
    k = 1
    for (i,bus) in ref[:bus]
        # Build a list of the loads and shunt elements connected to the bus i
        bus_shunts = [ref[:shunt][s] for s in ref[:bus_shunts][i]]

        # Active power balance at node i
        @constraint(model,
            sum(p_expr[a] for a in ref[:bus_arcs][i]) +                  # sum of active power flow on lines from bus i +
            sum(p_dc[a_dc] for a_dc in ref[:bus_arcs_dc][i]) ==     # sum of active power flow on HVDC lines from bus i =
            sum(pg[g] for g in ref[:bus_gens][i]) -                 # sum of active power generation at bus i -
            _loads[k] -                                               # sum of active load consumption at bus i -
            sum(shunt["gs"] for shunt in bus_shunts)*1.0^2          # sum of active shunt element injections at bus i
            + sigmap[i] - sigman[i]
        )
        k += 1
    end

    # Branch power flow physics and limit constraints
    for (i,branch) in ref[:branch]
        # Build the from variable id of the i-th branch, which is a tuple given by (branch id, from bus, to bus)
        f_idx = (i, branch["f_bus"], branch["t_bus"])

        p_fr = p[f_idx]                     # p_fr is a reference to the optimization variable p[f_idx]

        va_fr = va[branch["f_bus"]]         # va_fr is a reference to the optimization variable va on the from side of the branch
        va_to = va[branch["t_bus"]]         # va_fr is a reference to the optimization variable va on the to side of the branch

        # Compute the branch parameters and transformer ratios from the data
        g, b = PowerModels.calc_branch_y(branch)

        # DC Power Flow Constraint
        @constraint(model, p_fr == -b*(va_fr - va_to))
        # note: that upper and lower limits on the power flow (i.e. p_fr) are not included here.
        #   these limits were already enforced for p (which is the same as p_fr) when
        #   the optimization variable p was defined (around line 65).
    end

    # HVDC line constraints
    for (i,dcline) in ref[:dcline]
        # Build the from variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, from bus, to bus)
        f_idx = (i, dcline["f_bus"], dcline["t_bus"])
        # Build the to variable id of the i-th HVDC line, which is a tuple given by (hvdc line id, to bus, from bus)
        t_idx = (i, dcline["t_bus"], dcline["f_bus"])   # index of the ith HVDC line which is a tuple given by (line number, to bus, from bus)
        # note: it is necessary to distinguish between the from and to sides of a HVDC line due to power losses

        # Constraint defining the power flow and losses over the HVDC line
        @constraint(model, (1-dcline["loss1"])*p_dc[f_idx] + (p_dc[t_idx] - dcline["loss0"]) == 0)
    end
    return model
end

function build_dcopf_qps(base_qp, index, batch_size; tau=0.2)
    # NB: do not apply presolve here

    lvar0 = copy(base_qp.meta.lvar)
    uvar0 = copy(base_qp.meta.uvar)

    return [begin
        rng = Xoshiro(i)

        sigma = (1-tau) .+ (2tau) .* rand(rng, length(index))
        lvar_new = copy(lvar0)
        uvar_new = copy(uvar0)

        lvar_new[index] .*= sigma
        uvar_new[index] .*= sigma

        _shared_matrix_qp(
            base_qp, copy(base_qp.data.c),
            copy(base_qp.meta.lcon), copy(base_qp.meta.ucon),
            lvar_new, uvar_new,
        )
    end for i in 1:batch_size]
end

function load_instance(case)
    data = PowerModels.parse_file(joinpath(MATPOWER_DATA, case))
    model = dcopf_model(data)
    nbus = length(model[:va])
    opt = MadIPM.Optimizer()
    MOI.copy_to(opt, model)
    return opt.qp, nbus
end

function warmup(instance; cpu_solver)
    qp, _ = load_instance(instance)
    _warmup(qp; linear_solver=cpu_solver)
    return
end

function analyze_instance(case, batches; cpu_solver, tau=0.0, options...)
    results = zeros(length(batches) + 1, 6)
    # Load instance
    base_qp, nbus = load_instance(case)
    n = NLPModels.get_nvar(base_qp)
    index = (n-nbus+1:n)

    qp = MadIPM.standard_form_qp(scale_qp(base_qp))
    # Load LPs in host memory
    qps = build_dcopf_qps(qp, index, batches[end]; tau=tau)

    # Columns: batch size, mean iter, solve time, init time, one-step time,
    # one-factorization time. Row 1 is the CPU on the first instance of the
    # batch; the step and factorization are timed from a fresh initial point
    # on both sides.
    cpu_solver, stats, t_init_cpu, t_solve_cpu =
        timed_cpu_solve(qps[1]; linear_solver=cpu_solver, options...)
    MadIPM.initialize!(cpu_solver)
    t_factorization_cpu = @elapsed begin
        MadIPM.factorize_system!(cpu_solver)
    end
    t_iter_cpu = @elapsed begin
        MadIPM.mpc_step!(cpu_solver)
    end

    results[1, 1] = 1
    results[1, 2] = stats.iter
    results[1, 3] = t_solve_cpu
    results[1, 4] = t_init_cpu
    results[1, 5] = t_iter_cpu
    results[1, 6] = t_factorization_cpu

    for (k, nb) in enumerate(batches)
        refresh_memory()

        # Time on the GPU
        cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:nb])
        gpu_bnlp = to_gpu(cpu_bnlp)
        gpu_solver, stats, t_init_gpu, t_solve_gpu = timed_gpu_solve(gpu_bnlp; options...)
        # Time individual operations
        # (need to reinitialize structure first to avoid masking side-effect)
        MadIPM.initialize!(gpu_solver)
        t_factorization_gpu = CUDA.@elapsed begin
            MadIPM.factorize_system!(gpu_solver)
        end
        t_iter_gpu = CUDA.@elapsed begin
            MadIPM.mpc_step!(gpu_solver)
        end

        results[k+1, 1] = nb
        results[k+1, 2] = sum(stats.iter) / nb        # average
        results[k+1, 3] = t_solve_gpu                 # whole batch
        results[k+1, 4] = t_init_gpu
        results[k+1, 5] = t_iter_gpu
        results[k+1, 6] = t_factorization_gpu
    end
    return results
end

function solve_batch_dcopf(cases, nbatch, tau; cpu_solver, options...)
    results = zeros(length(cases), 8)

    for (k, case) in enumerate(cases)
        progress(case)
        refresh_memory()
        base_qp, nbus = load_instance(case)
        n = NLPModels.get_nvar(base_qp)
        index = (n-nbus+1:n)

        qp = MadIPM.standard_form_qp(scale_qp(base_qp))
        # Load LPs in host memory
        qps = build_dcopf_qps(qp, index, nbatch; tau=tau)

        # Columns: CPU converged count, mean iter, std iter, solve time of all
        # instances one after the other; then the same for the GPU batch. Both
        # times cover every instance, converged or not: an instance that runs
        # to the iteration limit costs its iterations on either side.
        # CPU
        cpu_runs = [timed_cpu_solve(qp_i; linear_solver=cpu_solver, options...) for qp_i in qps]
        stats_cpu = [r[2] for r in cpu_runs]

        status = [s.status for s in stats_cpu]
        iters = [s.iter for s in stats_cpu]
        has_converged = findall(isequal(MadNLP.SOLVE_SUCCEEDED), status)
        results[k, 1] = length(has_converged)
        results[k, 2] = mean(iters[has_converged])
        results[k, 3] = std(iters[has_converged])
        results[k, 4] = sum(r[4] for r in cpu_runs)

        # GPU
        cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
        gpu_bnlp = to_gpu(cpu_bnlp)
        _, stats_gpu, _, t_solve_gpu = timed_gpu_solve(gpu_bnlp; options...)
        has_converged = findall(isequal(MadNLP.SOLVE_SUCCEEDED), stats_gpu.status)
        results[k, 5] = length(has_converged)
        results[k, 6] = mean(stats_gpu.iter[has_converged])
        results[k, 7] = std(stats_gpu.iter[has_converged])
        results[k, 8] = t_solve_gpu
    end

    return [cases results]
end

# Columns: nvar, ncon, nnzj of the standard-form problem; then, per batch size b,
#   CPU on instances 1:b: converged count, mean iter, summed init time, summed
#       solve time (one CPU, sequentially), max init time, max solve time
#       (b CPUs with perfect scaling);
#   GPU solving the same instances 1:b as one batch: converged count, mean iter,
#       init time, solve time.
# Both sides solve the same perturbed instances; all times are wall clock. The
# CPU solves at most `cpu_max_batch` instances and stops early once
# `cpu_time_budget` seconds are used up; CPU blocks of larger batches are -1
# (not measured, never extrapolated).
function benchmark_dcopf(cases, batches; cpu_solver, cpu_max_batch=batches[end],
                         cpu_time_budget=Inf, tau=0.1)
    shift = 3
    m = shift + 10*length(batches)
    results = zeros(length(cases), m)

    options = (
        print_level=MadNLP.INFO,
        max_iter=300,
        tol=1e-6,
        regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
        scaling=false,
    )

    for (k, case) in enumerate(cases)
        progress(case)
        refresh_memory()
        # Load instance
        base_qp, nbus = load_instance(case)
        n = NLPModels.get_nvar(base_qp)
        index = (n-nbus+1:n)

        qp = MadIPM.standard_form_qp(scale_qp(base_qp))

        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)

        qps = build_dcopf_qps(qp, index, batches[end]; tau=tau)
        # CPU: the first instances of the largest batch, sequentially
        cpu = timed_cpu_sequential(qps[1:min(length(qps), cpu_max_batch)];
            linear_solver=cpu_solver, time_budget=cpu_time_budget, options...)
        n_cpu = length(cpu[1])
        for (l, batch) in enumerate(batches)
            cpu_cols = shift+10*(l-1) .+ (1:6)
            gpu_cols = shift+10*(l-1) .+ (7:10)
            results[k, cpu_cols] .= batch <= n_cpu ? cpu_summary(cpu..., batch) : -1
            # GPU: the same instances as one batch
            try
                refresh_memory()
                gpu_bnlp = to_gpu(ObjRHSBatchQuadraticModel(qps[1:batch]))
                _, stats, t_init, t_solve = timed_gpu_solve(gpu_bnlp; options...)
                results[k, gpu_cols] .= gpu_summary(stats, t_init, t_solve)
            catch ex
                println("Failure for $(case): $(ex)")
                results[k, gpu_cols] .= -1
            end
        end
    end

    return [cases results]
end

function decompose_timings(; cpu_solver)
    warmup(WARMUP_INSTANCE; cpu_solver=cpu_solver)

    for (case, batches) in [
        ("case89pegase.m", [2^i for i in 0:12]),
        ("case1354pegase.m", [2^i for i in 0:10]),
        ("case_ACTIVSg2000.m", [2^i for i in 0:10]),
        ("case6515rte.m", [2^i for i in 0:8]),
        ("case_ACTIVSg10k.m", [2^i for i in 0:8]),
    ]
        progress(case)
        results = analyze_instance(
            case,
            batches;
            cpu_solver=cpu_solver,
            tol=1e-6,
            regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
            max_iter=300,
            scaling=false,
        )
        mkpath("results")
        writedlm(joinpath("results", "3-decompose-dcopf-$(case).csv"), results)
    end
end

function parse_args(args::Vector{String})
    # Default options
    max_batch = 12
    device = nothing
    tol = 1e-6
    job = :comp
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
        elseif startswith(arg, "--job=")
            job = Symbol(split(arg, "=")[2])
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
        job=job,
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

    cases = select_dcopf_instances()

    if pargs.job == :benchmark
        batches = [2^i for i in 0:pargs.max_batch]
        progress("#instances: $(length(cases))")
        results = benchmark_dcopf(cases, batches; cpu_solver=cpu_solver,
            cpu_max_batch=2^pargs.cpu_max_batch, cpu_time_budget=pargs.cpu_time_budget)
        mkpath("results")
        writedlm(joinpath("results", "3-benchmark-dcopf.csv"), results)
    elseif pargs.job == :decompose
        decompose_timings(; cpu_solver=cpu_solver)
    elseif pargs.job == :comp
        nbatch = 64
        for tau ∈ [0.0, 0.1, 0.2, 0.3, 0.4]
            @info "\n$(tau)"
            results = solve_batch_dcopf(
                cases,
                nbatch,
                tau;
                cpu_solver=cpu_solver,
                tol=pargs.tol,
                regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
                max_iter=300,
                scaling=false,
                print_level=MadNLP.ERROR,
            )
            mkpath("results")
            writedlm(joinpath("results", "3-benchmark-batch-$(tau).csv"), results)
        end
    end
end

