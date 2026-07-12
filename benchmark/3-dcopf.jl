
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

function warmup(instance)
    qp, _ = load_instance(instance)
    _warmup(qp)
    return
end

function analyze_instance(case, batches; tau=0.0, options...)
    results = zeros(length(batches) + 1, 6)
    # Load instance
    base_qp, nbus = load_instance(case)
    n = NLPModels.get_nvar(base_qp)
    index = (n-nbus+1:n)

    qp = MadIPM.standard_form_qp(scale_qp(base_qp))
    # Load LPs in host memory
    qps = build_dcopf_qps(qp, index, batches[end]; tau=tau)

    # Time on the CPU
    t_init_cpu = @elapsed begin
        cpu_solver = MadIPM.MPCSolver(
            qp;
            linear_solver=Ma57Solver,
            options...
        )
    end
    stats = MadIPM.solve!(cpu_solver)
    t_factorization_cpu = @elapsed begin
        MadIPM.factorize_system!(cpu_solver)
    end
    t_iter_cpu = @elapsed begin
        MadIPM.mpc_step!(cpu_solver)
    end

    results[1, 1] = 1
    results[1, 2] = stats.iter
    results[1, 3] = stats.counters.total_time
    results[1, 4] = t_init_cpu
    results[1, 5] = t_iter_cpu
    results[1, 6] = t_factorization_cpu

    for (k, nb) in enumerate(batches)
        refresh_memory()

        # Time on the GPU
        cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:nb])
        gpu_bnlp = to_gpu(cpu_bnlp)
        t_init_gpu = CUDA.@elapsed begin
            gpu_solver = MadIPM.UniformBatchMPCSolver(
                gpu_bnlp;
                uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                cudss_algorithm = MadNLP.LDL,
                options...
            )
        end
        # Solve problem
        stats = MadIPM.solve!(gpu_solver)
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
        results[k+1, 3] = sum(stats.total_time) / nb  # average
        results[k+1, 4] = t_init_gpu
        results[k+1, 5] = t_iter_gpu
        results[k+1, 6] = t_factorization_gpu
    end
    return results
end

function solve_batch_dcopf(cases, nbatch, tau; options...)
    results = zeros(length(cases), 8)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        base_qp, nbus = load_instance(case)
        n = NLPModels.get_nvar(base_qp)
        index = (n-nbus+1:n)

        qp = MadIPM.standard_form_qp(scale_qp(base_qp))
        # Load LPs in host memory
        qps = build_dcopf_qps(qp, index, nbatch; tau=tau)

        # CPU
        stats_cpu = madipm.(qps; linear_solver=Ma57Solver, options...)

        status = [s.status for s in stats_cpu]
        iters = [s.iter for s in stats_cpu]
        has_converged = findall(isequal(MadNLP.SOLVE_SUCCEEDED), status)
        results[k, 1] = length(has_converged)
        results[k, 2] = mean(iters[has_converged])
        results[k, 3] = std(iters[has_converged])
        results[k, 4] = sum([s.counters.total_time for s in stats_cpu[has_converged]])

        # GPU
        cpu_bnlp = ObjRHSBatchQuadraticModel(qps)
        gpu_bnlp = to_gpu(cpu_bnlp)
        gpu_solver = MadIPM.UniformBatchMPCSolver(
            gpu_bnlp;
            uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
            cudss_algorithm = MadNLP.LDL,
            options...
        )
        # Solve problem
        stats_gpu = MadIPM.solve!(gpu_solver)
        has_converged = findall(isequal(MadNLP.SOLVE_SUCCEEDED), stats_gpu.status)
        results[k, 5] = length(has_converged)
        results[k, 6] = mean(stats_gpu.iter[has_converged])
        results[k, 7] = std(stats_gpu.iter[has_converged])
        results[k, 8] = sum(stats_gpu.total_time[has_converged]) / nbatch
    end

    return [cases results]
end

function benchmark_dcopf(cases, batches; tau=0.1)
    m = 5 + 2*length(batches)
    shift1 = 5
    shift2 = shift1 + length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        # Load instance
        base_qp, nbus = load_instance(case)
        n = NLPModels.get_nvar(base_qp)
        index = (n-nbus+1:n)

        qp = MadIPM.standard_form_qp(scale_qp(base_qp))

        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)

        # Launch on CPU
        cpu_solver = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.INFO,
            max_iter=300,
            tol=1e-6,
            regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
            linear_solver=Ma57Solver,
            scaling=false,
        )
        stats = MadIPM.solve!(cpu_solver)
        results[k, 4] = stats.iter
        results[k, 5] = stats.counters.total_time
        # Launch on GPU (batch)
        qps = build_dcopf_qps(qp, index, batches[end]; tau=tau)
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            try
                cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:batch])
                gpu_bnlp = to_gpu(cpu_bnlp)
                gpu_solver = MadIPM.UniformBatchMPCSolver(
                    gpu_bnlp;
                    print_level=MadNLP.INFO,
                    tol=1e-6,
                    max_iter=300,
                    regularization = MadIPM.FixedRegularization(1e-8, -1e-8),
                    uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                    cudss_algorithm = MadNLP.LDL,
                    scaling=false,
                )
                stats = MadIPM.solve!(gpu_solver)
                println(stats.total_time)
                results[k, shift1+l] = sum(stats.iter) / batch
                results[k, shift2+l] = sum(stats.total_time) / batch
            catch ex
                println("Failure for $(case): $(ex)")
                results[k, shift1+l] = -1
                results[k, shift2+l] = -1
            end
        end
    end

    return [cases results]
end

function decompose_timings()
    warmup(WARMUP_INSTANCE)

    for (case, batches) in [
        ("case89pegase.m", [2^i for i in 0:12]),
        ("case1354pegase.m", [2^i for i in 0:10]),
        ("case_ACTIVSg2000.m", [2^i for i in 0:10]),
        ("case6515rte.m", [2^i for i in 0:8]),
        ("case_ACTIVSg10k.m", [2^i for i in 0:8]),
    ]
        @info case
        results = analyze_instance(
            case,
            batches;
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
    for arg in args
        if startswith(arg, "--tol=")
            tol = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--max-batch=")
            max_batch = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--device=")
            device = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--job=")
            benchmark = Symbol(split(arg, "=")[2])
        end
    end
    return (
        max_batch=max_batch,
        tol=tol,
        job=job,
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

    cases = select_dcopf_instances()

    if pargs.job == :benchmark
        batches = [2^i for i in 0:pargs.max_batch]
        @info "#instances: $(length(cases))"
        results = benchmark_dcopf(cases, batches)
        mkpath("results")
        writedlm(joinpath("results", "3-benchmark-dcopf.csv"), results)
    elseif pargs.job == :decompose
        decompose_timings()
    elseif pargs.job == :comp
        nbatch = 64
        for tau ∈ [0.0, 0.1, 0.2, 0.3, 0.4]
            @info "\n$(tau)"
            results = solve_batch_dcopf(
                cases,
                nbatch,
                tau;
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

main()

