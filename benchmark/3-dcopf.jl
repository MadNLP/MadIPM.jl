
include("common.jl")

using JuMP
using PowerModels

PowerModels.silence()

const WARMUP_INSTANCE = "case300.m"
const MATPOWER_DATA = "/home/fpacaud/dev/matpower/data"

function select_dcopf_instances()
    return [
        "case118.m",
        "case1354pegase.m",
        "case13659pegase.m",
        "case1888rte.m",
        "case1951rte.m",
        "case2848rte.m",
        "case2868rte.m",
        "case2869pegase.m",
        "case6468rte.m",
        "case6470rte.m",
        "case6495rte.m",
        "case6515rte.m",
        "case8387pegase.m",
        "case89pegase.m",
        "case9241pegase.m",
        "case_ACTIVSg10k.m",
        "case_ACTIVSg2000.m",
        "case_ACTIVSg200.m",
        "case_ACTIVSg25k.m",
        "case_ACTIVSg500.m",
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
    # more easilly once converted to QuadraticModels format
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
        1e8 * sum(sigmap[i] + sigman[i] for i in keys(ref[:bus]))
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

function build_dcopf_qps(base_qp, nbus, batch_size; tau=0.2)
    # NB: do not apply presolve here

    n = base_qp.meta.nvar
    m = base_qp.meta.ncon

    lvar0 = copy(base_qp.meta.lvar)
    uvar0 = copy(base_qp.meta.uvar)

    return [begin
        rng = Xoshiro(i)

        sigma = (1-tau) .+ (2tau) .* rand(rng, nbus)
        lvar_new = copy(lvar0)
        uvar_new = copy(uvar0)

        lvar_new[n-nbus+1:n] .*= sigma
        uvar_new[n-nbus+1:n] .*= sigma

        QuadraticModel(
            base_qp.data.c,
            base_qp.data.H;
            A = base_qp.data.A,
            lcon = copy(base_qp.meta.lcon),
            ucon = copy(base_qp.meta.ucon),
            lvar = lvar_new,
            uvar = uvar_new,
            x0 = copy(base_qp.meta.x0),
            c0 = base_qp.data.c0,
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

function benchmark_dcopf(cases, batches)
    m = 5 + length(batches)
    results = zeros(length(cases), m)

    for (k, case) in enumerate(cases)
        @info case
        refresh_memory()
        # Load instance
        qp, nbus = load_instance(case)
        results[k, 1] = NLPModels.get_nvar(qp)
        results[k, 2] = NLPModels.get_ncon(qp)
        results[k, 3] = NLPModels.get_nnzj(qp)
        # Launch on CPU
        cpu_solver = MadIPM.MPCSolver(
            qp;
            print_level=MadNLP.INFO,
            max_iter=500,
            tol=1e-6,
            regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
            linear_solver=Ma57Solver,
        )
        stats = MadIPM.solve!(cpu_solver)
        results[k, 4] = stats.iter
        results[k, 5] = stats.counters.total_time
        # Launch on GPU (batch)
        qps = build_dcopf_qps(qp, nbus, batches[end])
        for (l, batch) in enumerate(batches)
            # Test pure scalability, do not change cost vector here.
            # try
                cpu_bnlp = ObjRHSBatchQuadraticModel(qps[1:batch])
                gpu_bnlp = convert(ObjRHSBatchQuadraticModel{Float64, CuVector{Float64}}, cpu_bnlp)
                gpu_solver = MadIPM.UniformBatchMPCSolver(
                    gpu_bnlp;
                    print_level=MadNLP.INFO,
                    tol=1e-6,
                    max_iter=500,
                    regularization = MadIPM.FixedRegularization(1e-10, -1e-10),
                    uniformbatch_linear_solver = MadNLPGPU.CUDSSSolver,
                    cudss_algorithm = MadNLP.LDL,
                    cudss_pivot_epsilon=1e-8,
                    scaling=false,
                    # rethrow_error=true,
                )
                stats = MadIPM.solve!(gpu_solver)
                println(stats.total_time)
                results[k, 5+l] = sum(stats.total_time) / batch
            # catch ex
            #     println("Failure for $(case): $(ex)")
            #     results[k, 5+l] = -1
            # end
        end
    end

    return [cases results]
end

function main()
    @info "Warmup"
    warmup(WARMUP_INSTANCE)

    batches = [2^i for i in 0:4]
    cases = select_dcopf_instances()[2:2]
    @info "#instances: $(length(cases))"
    results = benchmark_dcopf(cases, batches)
    writedlm(joinpath("results", "3-benchmark-dcopf.csv"), results)
end

main()

