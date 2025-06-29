using LinearAlgebra
using MadNLP
using MadIPM

include("common.jl")
include("cuda_wrapper.jl")
include("qp_gpu.jl")

src = fetch_netlib()
sif_files = filter(x -> endswith(x, ".SIF"), readdir(src))
num_sif_problems = length(sif_files)

sif = "WOODW.SIF"
path_sif = joinpath(src, sif)

# Read the SIF file
qpdat = import_mps(path_sif)

# Instantiate QuadraticModel
qp = QuadraticModel(qpdat)
new_qp, flag = presolve_qp(qp)
scaled_qp = scale_qp(new_qp)
standard_qp = standard_form_qp(scaled_qp)

# Transfer data to the GPU
qp_gpu = transfer_to_gpu(standard_qp)

for (kkt, algo) in ((MadNLP.ScaledSparseKKTSystem, MadNLP.LDL     ),
                    (MadNLP.SparseKKTSystem      , MadNLP.LDL     ),
                    (MadIPM.NormalKKTSystem      , MadNLP.CHOLESKY))

    solver = MadIPM.MPCSolver(
        qp_gpu;
        max_iter=100,
        tol=1e-7,
        kkt_system=kkt,
        linear_solver=MadNLPGPU.CUDSSSolver,
        cudss_algorithm=algo,
        print_level=MadNLP.INFO,
        scaling=true,
        max_ncorr=0,
        step_rule=MadIPM.AdaptiveStep(0.995),
        regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
        rethrow_error=true,
    )

    MadIPM.solve!(solver)
end
