# MadIPM.jl

MadIPM.jl is a GPU-accelerated optimization solver for linear and quadratic programming.
The solver implements the Mehrotra predictor-corrector method in pure Julia,
and supports the solution of large-scale linear programs on the GPU using NVIDIA cuDSS.

## Installation

The package is currently not registered, but you can install it using:

```julia

julia> ] add https://github.com/MadNLP/MadIPM.jl

```

## Basic usage

### JuMP

MadIPM supports the JuMP ecosystem.
For instance, you can solve any LP formulated with JuMP by using:

```julia
using JuMP
using MadIPM

c = rand(10)
model = Model(MadIPM.Optimizer)
@variable(model, 0 <= x[1:10], start=0.5)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, c' * x)
JuMP.optimize!(model)
```

### QuadraticModels

We detail here how to solve a LP stored in a MPS file `mylp.mps` using [QPSReader](https://github.com/JuliaSmoothOptimizers/QPSReader.jl) and [QuadraticModels](https://github.com/JuliaSmoothOptimizers/QuadraticModels.jl).

```julia

using QPSReader
using QuadraticModels
using MadIPM

qpdat = readqps("mylp.mps")
qp = QuadraticModel(qpdat)

solver = MadIPM.MPCSolver(qp)
results = MadIPM.solve!(solver)

```

### Custom usage

MadIPM takes as input any linear program (LP) or quadratic program (QP) represented as an `AbstractNLPModel`,
following the specification in [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/).

For any `qp <: AbstractNLPModel`, you can pass it to MadIPM using:

```julia
solver = MadIPM.MPCSolver(qp)
results = MadIPM.solve!(solver)

```

## Solving a LP with CUDA

MadIPM supports GPU acceleration using NVIDIA cuDSS.
It requires specifying your problem in a `QuadraticProblem` first.

The data are moved to the GPU using:
```julia
using CUDA, KernelAbstractions, MadNLPGPU
using MadIPM

qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp)

```
Then, you can pass the structure `qp_gpu` to MadIPM by switching
the linear solver to NVIDIA cuDSS:
```julia
solver = MadIPM.MPCSolver(
    qp_gpu;
    linear_solver=MadNLPGPU.CUDSSSolver,
)
results = MadIPM.solve!(solver)

```
As a result, all the solution happens on the GPU, with minimum data transfer
between the host and the device.

