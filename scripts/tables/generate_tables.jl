using DelimitedFiles
using DataFrames
using SolverBenchmark

collection = "benchmark-miplib"
results_cpu = "$collection-cpu.txt"
results_gpu = "$collection-gpu.txt"
verbose = false

results_dir = @__DIR__
tex_dir = joinpath(@__DIR__, "tex")
path_tex = joinpath(@__DIR__, "tex", "$collection.tex")

# Create the folder "tex" if needed
isdir(tex_dir) || mkdir(tex_dir)

df = DataFrame(
  instance=String[],
  nvar=Int[],
  ncon=Int[],
  nnzj=Int[],
  # iter_cpu=Int[],
  time_cpu=Float64[],
  # iter_gpu=Int[],
  time_gpu=Float64[],
  ratio_time=Float64[]
)

data_cpu = readdlm(results_cpu, '\t')
data_gpu = readdlm(results_gpu, '\t')
m_cpu, n_cpu = size(data_cpu)
m_gpu, n_gpu = size(data_gpu)
@assert n_cpu == 10
@assert n_gpu == 10
@assert m_cpu == m_gpu
for k = 1:m_cpu  # Number of instances
  instance = data_cpu[k,1]
  nvar = data_cpu[k,2]
  ncon = data_cpu[k,3]
  nnzj = data_cpu[k,4]
  nnzh = data_cpu[k,5]

  status_cpu = data_cpu[k,6]
  iter_cpu = data_cpu[k,7]
  objective_cpu = data_cpu[k,8]
  total_time_cpu = data_cpu[k,9]
  linear_solver_time_cpu = data_cpu[k,10]

  status_gpu = data_gpu[k,6]
  iter_gpu = data_gpu[k,7]
  objective_gpu = data_gpu[k,8]
  total_time_gpu = data_gpu[k,9]
  linear_solver_time_gpu = data_gpu[k,10]

  ratio_total_time = total_time_cpu / total_time_gpu
  ratio_linear_solver_time = linear_solver_time_cpu / linear_solver_time_gpu

  if ratio_total_time > 1
    verbose && println("ratio_total_time | ", instance, " | ", ratio_total_time)
  end

  if ratio_linear_solver_time > 1
    verbose && println("ratio_linear_solver_time | ", instance, " | ", ratio_linear_solver_time)
  end

  st_cpu = convert(Int, status_cpu)
  st_gpu = convert(Int, status_gpu)
  if (st_cpu == 1) && (st_gpu == 1)
    # results = (instance, nvar, ncon, nnzj, iter_cpu, linear_solver_time_cpu, iter_gpu, linear_solver_time_gpu, ratio_linear_solver_time)
    results = (instance, nvar, ncon, nnzj, total_time_cpu, total_time_gpu, ratio_total_time)
    push!(df, results)
  end
end

if !isempty(df)
  open(path_tex, "w") do io
    pretty_latex_stats(io, df)
  end

  text = read(path_tex, String)
  text = "\\documentclass{article}\n\\usepackage{longtable}\n" *
         "\\usepackage{pdflscape}\n\\usepackage{nopageno}\n" *
         "\\begin{document}\n\\begin{landscape}\n" * text
  text = text * "\\end{landscape}\n\\end{document}"
  write(path_tex, text)
end
