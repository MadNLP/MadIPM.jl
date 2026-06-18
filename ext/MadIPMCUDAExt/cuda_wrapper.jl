using MadNLPGPU
import LinearAlgebra: BlasFloat

@kernel function _transfer_to_map!(dest, to_map, src)
    k = @index(Global, Linear)
    @inbounds begin
        Atomix.@atomic dest[to_map[k]] += src[k]
    end
end

function MadNLP.transfer!(
    dest::cuSPARSE.CuSparseMatrixCSC{Tv},
    src::MadNLP.SparseMatrixCOO{Tv},
    map::CuVector{Int},
) where {Tv}
    return MadNLP._transfer!(dest.nzVal, src.V, map)
end

function MadNLP._transfer!(dest::CuVector{T}, src::CuVector{T}, map::CuVector{Int}) where T
    fill!(dest, zero(T))
    if length(map) > 0
        backend = CUDABackend()
        _transfer_to_map!(backend)(dest, map, src; ndrange=length(map))
    end
    return
end

function MadNLP.compress_hessian!(
    kkt::MadNLP.SparseKKTSystem{T,VT,MT},
) where {T,VT,MT<:cuSPARSE.CuSparseMatrixCSC{T,Int32}}
    MadNLP.transfer!(kkt.hess_com, kkt.hess_raw, kkt.hess_csc_map)
end

function MadNLP.compress_jacobian!(
    kkt::MadIPM.NormalKKTSystem{T,VT,MT},
) where {T,VT,MT<:cuSPARSE.CuSparseMatrixCSC{T,Int32}}
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzVal, 0.0)
    kkt.AT.nzVal .= kkt.A.V[kkt.A_csr_map]
    return
end

function MadIPM.coo_to_csr(
    n_rows,
    n_cols,
    Ai::CuVector{Ti},
    Aj::CuVector{Ti},
    Ax::CuVector{Tv},
) where {Tv, Ti}
    @assert length(Ai) == length(Aj) == length(Ax)
    B = sparse(Ai, Aj, Ax, n_rows, n_cols; fmt=:csr)
    return (B.rowPtr, B.colVal, B.nzVal)
end

@kernel function assemble_normal_system_kernel!(@Const(n_rows), @Const(n_cols), @Const(Jtp), @Const(Jtj), @Const(Jtx),
                                                @Const(Cp), @Const(Cj), Cx, @Const(Dx), @Const(Tv))
    i = @index(Global, Linear)

    for c in Cp[i]:Cp[i+1]-1
        j = Cj[c]
        acc = zero(Tv)

        p1 = Jtp[i]
        p2 = Jtp[j]
        p1_end = Jtp[i+1] - 1
        p2_end = Jtp[j+1] - 1

        while p1 <= p1_end && p2 <= p2_end
            k1 = Jtj[p1]
            k2 = Jtj[p2]

            if k1 == k2
                acc += Jtx[p1] * Dx[k1] * Jtx[p2]
                p1 += 1
                p2 += 1
            elseif k1 < k2
                p1 += 1
            else
                p2 += 1
            end
        end

        Cx[c] = acc
    end
    nothing
end

function MadIPM.assemble_normal_system!(
    n_rows,
    n_cols,
    Jtp::CuArray{Ti},
    Jtj::CuArray{Ti},
    Jtx::CuArray{Tv},
    Cp::CuArray{Ti},
    Cj::CuArray{Ti},
    Cx::CuArray{Tv},
    Dx::CuArray{Tv},
) where {Ti, Tv}
    backend = CUDABackend()
    kernel! = assemble_normal_system_kernel!(backend)
    kernel!(n_rows, n_cols, Jtp, Jtj, Jtx, Cp, Cj, Cx, Dx, Tv; ndrange = n_rows)
end



MadIPM.sparse_csc_format(::Type{<:CuArray}) = CuSparseMatrixCSC
MadIPM._colptr(A::CuSparseMatrixCSC) = A.colPtr
MadIPM._rowval(A::CuSparseMatrixCSC) = A.rowVal
MadIPM._nzval(A::CuSparseMatrixCSC) = A.nzVal

# we introduce a new constructor that takes the nzvals as a matrix explicitly
function MadNLPGPU.CUDSSSolver(
    aug_com::cuSPARSE.CuSparseMatrixCSC{T,Cint},
    nzvals_mat::CuMatrix{T},
    n::Int;
    opt::MadNLPGPU.CudssSolverOptions = MadNLPGPU.CudssSolverOptions(),
) where T
    batch_nzVal = vec(nzvals_mat)
    batch_aug_com = cuSPARSE.CuSparseMatrixCSC(
        aug_com.colPtr, aug_com.rowVal, batch_nzVal, size(aug_com),
    )
    solver = MadNLPGPU.CUDSSSolver(batch_aug_com; opt=opt)
    solver.tril.nzVal = batch_nzVal
    return solver
end

MadIPM.is_factorized(::MadNLPGPU.CUDSSSolver) = true

function MadIPM.factorize_active!(s::MadNLPGPU.CUDSSSolver, active::MadIPM.BatchView)
    na = MadIPM.local_batch_size(active)
    CUDSS.cudss_set(s.inner, "ubatch_size", na)
    MadNLP.factorize!(s)
    return
end

function MadIPM.solve_active!(s::MadNLPGPU.CUDSSSolver{T}, rhs::CuMatrix{T}, active::MadIPM.BatchView) where T
    na = MadIPM.local_batch_size(active)
    n = size(rhs, 1)
    rhs_active = unsafe_wrap(CuArray{T, 2}, pointer(rhs), (n, na))
    CUDSS.cudss_update(s.b_gpu, rhs_active)
    CUDSS.cudss_update(s.x_gpu, rhs_active)
    CUDSS.cudss("solve", s.inner, s.x_gpu, s.b_gpu, asynchronous=s.opt.cudss_asynchronous)
    return
end
