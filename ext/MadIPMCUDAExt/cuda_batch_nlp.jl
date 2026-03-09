import QuadraticModels: ObjRHSBatchQuadraticModel, BatchQuadraticModel, QPData

function NLPModels.obj!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
    bx::AbstractMatrix{T}, bf::AbstractVector{T},
) where {T, S, M1 <: MadIPMOperator, M2, MT}
    if !bqp.meta.islp
        mul!(bqp._HX, bqp.data.H, bx)
        bf .= bqp.data.c0 .+ vec(sum(bqp.c_batch .* bx, dims=1)) .+ T(0.5) .* vec(sum(bx .* bqp._HX, dims=1))
    else
        bf .= bqp.data.c0 .+ vec(sum(bqp.c_batch .* bx, dims=1))
    end
    return bf
end

function NLPModels.grad!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
    bx::AbstractMatrix{T}, bg::AbstractMatrix{T},
) where {T, S, M1 <: MadIPMOperator, M2, MT}
    if !bqp.meta.islp
        mul!(bg, bqp.data.H, bx)
        bg .+= bqp.c_batch
    else
        copyto!(bg, bqp.c_batch)
    end
    return bg
end

function NLPModels.cons!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
    bx::AbstractMatrix{T}, bc::AbstractMatrix{T},
) where {T, S, M1, M2 <: MadIPMOperator, MT}
    mul!(bc, bqp.data.A, bx)
    return bc
end

function NLPModels.jac_structure!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
    jrows::AbstractVector{<:Integer},
    jcols::AbstractVector{<:Integer},
) where {T, S, M1, M2 <: MadIPMOperator}
    fill_structure!(bqp.data.A.A, jrows, jcols)
    return jrows, jcols
end

function NLPModels.hess_structure!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
    hrows::AbstractVector{<:Integer},
    hcols::AbstractVector{<:Integer},
) where {T, S, M1 <: MadIPMOperator, M2}
    fill_structure!(bqp.data.H.A, hrows, hcols)
    return hrows, hcols
end

function NLPModels.jac_coord!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
    bx::AbstractMatrix,
    bjvals::AbstractMatrix,
) where {T, S, M1, M2 <: MadIPMOperator}
    bjvals .= bqp.data.A.A.nzVal
    return bjvals
end

function NLPModels.hess_coord!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2},
    bx::AbstractMatrix,
    by::AbstractMatrix,
    bobj_weight::AbstractVector,
    bhvals::AbstractMatrix,
) where {T, S, M1 <: MadIPMOperator, M2}
    H = bqp.data.H.A
    nnzh = nnz(H)
    nnzh == 0 && return bhvals
    bhvals .= H.nzVal .* bobj_weight'
    return bhvals
end

function NLPModels.hprod!(
    bqp::ObjRHSBatchQuadraticModel{T, S, M1, M2, MT},
    bx::AbstractMatrix{T}, by::AbstractMatrix{T}, bv::AbstractMatrix{T},
    bobj_weight::AbstractVector{T}, bHv::AbstractMatrix{T},
) where {T, S, M1 <: MadIPMOperator, M2, MT}
    mul!(bHv, bqp.data.H, bv)
    bHv .*= bobj_weight'
    return bHv
end

function _expand_symmetric_coo(H::SparseMatrixCOO{Tv, Ti}) where {Tv, Ti}
    rows, cols, vals = H.rows, H.cols, H.vals
    m, n = size(H)
    offdiag = findall(i -> rows[i] != cols[i], 1:length(rows))
    new_rows = vcat(rows, cols[offdiag])
    new_cols = vcat(cols, rows[offdiag])
    new_vals = vcat(vals, vals[offdiag])
    return SparseMatrixCOO(m, n, new_rows, new_cols, new_vals)
end

function Base.convert(::Type{ObjRHSBatchQuadraticModel{T, S}}, bnlp::ObjRHSBatchQuadraticModel{T}) where {T, S<:CuArray}
    nbatch = bnlp.meta.nbatch
    nvar = bnlp.meta.nvar
    ncon = bnlp.meta.ncon

    H_orig_csr = CUSPARSE.CuSparseMatrixCSR(bnlp.data.H)

    H_full = _expand_symmetric_coo(bnlp.data.H)
    H_full_csr = CUSPARSE.CuSparseMatrixCSR(H_full)

    A_csr = CUSPARSE.CuSparseMatrixCSR(bnlp.data.A)

    H_op = MadIPMOperator(H_full_csr; symmetric=false, spmm_ncols=nbatch)
    H_op.A = H_orig_csr
    A_op = MadIPMOperator(A_csr; symmetric=false, spmm_ncols=nbatch)

    c_gpu = CuVector{T}(bnlp.data.c)
    v_gpu = CuVector{T}(bnlp.data.v)
    data_gpu = QPData(bnlp.data.c0, c_gpu, v_gpu, H_op, A_op)

    c_batch_gpu = CuMatrix{T}(bnlp.c_batch)
    _HX_gpu = CUDA.zeros(T, nvar, nbatch)
    _AX_gpu = CUDA.zeros(T, ncon, nbatch)

    VT = typeof(c_gpu)
    MT = typeof(c_batch_gpu)

    meta_gpu = NLPModels.BatchNLPModelMeta{T, MT}(
        nbatch, nvar;
        x0 = CuMatrix{T}(bnlp.meta.x0),
        lvar = CuMatrix{T}(bnlp.meta.lvar),
        uvar = CuMatrix{T}(bnlp.meta.uvar),
        ncon = ncon,
        lcon = CuMatrix{T}(bnlp.meta.lcon),
        ucon = CuMatrix{T}(bnlp.meta.ucon),
        nnzj = bnlp.meta.nnzj,
        nnzh = bnlp.meta.nnzh,
        islp = bnlp.meta.islp,
    )

    return ObjRHSBatchQuadraticModel{T, VT, typeof(H_op), typeof(A_op), MT}(
        meta_gpu, data_gpu, c_batch_gpu, _HX_gpu, _AX_gpu,
    )
end

function Base.convert(::Type{BatchQuadraticModel{T, MT}}, bnlp::BatchQuadraticModel{T}) where {T, MT<:CuMatrix}
    nbatch = bnlp.meta.nbatch
    nvar = bnlp.meta.nvar
    ncon = bnlp.meta.ncon

    c_batch_gpu = MT(bnlp.c_batch)
    c0_batch_gpu = CuVector{T}(bnlp.c0_batch)
    H_nzvals_gpu = MT(bnlp.H_nzvals)
    A_nzvals_gpu = MT(bnlp.A_nzvals)

    hess_rows_gpu = CuVector{Int}(bnlp.hess_rows)
    hess_cols_gpu = CuVector{Int}(bnlp.hess_cols)

    _jac_scatter_gpu = CUSPARSE.CuSparseMatrixCSC(bnlp._jac_scatter)
    _jact_scatter_gpu = CUSPARSE.CuSparseMatrixCSC(bnlp._jact_scatter)
    _hess_scatter_gpu = CUSPARSE.CuSparseMatrixCSC(bnlp._hess_scatter)

    _hess_sym_gather_cols_gpu = CuVector{Int}(bnlp._hess_sym_gather_cols)
    _hess_sym_nzidx_gpu = CuVector{Int}(bnlp._hess_sym_nzidx)

    _HX_gpu = CUDA.zeros(T, nvar, nbatch)

    VT = typeof(c0_batch_gpu)
    VI = typeof(hess_rows_gpu)
    SpMT_J = typeof(_jac_scatter_gpu)
    SpMT_H = typeof(_hess_scatter_gpu)

    meta_gpu = NLPModels.BatchNLPModelMeta{T, MT}(
        nbatch, nvar;
        x0 = MT(bnlp.meta.x0),
        lvar = MT(bnlp.meta.lvar),
        uvar = MT(bnlp.meta.uvar),
        ncon = ncon,
        lcon = MT(bnlp.meta.lcon),
        ucon = MT(bnlp.meta.ucon),
        nnzj = bnlp.meta.nnzj,
        nnzh = bnlp.meta.nnzh,
        islp = bnlp.meta.islp,
    )

    return BatchQuadraticModel{T, MT, SpMT_J, SpMT_H, VT, VI}(
        meta_gpu,
        c_batch_gpu, c0_batch_gpu, H_nzvals_gpu, A_nzvals_gpu,
        hess_rows_gpu, hess_cols_gpu,
        _jac_scatter_gpu, _jact_scatter_gpu, _hess_scatter_gpu,
        _hess_sym_gather_cols_gpu, _hess_sym_nzidx_gpu,
        _HX_gpu,
    )
end
