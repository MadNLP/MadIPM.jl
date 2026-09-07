function scale_model(qp::QuadraticModel, Dr::AbstractVector, Dc::AbstractVector)
    A_src = operator_sparse_matrix(qp.data.A)
    Q_src = operator_sparse_matrix(qp.data.Q)
    data = QPData(
        _scaled_copy(A_src, Dr, Dc),
        qp.data.c .* Dc,
        _scaled_copy(Q_src, Dc, Dc);
        lvar = qp.meta.lvar ./ Dc,
        uvar = qp.meta.uvar ./ Dc,
        lcon = qp.meta.lcon .* Dr,
        ucon = qp.meta.ucon .* Dr,
        c0 = @inbounds qp.data.c0[1]
    )
    return QuadraticModel(
        data;
        x0 = qp.meta.x0 ./ Dc,
        y0 = qp.meta.y0 .* Dr,
        minimize = qp.meta.minimize,
        name = qp.meta.name,
    )
end

function scale_model(lp::LinearModel, Dr::AbstractVector, Dc::AbstractVector)
    A_src = operator_sparse_matrix(lp.data.A)
    data = LPData(
        _scaled_copy(A_src, Dr, Dc),
        lp.data.c .* Dc;
        lvar = lp.meta.lvar ./ Dc,
        uvar = lp.meta.uvar ./ Dc,
        lcon = lp.meta.lcon .* Dr,
        ucon = lp.meta.ucon .* Dr,
        c0 = @inbounds lp.data.c0[1]
    )
    return LinearModel(
        data;
        x0 = lp.meta.x0 ./ Dc,
        y0 = lp.meta.y0 .* Dr,
        minimize = lp.meta.minimize,
        name = lp.meta.name,
    )
end

function _scaled_copy(A::SparseMatrixCOO, Dr::AbstractVector, Dc::AbstractVector)
    rows = copy(A.rows);
    cols = copy(A.cols);
    vals = copy(A.vals)
    @inbounds for k in eachindex(vals)
        vals[k] *= Dr[rows[k]] * Dc[cols[k]]
    end
    return SparseMatrixCOO(size(A, 1), size(A, 2), rows, cols, vals)
end

function _scaled_copy(A::SparseMatrixCSC, Dr::AbstractVector, Dc::AbstractVector)
    rowval = rowvals(A);
    nzval = nonzeros(A)
    out_nz = copy(nzval)
    @inbounds for j in axes(A, 2)
        Dcj = Dc[j]
        for k in nzrange(A, j)
            out_nz[k] *= Dr[rowval[k]] * Dcj
        end
    end
    return SparseMatrixCSC(size(A, 1), size(A, 2), copy(getcolptr(A)), copy(rowval), out_nz)
end
