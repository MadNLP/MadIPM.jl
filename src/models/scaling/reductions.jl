function scale_rows_cols!(A::SparseMatrixCSC, rscale::AbstractVector, cscale::AbstractVector)
    rows, vals = rowvals(A), nonzeros(A)
    @inbounds for j in axes(A, 2)
        cj = cscale[j]
        for k in nzrange(A, j)
            vals[k] *= rscale[rows[k]] * cj
        end
    end
    return A
end

function scale_rows_cols!(A::SparseCOO, rscale::AbstractVector, cscale::AbstractVector)
    A.nzval .*= view(rscale, A.rowval) .* view(cscale, A.colval)
    return A
end

function scaled_maxabs!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T},
                        rowpair, colpair, A::SparseCOO{T},
                        drow::AbstractVector{T}, dcol::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds for k in eachindex(A.nzval)
        i, j = A.rowval[k], A.colval[k]
        ri = rowpair[i] > 0
        cj = colpair[j] > 0
        (ri || cj) || continue
        s = abs(A.nzval[k]) / (drow[i] * dcol[j])
        ri && s > rownrm[i] && (rownrm[i] = s)
        cj && s > colnrm[j] && (colnrm[j] = s)
    end
    return rownrm, colnrm
end

function scaled_argmax!(rowcand, colcand,
                        rownrm::AbstractVector{T}, colnrm::AbstractVector{T},
                        rowpair, colpair, A::SparseCOO{T},
                        drow::AbstractVector{T}, dcol::AbstractVector{T}) where {T<:AbstractFloat}
    @inbounds for k in eachindex(A.nzval)
        i, j = A.rowval[k], A.colval[k]
        ri = rowpair[i] > 0 && rownrm[i] > 0
        cj = colpair[j] > 0 && colnrm[j] > 0
        (ri || cj) || continue
        s = abs(A.nzval[k]) / (drow[i] * dcol[j])
        ri && s == rownrm[i] && j < rowcand[i] && (rowcand[i] = j)
        cj && s == colnrm[j] && i < colcand[j] && (colcand[j] = i)
    end
    return rowcand, colcand
end

function scaled_maxabs!(rownrm::AbstractVector{T}, colnrm::AbstractVector{T},
                        rowpair, colpair, A::SparseMatrixCSC{T},
                        drow::AbstractVector{T}, dcol::AbstractVector{T}) where {T<:AbstractFloat}
    rows, vals = rowvals(A), nonzeros(A)
    @inbounds for j in axes(A, 2)
        cj = colpair[j] > 0
        dj = dcol[j]
        for k in nzrange(A, j)
            i = rows[k]
            ri = rowpair[i] > 0
            (ri || cj) || continue
            s = abs(vals[k]) / (drow[i] * dj)
            ri && s > rownrm[i] && (rownrm[i] = s)
            cj && s > colnrm[j] && (colnrm[j] = s)
        end
    end
    return rownrm, colnrm
end

function scaled_argmax!(rowcand, colcand,
                        rownrm::AbstractVector{T}, colnrm::AbstractVector{T},
                        rowpair, colpair, A::SparseMatrixCSC{T},
                        drow::AbstractVector{T}, dcol::AbstractVector{T}) where {T<:AbstractFloat}
    rows, vals = rowvals(A), nonzeros(A)
    @inbounds for j in axes(A, 2)
        cj = colpair[j] > 0 && colnrm[j] > 0
        dj = dcol[j]
        for k in nzrange(A, j)
            i = rows[k]
            ri = rowpair[i] > 0 && rownrm[i] > 0
            (ri || cj) || continue
            s = abs(vals[k]) / (drow[i] * dj)
            ri && s == rownrm[i] && j < rowcand[i] && (rowcand[i] = j)
            cj && s == colnrm[j] && i < colcand[j] && (colcand[j] = i)
        end
    end
    return rowcand, colcand
end
