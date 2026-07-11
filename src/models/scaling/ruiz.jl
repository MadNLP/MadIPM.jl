struct RuizScaling{R,C} <: AbstractScaling{R,C}
    row::R
    col::C
end

struct RuizWorkspace{S,R,C,P,Q,W}
    scaling::S
    rownrm::R
    colnrm::C
    rowpair::P   # per-row state: 0 = retired/empty, > 0 = active with this partner column
    colpair::Q   # per-column state, symmetric to `rowpair`
    rowcand::P   # smallest-index argmax candidates gathered by `scaled_argmax!`
    colcand::Q
    storage::W
end

ruiz_equilibration(A; kwargs...) = ruiz_equilibration!(copy(A); kwargs...)

function RuizWorkspace(A)
    drow, dcol = scaling_vectors(A)
    rowpair = similar(drow, Int)
    colpair = similar(dcol, Int)
    return RuizWorkspace(
        RuizScaling(drow, dcol),
        similar(drow), similar(dcol),
        rowpair, colpair,
        similar(rowpair), similar(colpair),
        A,
    )
end
RuizWorkspace(A::SparseMatrixCOO) = RuizWorkspace(SparseCOO(A))

ruiz_equilibration!(A; kwargs...) = ruiz_equilibration!(A, RuizWorkspace(A); kwargs...)

function ruiz_equilibration!(A, ws::RuizWorkspace;
                             max_iter::Integer = RUIZ_DEFAULT_MAXITER,
                             eps::Real = 0.0,
                             strict::Bool = false)
    drow, dcol = ws.scaling.row, ws.scaling.col
    fill!(drow, one(eltype(drow)))
    fill!(dcol, one(eltype(dcol)))
    fill!(ws.rowpair, 1)
    fill!(ws.colpair, 1)

    converged = false
    for sweep in 0:max_iter
        if sweep > 0 && !(any(>(0), ws.rowpair) || any(>(0), ws.colpair))
            converged = true
            break
        end
        _ruiz_sweep!(ws)
        if eps > 0 && sweep > 0 &&
           _active_deviation(ws.rowpair, ws.rownrm) < eps &&
           _active_deviation(ws.colpair, ws.colnrm) < eps
            converged = true
            break
        end
    end
    strict && eps > 0 && !converged &&
        throw(ScalingConvergenceError(Int(max_iter), Float64(eps)))

    drow .= inv.(drow)
    dcol .= inv.(dcol)
    scale_rows_cols!(ws.storage, drow, dcol)
    return ws.storage, ws.scaling
end

function _ruiz_sweep!(ws::RuizWorkspace)
    drow, dcol = ws.scaling.row, ws.scaling.col
    z = zero(eltype(ws.rownrm))
    fill!(ws.rownrm, z)
    fill!(ws.colnrm, z)
    scaled_maxabs!(ws.rownrm, ws.colnrm, ws.rowpair, ws.colpair, ws.storage, drow, dcol)
    fill!(ws.rowcand, typemax(Int))
    fill!(ws.colcand, typemax(Int))
    scaled_argmax!(ws.rowcand, ws.colcand, ws.rownrm, ws.colnrm,
                   ws.rowpair, ws.colpair, ws.storage, drow, dcol)
    ws.rowpair .= ifelse.(ws.rowpair .> 0,
                          ifelse.(ws.rowcand .== typemax(Int), 0, ws.rowcand),
                          ws.rowpair)
    ws.colpair .= ifelse.(ws.colpair .> 0,
                          ifelse.(ws.colcand .== typemax(Int), 0, ws.colcand),
                          ws.colpair)
    drow .= ifelse.(ws.rowpair .> 0, drow .* sqrt.(ws.rownrm), drow)
    dcol .= ifelse.(ws.colpair .> 0, dcol .* sqrt.(ws.colnrm), dcol)
    settle_pairs!(ws.rowpair, ws.colpair)
    return ws
end

function settle_pairs!(rowpair::Vector{Int}, colpair::Vector{Int})
    @inbounds for j in eachindex(colpair)
        i = colpair[j]
        if i > 0 && rowpair[i] == j
            rowpair[i] = 0
            colpair[j] = 0
        end
    end
    return nothing
end

_active_deviation(pair, nrm) =
    mapreduce((p, w) -> p > 0 ? abs(one(w) - w) : zero(w), max, pair, nrm;
              init = zero(eltype(nrm)))
