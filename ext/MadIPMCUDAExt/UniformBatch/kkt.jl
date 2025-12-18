abstract type AbstractBatchKKTSystem{KKTSystem,LS} end

"""
    UniformBatchKKTSystem{KKTSystem, LS, T, VT, VI} <: AbstractBatchKKTSystem{KKTSystem,LS}

Batch wrapper for a set of sparse KKT systems that share the same sparsity pattern,
enabling uniform-batch factorization and solves with a single `linear_solver`
(e.g. a `MadNLPGPU.CUDSSSolver` in uniform batch mode).

`UniformBatchKKTSystem` owns:
- the collection of individual KKT systems,
- a pool of unreduced KKT vectors,
- contiguous storage for all right-hand sides and nonzero values in the batch,
- and bookkeeping to track which systems are currently active in the batch.

It assumes all KKT systems in `kkts` have identical `aug_com.colPtr` and
`aug_com.rowVal`, so that only their `nzVal` and RHS differ across the batch.

# Fields
- `kkts::Vector{KKTSystem}`: Collection of individual KKT systems.
- `vecs::Vector{MadNLP.UnreducedKKTVector{T,VT,VI}}`:
  Per-system unreduced KKT vectors holding the full primal/dual RHS and solution
  for each problem. These are the vectors that are packed/unpacked into the
  batched RHS buffer.
- `linear_solver::LS`:
  Shared linear solver instance that operates on the batched KKT data. 
- `batch_rhs::VT`:
  Contiguous storage for the right-hand side of *all* systems in the batch.
- `batch_nzVal::VT`:
  Contiguous storage for the nonzero values of *all* KKT matrices in the batch,
  with fixed block size `nzVal_size` per system. Each KKT’s `aug_com.nzVal`
  is made to point into its corresponding slice of this buffer.
- `rhs_slices::Vector{VT}`:
  Views or slices into `batch_rhs`, one per system.
- `nzVal_slices::Vector{VT}`:
  Views or slices into `batch_nzVal`, one per system.
- `rhs_size::Int`:
  Length of the RHS vector for a single KKT system. Used to compute offsets
  into `batch_rhs`.
- `nzVal_size::Int`:
  Number of nonzero entries in a single KKT matrix. Used to compute offsets
  into `batch_nzVal`.
- `batch_map::Vector{Int}`:
  Maps logical system indices (`1:batch_size`) to positions in the *active*
  batch. If some systems are inactive, this map is used to compact the batch
  before calling the linear solver.
- `batch_map_rev::Vector{Int}`:
  Reverse map of `batch_map`. Given a position in the active batch, this tells
  you which original system index it corresponds to.
- `batch_size::Int`:
  Total number of KKT systems managed by this batch object (length of `kkts`).
- `active_batch_size::Base.RefValue{Int}`:
  Current number of active systems in the batch. This is typically the value
  passed to the underlying batched linear solver (e.g. `ubatch_size` in CUDSS).
- `is_active::BitVector`:
  Boolean mask indicating which systems are currently active (participating) in
  the batch solve. Inactive systems are skipped when packing `active_rhs` and
  configuring the solver.
- `active_rhs::Base.RefValue{VT}`:
  Reference to the RHS buffer actually passed to the linear solver. This is
  usually a view into `batch_rhs` that only covers the active systems in
  compact form, after applying `batch_map`.
"""
struct UniformBatchKKTSystem{  # NOTE: move to MadIPM/MadNLP
    KKTSystem<:MadNLP.AbstractSparseKKTSystem, LS, T, VT<:AbstractVector{T}, VI
} <: AbstractBatchKKTSystem{KKTSystem,LS}
    kkts::Vector{KKTSystem}
    vecs::Vector{MadNLP.UnreducedKKTVector{T,VT,VI}}
    linear_solver::LS
    batch_rhs::VT
    batch_nzVal::VT
    rhs_slices::Vector{VT}
    nzVal_slices::Vector{VT}
    rhs_size::Int
    nzVal_size::Int
    batch_map::Vector{Int}
    batch_map_rev::Vector{Int}
    batch_size::Int
    active_batch_size::Base.RefValue{Int}
    is_active::BitVector
    active_rhs::Base.RefValue{VT}
end

all_done(bkkt::UniformBatchKKTSystem) = (bkkt.active_batch_size[] == 0)
is_active(bkkt::UniformBatchKKTSystem, i) = bkkt.is_active[i]

function UniformBatchKKTSystem(  # NOTE: move to MadNLPGPU
    kkts::Vector{KKTSystem},
    vecs::Vector{KKTVector},
    linear_solver::Type{MadNLPGPU.CUDSSSolver};
    opt_linear_solver=MadNLP.default_options(linear_solver),
) where {T,VT,KKTSystem<:MadNLP.AbstractSparseKKTSystem,KKTVector<:MadNLP.UnreducedKKTVector{T,VT}}
    kkt1 = first(kkts)

    for kkt in kkts
        @assert kkt.aug_com.colPtr == kkt1.aug_com.colPtr "Cannot use UniformBatchKKTSystem when KKTSystems do not share sparsity structure (colPtr)."
        @assert kkt.aug_com.rowVal == kkt1.aug_com.rowVal "Cannot use UniformBatchKKTSystem when KKTSystems do not share sparsity structure (rowVal)."
    end

    vec1 = first(vecs)
    rhs_size = get_rhs_size(kkt1, vec1)
    nzVal_size = length(nonzeros(kkt1.aug_com))

    batch_size = length(kkts)
    batch_rhs = fill!(VT(undef, rhs_size * batch_size), zero(T))
    batch_nzVal = fill!(VT(undef, nzVal_size * batch_size), zero(T))

    rhs_slices = [MadNLP._madnlp_unsafe_wrap(batch_rhs, rhs_size, (j - 1) * rhs_size + 1) for j in 1:batch_size]
    nzVal_slices = [MadNLP._madnlp_unsafe_wrap(batch_nzVal, nzVal_size, (j - 1) * nzVal_size + 1) for j in 1:batch_size]

    batch_aug_com = similar(kkt1.aug_com)
    batch_aug_com.nzVal = batch_nzVal
    _linear_solver = linear_solver(
        batch_aug_com; opt = opt_linear_solver
    )

    bkkt = UniformBatchKKTSystem(
        kkts, vecs, _linear_solver,
        batch_rhs, batch_nzVal,
        rhs_slices, nzVal_slices,
        rhs_size, nzVal_size,
        collect(1:batch_size), collect(1:batch_size),
        batch_size, Ref(batch_size),
        trues(batch_size),
        Ref(MadNLP._madnlp_unsafe_wrap(batch_rhs, rhs_size * batch_size, 1)),
    )

    update_pointers!(bkkt)

    return bkkt
end

function update_batch!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    # NOTE: only called if an update is needed
    active_pos = 0
    for i in 1:bkkt.batch_size
        if bkkt.is_active[i]
            active_pos += 1
            bkkt.batch_map[i] = active_pos
            bkkt.batch_map_rev[active_pos] = i
        else
            bkkt.batch_map[i] = 0
        end
    end

    for j in (active_pos + 1):bkkt.batch_size
        bkkt.batch_map_rev[j] = 0
    end

    bkkt.active_batch_size[] = active_pos
    bkkt.active_rhs[] = MadNLP._madnlp_unsafe_wrap(bkkt.batch_rhs, active_pos * bkkt.rhs_size, 1)
    bkkt.linear_solver.tril.nzVal = MadNLP._madnlp_unsafe_wrap(bkkt.batch_nzVal, active_pos * bkkt.nzVal_size, 1)

    update_pointers!(bkkt)

    MadNLPGPU.CUDSS.cudss_set(bkkt.linear_solver.inner, "ubatch_size", active_pos)
    return
end

function update_pointers!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    for (i, kkt_i) in enumerate(bkkt.kkts)
        batch_pos = bkkt.batch_map[i]
        if batch_pos > 0
            kkt_i.aug_com.nzVal = bkkt.nzVal_slices[batch_pos]
        end
    end
    return
end

function batch_factorize!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    MadNLP.factorize!(bkkt.linear_solver)
    return
end

function batch_solve!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    copy_batch_rhs!(bkkt)
    MadNLP.solve!(bkkt.linear_solver, bkkt.active_rhs[])
    copy_batch_solution!(bkkt)
    return
end

# TODO: can be better (each copyto syncs)
function copy_batch_rhs!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    for active_i in 1:bkkt.active_batch_size[]
        dest = bkkt.rhs_slices[active_i]
        vec = get_active_vec(bkkt, active_i)
        src = get_rhs(KKTSystem, vec)
        copyto!(dest, src)
    end
    return
end

function copy_batch_solution!(bkkt::UniformBatchKKTSystem{KKTSystem,LS}) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    for active_i in 1:bkkt.active_batch_size[]
        vec = get_active_vec(bkkt, active_i)
        dest = get_rhs(KKTSystem, vec)
        src = bkkt.rhs_slices[active_i]
        copyto!(dest, src)
    end
    return
end

function get_active_vec(bkkt::UniformBatchKKTSystem{KKTSystem,LS}, active_i) where {KKTSystem,LS<:MadNLPGPU.CUDSSSolver}
    return bkkt.vecs[bkkt.batch_map_rev[active_i]]
end