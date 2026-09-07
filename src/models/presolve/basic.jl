"""
    BasicPresolver(; max_passes=16, verbose=false)

Pure-Julia presolver that repeatedly applies simple reductions until no
progress is made (or `max_passes` is reached):

- inconsistent variable or constraint bounds are detected up front;
- *fixed variables* (`lvar[j] == uvar[j]`) are eliminated, propagating the
  fixed contribution into the objective constant `c0`, the linear
  coefficients of remaining variables (via `Q` cross-terms when present), and
  the constraint bounds;
- *singleton rows* transfer the bounds of one-variable constraints onto the
  variable and remove the constraint. [`recover_solution`](@ref) reports a
  zero multiplier for the removed row; its dual weight appears on the
  variable bound instead;
- *forcing rows* — rows whose implied activity range (from the variable
  bounds) touches one of the row bounds — pin every variable in the row to
  the bound attaining that activity and are removed. [`recover_solution`](@ref)
  rebuilds the row multiplier from the reduced costs of the pinned variables;
- *redundant rows* — rows whose implied activity range lies inside the row
  bounds — are removed with an exactly-zero multiplier; when only one side can
  never bind, that side is relaxed to `±Inf` instead (which may turn the row
  into a free row);
- *free rows* (`(-Inf, Inf)` bounds) never bind and are removed with an
  exactly-zero multiplier;
- *empty rows* — constraints without a nonzero in any active variable — are
  removed, detecting infeasibility when the surviving bounds cannot bracket 0
  (up to a tolerance proportional to what was folded into them);
- *empty columns* — variables appearing in no active constraint — are fixed
  to the optimum for the model objective sense, detecting unboundedness when
  the favored direction is unbounded;
- *free singleton columns* — free variables appearing in exactly one
  constraint and in no quadratic term — are eliminated together with that
  constraint (the variable absorbs any row activity). The substituted
  objective picks the row activity; [`recover_solution`](@ref)
  back-substitutes the variable value and the row multiplier `c[j]/a[i,j]`.
  Detects unboundedness when the objective pushes the row activity toward an
  infinite bound.

Multipliers reported by [`recover_solution`](@ref) follow the convention
`c + Q x - Aᵀ y = z_l - z_u` with `z_l, z_u >= 0`.
"""
struct BasicPresolver <: AbstractPresolver
    max_passes::Int
    verbose::Bool

    function BasicPresolver(; max_passes::Int = 16, verbose::Bool = false)
        max_passes >= 1 || throw(ArgumentError("max_passes must be >= 1"))
        return new(max_passes, verbose)
    end
end

Base.show(io::IO, p::BasicPresolver) =
    print(io, "BasicPresolver(max_passes: ", p.max_passes, ")")

struct _FreeSingletonColOp{T}
    i::Int
    j::Int
    aij::T
    activity::T
    yi::T
    row_idx::Vector{Int}
    row_val::Vector{T}
end

struct _ForcingRowOp{T}
    i::Int
    at_lower::Bool        # activity pinned to lcon (true) or ucon (false)
    equality::Bool        # lcon == ucon when detected: multiplier sign unconstrained
    col_idx::Vector{Int}  # variables pinned by this row (already-fixed ones excluded)
    col_val::Vector{T}    # their coefficients a_ij
end

struct BasicPresolveResult{T,M<:ScalarModel,MQ} <: AbstractPresolveResult
    reduced_model::M
    var_map::Vector{Int}
    con_map::Vector{Int}
    fixed_var_idx::Vector{Int}
    fixed_var_val::Vector{T}
    free_col_ops::Vector{_FreeSingletonColOp{T}}
    forcing_ops::Vector{_ForcingRowOp{T}}
    minimize::Bool                # objective sense        } used to rebuild the
    c::Vector{T}                  # original linear cost   } multipliers of
    A::SparseMatrixCSC{T,Int}    # original A             } forcing rows
    Q::MQ                         # original Q (lower) or nothing
    n_orig::Int
    m_orig::Int
end

struct BasicSolvedResult{T,MQ} <: AbstractPresolveResult
    fixed_var_idx::Vector{Int}
    fixed_var_val::Vector{T}
    free_col_ops::Vector{_FreeSingletonColOp{T}}
    forcing_ops::Vector{_ForcingRowOp{T}}
    minimize::Bool
    c::Vector{T}
    A::SparseMatrixCSC{T,Int}
    Q::MQ
    n_orig::Int
    m_orig::Int
    objective_value::T
end

mutable struct _BasicScratch{T}
    minimize::Bool
    c::Vector{T};
    c0::T
    c_orig::Vector{T}         # untouched copy of `c`, for dual recovery
    lvar::Vector{T};
    uvar::Vector{T}
    lcon::Vector{T};
    ucon::Vector{T}
    A::SparseMatrixCSC{T,Int}
    At::SparseMatrixCSC{T,Int}
    Q::Union{Nothing,SparseMatrixCSC{T,Int}}  # nothing for LinearModel
    row_scale::Vector{T}      # magnitude of the bounds plus everything folded into them
    var_keep::BitVector
    con_keep::BitVector
    fixed_idx::Vector{Int}
    fixed_val::Vector{T}
    free_col_ops::Vector{_FreeSingletonColOp{T}}
    forcing_ops::Vector{_ForcingRowOp{T}}
end

_to_csc(A_op) = _to_csc_storage(operator_sparse_matrix(A_op))
_to_csc_sym(Q_op) = _to_csc_storage(operator_sparse_matrix(Q_op))
_to_csc_storage(A::SparseMatrixCSC) = A
_to_csc_storage(A::SparseMatrixCOO) = sparse(A.rows, A.cols, A.vals, size(A)...)

_finite_mag(x::T) where {T} = isfinite(x) ? abs(x) : zero(T)

function _scratch(model::ScalarModel)
    d = model.data
    T = eltype(d.c)
    A = _to_csc(d.A)
    Q = model isa QuadraticModel ? _to_csc_sym(d.Q) : nothing
    c = Vector{T}(d.c)
    lcon = Vector{T}(d.lcon)
    ucon = Vector{T}(d.ucon)
    return _BasicScratch{T}(
        model.meta.minimize,
        c,
        (@inbounds d.c0[1]),
        copy(c),
        Vector{T}(d.lvar),
        Vector{T}(d.uvar),
        lcon,
        ucon,
        A,
        sparse(transpose(A)),
        Q,
        max.(one(T), _finite_mag.(lcon), _finite_mag.(ucon)),
        trues(length(d.c)),
        trues(size(A, 1)),
        Int[],
        T[],
        _FreeSingletonColOp{T}[],
        _ForcingRowOp{T}[],
    )
end

# Tolerance for "this emptied row's residual bounds still bracket zero".
# Everything folded into a row's bounds carries rounding of order
# eps * |folded|, which is exactly what `row_scale` tracks.
_row_tol(s::_BasicScratch{T}, i::Int) where {T} = sqrt(eps(T)) * s.row_scale[i]

# ---- Per-pass reductions ----------------------------------------------------

# Detect inconsistent bounds. Returns `:infeasible` or `0` so the top-level
# driver can treat it like the other pass functions.
function _pass_bounds!(s::_BasicScratch)
    for j in eachindex(s.var_keep)
        s.var_keep[j] || continue
        s.lvar[j] > s.uvar[j] && return :infeasible
    end
    for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        s.lcon[i] > s.ucon[i] && return :infeasible
    end
    return 0
end

# A row with one active nonzero a*x_j and bounds l <= a*x_j <= u directly
# implies bounds on x_j: transfer them and remove the (now redundant) row.
# Recovery reports y_i = 0 for the removed row; at an optimum its dual weight
# shows up on the transferred variable bound instead.
function _pass_singleton_rows!(s::_BasicScratch{T}) where {T}
    At = s.At
    n_removed = 0
    @inbounds for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        singleton_j = 0
        singleton_a = zero(T)
        n_active = 0
        for p in nzrange(At, i)
            j = At.rowval[p]
            s.var_keep[j] || continue
            a = At.nzval[p]
            iszero(a) && continue
            n_active += 1
            n_active > 1 && break
            singleton_j = j
            singleton_a = a
        end
        n_active == 1 || continue

        implied_l, implied_u = _singleton_bounds(s.lcon[i], s.ucon[i], singleton_a)
        new_l = max(s.lvar[singleton_j], implied_l)
        new_u = min(s.uvar[singleton_j], implied_u)
        new_l > new_u && return :infeasible
        s.lvar[singleton_j] = new_l
        s.uvar[singleton_j] = new_u
        s.con_keep[i] = false
        n_removed += 1
    end
    return n_removed
end

@inline function _singleton_bounds(l, u, a)
    return a > 0 ? (l / a, u / a) : (u / a, l / a)
end

# Implied activity range of row i from the variable bounds,
#   lact = Σ_j (a_ij > 0 ? a_ij·lvar_j : a_ij·uvar_j),
#   uact = Σ_j (a_ij > 0 ? a_ij·uvar_j : a_ij·lvar_j),
# over active nonzero entries (lact ∈ [-∞, ∞), uact ∈ (-∞, ∞], never NaN), then
#   uact < lcon or lact > ucon    → infeasible;
#   uact == lcon (finite)         → forcing: every variable sits at the bound attaining uact;
#   lact == ucon (finite)         → forcing, symmetric;
#   lcon <= lact and uact <= ucon → redundant row, dropped (multiplier 0);
#   lcon <= lact only             → the lower side can never bind: lcon = -∞;
#   uact <= ucon only             → the upper side can never bind: ucon = +∞.
function _pass_row_activity!(s::_BasicScratch{T}) where {T}
    At = s.At
    n_forced = 0
    n_removed = 0
    n_relaxed = 0
    @inbounds for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        l_i, u_i = s.lcon[i], s.ucon[i]
        (l_i == -T(Inf) && u_i == T(Inf)) && continue
        lact = zero(T)
        uact = zero(T)
        n_active = 0
        for p in nzrange(At, i)
            j = At.rowval[p]
            s.var_keep[j] || continue
            a = At.nzval[p]
            iszero(a) && continue
            n_active += 1
            lj, uj = s.lvar[j], s.uvar[j]
            if a > 0
                lact += a * lj
                uact += a * uj
            else
                lact += a * uj
                uact += a * lj
            end
        end
        n_active == 0 && continue   # `_pass_empty_rows!` owns these
        (uact < l_i || lact > u_i) && return :infeasible
        if uact == l_i && isfinite(l_i)
            _force_row!(s, i, true)
            n_forced += 1
        elseif lact == u_i && isfinite(u_i)
            _force_row!(s, i, false)
            n_forced += 1
        else
            low_redundant = l_i <= lact
            up_redundant = uact <= u_i
            if low_redundant && up_redundant
                s.con_keep[i] = false
                n_removed += 1
            elseif low_redundant && l_i != -T(Inf)
                s.lcon[i] = -T(Inf)
                n_relaxed += 1
            elseif up_redundant && u_i != T(Inf)
                s.ucon[i] = T(Inf)
                n_relaxed += 1
            end
        end
    end
    return n_forced, n_removed, n_relaxed
end

# Pin every active variable of forcing row i to the bound attaining the
# extreme activity (finite by construction) and drop the row now, so the
# next `_pass_fixed_vars!` never folds the pinned values back into it and
# `_pass_empty_rows!` never sees a residual rounding error on it. Variables
# that were already fixed contribute a constant and impose nothing on the
# multiplier, so they are left out of the op record.
function _force_row!(s::_BasicScratch{T}, i::Int, at_lower::Bool) where {T}
    At = s.At
    col_idx = Int[]
    col_val = T[]
    @inbounds for p in nzrange(At, i)
        j = At.rowval[p]
        s.var_keep[j] || continue
        a = At.nzval[p]
        iszero(a) && continue
        s.lvar[j] == s.uvar[j] && continue
        to_upper = at_lower ? (a > 0) : (a < 0)
        if to_upper
            s.lvar[j] = s.uvar[j]
        else
            s.uvar[j] = s.lvar[j]
        end
        push!(col_idx, j)
        push!(col_val, a)
    end
    push!(
        s.forcing_ops,
        _ForcingRowOp{T}(i, at_lower, s.lcon[i] == s.ucon[i], col_idx, col_val),
    )
    s.con_keep[i] = false
    return nothing
end

# Fix all variables j with `lvar[j] == uvar[j]`. Propagates v = lvar[j] = uvar[j]
# through the objective (`c0` and Q cross-terms), the surviving linear cost
# (`c[k] += v * Q[k,j]`), and the constraint bounds. Returns the count fixed.
function _pass_fixed_vars!(s::_BasicScratch{T}) where {T}
    n_fixed = 0
    for j in eachindex(s.var_keep)
        s.var_keep[j] || continue
        s.lvar[j] == s.uvar[j] || continue
        v = s.lvar[j]
        _eliminate_var!(s, j, v)
        n_fixed += 1
    end
    return n_fixed
end

function _eliminate_var!(s::_BasicScratch{T}, j::Int, v::T) where {T}
    s.c0 += s.c[j] * v
    _propagate_quadratic!(s, j, v)
    _propagate_constraints!(s, j, v)
    s.var_keep[j] = false
    push!(s.fixed_idx, j);
    push!(s.fixed_val, v)
    return nothing
end

# c0 += ½v²·Q[j,j];  c[k] += v·Q[k,j]  for k != j active.
# Q is stored lower-triangular (Symmetric(csc, :L)): for `j`'s contributions we
# scan column `j` (k >= j entries) and row `j` (which lives as column k for
# k < j entries, accessed via the symmetric reflection).
_propagate_quadratic!(::_BasicScratch, ::Int, _) = nothing
function _propagate_quadratic!(s::_BasicScratch{T}, j::Int, v::T) where {T}
    s.Q === nothing && return nothing
    Q = s.Q
    # Column j: stored entries are (k, j) with k >= j.
    @inbounds for p in nzrange(Q, j)
        k, q_kj = Q.rowval[p], Q.nzval[p]
        if k == j
            s.c0 += T(0.5) * v * v * q_kj
        elseif s.var_keep[k]
            s.c[k] += v * q_kj
        end
    end
    # Row j via reflection: scan columns k < j; entry (j, k) is stored as
    # rowval[p] == j in column k. Skip k >= j (already covered above).
    @inbounds for k = 1:(j-1)
        s.var_keep[k] || continue
        for p in nzrange(Q, k)
            Q.rowval[p] == j || continue
            s.c[k] += v * Q.nzval[p]
            break
        end
    end
    return nothing
end

@inline function _propagate_constraints!(s::_BasicScratch{T}, j::Int, v::T) where {T}
    A = s.A
    @inbounds for p in nzrange(A, j)
        i = A.rowval[p]
        s.con_keep[i] || continue
        contrib = A.nzval[p] * v
        s.lcon[i] -= contrib
        s.ucon[i] -= contrib
        s.row_scale[i] += abs(contrib)
    end
    return nothing
end


# Rows with (-Inf, Inf) bounds never bind: drop them. Their multiplier is
# exactly zero, so recovery needs no record.
function _pass_free_rows!(s::_BasicScratch{T}) where {T}
    n_removed = 0
    @inbounds for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        (s.lcon[i] == -T(Inf) && s.ucon[i] == T(Inf)) || continue
        s.con_keep[i] = false
        n_removed += 1
    end
    return n_removed
end


# Drop active rows whose entries are all in dropped columns or are zero.
# Returns the count removed; or `:infeasible` if a surviving row's bounds are
# inconsistent (lcon > 0 or ucon < 0, beyond the rounding folded into them).
function _pass_empty_rows!(s::_BasicScratch{T}) where {T}
    At = s.At
    n_removed = 0
    @inbounds for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        active = false
        for p in nzrange(At, i)
            if s.var_keep[At.rowval[p]] && !iszero(At.nzval[p])
                active = true;
                break
            end
        end
        active && continue
        tol = _row_tol(s, i)
        (s.lcon[i] > tol || s.ucon[i] < -tol) && return :infeasible
        s.con_keep[i] = false
        n_removed += 1
    end
    return n_removed
end


# Variables with no active row entry: fix at the bound/stationary point that
# optimizes `c[j]·x[j] + ½·Q[j,j]·x[j]²` over `[lvar[j], uvar[j]]`, respecting
# the model objective sense. With Q[j,j] = 0 this collapses to the LP rule
# (sign of c[j]). Returns `:unbounded` if the optimum is at ±∞.
function _pass_empty_cols!(s::_BasicScratch{T}) where {T}
    A = s.A
    n_fixed = 0
    for j in eachindex(s.var_keep)
        s.var_keep[j] || continue
        active = false
        @inbounds for p in nzrange(A, j)
            if s.con_keep[A.rowval[p]] && !iszero(A.nzval[p])
                active = true;
                break
            end
        end
        active && continue
        v_or_status = _empty_col_value(s, j)
        v_or_status === :unbounded && return :unbounded
        _eliminate_var!(s, j, v_or_status::T)
        n_fixed += 1
    end
    return n_fixed
end

# Minimizer of `c[j]·x + ½·q_jj·x²` on `[l, u]`:
#  - q_jj > 0  → unconstrained min `x* = -c[j]/q_jj`, then clamp.
#  - q_jj == 0 → linear: lower if c[j]>0, upper if c[j]<0, any feasible if c[j]==0.
#  - q_jj < 0  → concave: min at one of the bounds (or unbounded if both ±Inf).
# Minimizer of `c[j]·x + ½·q_jj·x²` on `[l, u]`:
#  - q_jj > 0  → unconstrained min `x* = -c[j]/q_jj`, then clamp.
#  - q_jj == 0 → linear: lower if c[j]>0, upper if c[j]<0, any feasible if c[j]==0.
#  - q_jj < 0  → concave: min at one of the bounds (or unbounded if both ±Inf).
function _empty_col_value(s::_BasicScratch{T}, j::Int) where {T}
    return s.minimize ? _empty_col_min_value(s, j) : _empty_col_max_value(s, j)
end

function _empty_col_min_value(s::_BasicScratch{T}, j::Int) where {T}
    l, u = s.lvar[j], s.uvar[j]
    cj = s.c[j]
    q_jj = _q_diag(s, j)

    if q_jj > 0
        x_star = -cj / q_jj
        return clamp(x_star, l, u)
    elseif q_jj == 0
        if cj > 0
            l == -Inf && return :unbounded
            return l
        elseif cj < 0
            u == Inf && return :unbounded
            return u
        else
            return clamp(zero(T), l, u)
        end
    else  # q_jj < 0 (concave)
        l == -Inf && u == Inf && return :unbounded
        if l == -Inf
            return u
        elseif u == Inf
            return l
        end
        obj_l = cj * l + T(0.5) * q_jj * l * l
        obj_u = cj * u + T(0.5) * q_jj * u * u
        return obj_l <= obj_u ? l : u
    end
end

# Maximizer of `c[j]·x + ½·q_jj·x²` on `[l, u]`:
#  - q_jj < 0  → unconstrained max `x* = -c[j]/q_jj`, then clamp.
#  - q_jj == 0 → linear: upper if c[j]>0, lower if c[j]<0, any feasible if c[j]==0.
#  - q_jj > 0  → convex: max at one of the bounds (or unbounded if either side is ±Inf).
function _empty_col_max_value(s::_BasicScratch{T}, j::Int) where {T}
    l, u = s.lvar[j], s.uvar[j]
    cj = s.c[j]
    q_jj = _q_diag(s, j)

    if q_jj < 0
        x_star = -cj / q_jj
        return clamp(x_star, l, u)
    elseif q_jj == 0
        if cj > 0
            u == Inf && return :unbounded
            return u
        elseif cj < 0
            l == -Inf && return :unbounded
            return l
        else
            return clamp(zero(T), l, u)
        end
    else  # q_jj > 0 (convex)
        (l == -Inf || u == Inf) && return :unbounded
        obj_l = cj * l + T(0.5) * q_jj * l * l
        obj_u = cj * u + T(0.5) * q_jj * u * u
        return obj_l >= obj_u ? l : u
    end
end

_q_diag(s::_BasicScratch{T}, j::Int) where {T} =
    s.Q === nothing ? zero(T) : _csc_diag(s.Q, j)
function _csc_diag(Q::SparseMatrixCSC{T}, j::Int) where {T}
    @inbounds for p in nzrange(Q, j)
        Q.rowval[p] == j && return Q.nzval[p]
    end
    return zero(T)
end


# Does x_j still appear in a quadratic term? Cross terms with eliminated
# variables don't count: their contribution was already folded into c[j] by
# `_propagate_quadratic!` when the partner was fixed.
function _q_involved(s::_BasicScratch{T}, j::Int) where {T}
    s.Q === nothing && return false
    Q = s.Q
    # Column j: stored entries are (k, j) with k >= j.
    @inbounds for p in nzrange(Q, j)
        k = Q.rowval[p]
        iszero(Q.nzval[p]) && continue
        (k == j || s.var_keep[k]) && return true
    end
    # Row j via reflection: entry (j, k) for k < j is stored in column k.
    @inbounds for k = 1:(j-1)
        s.var_keep[k] || continue
        for p in nzrange(Q, k)
            Q.rowval[p] == j && !iszero(Q.nzval[p]) && return true
        end
    end
    return false
end

# A free variable x_j appearing in exactly one active row i (pivot a_ij) and
# no quadratic term can absorb any activity of that row, so both leave the
# problem. Substituting x_j = (t - Σ_{l≠j} a_il x_l)/a_ij with row activity t
# turns c[j]·x_j into yi·t - Σ yi·a_il·x_l where yi = c[j]/a_ij, so
# c0 += yi·t and c[l] -= yi·a_il. The optimal t sits on the row bound favored
# by the objective sense; if that bound is infinite the problem is unbounded.
function _pass_free_singleton_cols!(s::_BasicScratch{T}) where {T}
    A, At = s.A, s.At
    n_eliminated = 0
    for j in eachindex(s.var_keep)
        s.var_keep[j] || continue
        (s.lvar[j] == -T(Inf) && s.uvar[j] == T(Inf)) || continue
        row_i = 0
        aij = zero(T)
        n_active = 0
        @inbounds for p in nzrange(A, j)
            i = A.rowval[p]
            s.con_keep[i] || continue
            a = A.nzval[p]
            iszero(a) && continue
            n_active += 1
            n_active > 1 && break
            row_i = i
            aij = a
        end
        n_active == 1 || continue
        abs(aij) > sqrt(eps(T)) || continue  # skip near-zero pivots
        _q_involved(s, j) && continue

        yi = s.c[j] / aij
        if iszero(yi)
            # x_j only pads the row: any finite activity works.
            t = clamp(zero(T), s.lcon[row_i], s.ucon[row_i])
            isfinite(t) || continue
        else
            bind_low = s.minimize ? (yi > 0) : (yi < 0)
            t = bind_low ? s.lcon[row_i] : s.ucon[row_i]
            isfinite(t) || return :unbounded
        end

        row_idx = Int[]
        row_val = T[]
        @inbounds for p in nzrange(At, row_i)
            l = At.rowval[p]
            a_il = At.nzval[p]
            (l != j && s.var_keep[l] && !iszero(a_il)) || continue
            push!(row_idx, l)
            push!(row_val, a_il)
            s.c[l] -= yi * a_il
        end
        s.c0 += yi * t
        s.var_keep[j] = false
        s.con_keep[row_i] = false
        push!(
            s.free_col_ops,
            _FreeSingletonColOp{T}(row_i, j, aij, t, yi, row_idx, row_val),
        )
        n_eliminated += 1
    end
    return n_eliminated
end


# ---- Top-level driver -------------------------------------------------------

apply_presolve(p::BasicPresolver, model::ScalarModel) = _basic_apply(p, model)

function _basic_apply(p::BasicPresolver, model::ScalarModel)
    s = _scratch(model)
    total_fixed = 0
    total_removed = 0
    total_eliminated = 0
    total_relaxed = 0

    for pass = 1:p.max_passes
        pass_fixed = 0
        pass_removed = 0
        pass_eliminated = 0
        pass_forced = 0
        pass_relaxed = 0

        _pass_bounds!(s) === :infeasible && return PRESOLVE_INFEASIBLE, nothing

        r = _pass_singleton_rows!(s)
        r === :infeasible && return PRESOLVE_INFEASIBLE, nothing
        pass_removed += r::Int

        r = _pass_row_activity!(s)
        r === :infeasible && return PRESOLVE_INFEASIBLE, nothing
        n_forced, n_redundant, n_relaxed = r::Tuple{Int,Int,Int}
        pass_forced += n_forced
        pass_removed += n_forced + n_redundant
        pass_relaxed += n_relaxed

        # Forcing rows leave their variables with lvar == uvar: eliminate them now.
        pass_fixed += _pass_fixed_vars!(s)
        pass_removed += _pass_free_rows!(s)

        r = _pass_empty_rows!(s)
        r === :infeasible && return PRESOLVE_INFEASIBLE, nothing
        pass_removed += r::Int

        r = _pass_empty_cols!(s)
        r === :unbounded && return PRESOLVE_UNBOUNDED, nothing
        pass_fixed += r::Int

        r = _pass_free_singleton_cols!(s)
        r === :unbounded && return PRESOLVE_UNBOUNDED, nothing
        pass_eliminated += r::Int

        total_fixed += pass_fixed
        total_removed += pass_removed
        total_eliminated += pass_eliminated
        total_relaxed += pass_relaxed

        progress =
            pass_fixed > 0 || pass_removed > 0 || pass_eliminated > 0 || pass_relaxed > 0
        if p.verbose && progress
            msg =
                "BasicPresolver pass $pass: fixed $pass_fixed variable(s), " *
                "removed $pass_removed constraint(s) ($pass_forced forcing), " *
                "relaxed $pass_relaxed row bound(s), " *
                "eliminated $pass_eliminated free singleton column(s)"
            @info msg
        end

        progress || break
    end

    if !any(s.var_keep)
        solved = _build_solved(s)
        solved === :infeasible && return PRESOLVE_INFEASIBLE, nothing
        return PRESOLVE_SOLVED, solved::BasicSolvedResult
    end

    if total_fixed == 0 && total_removed == 0 && total_eliminated == 0 && total_relaxed == 0
        return PRESOLVE_UNCHANGED, NoPresolveResult(model)
    end

    return PRESOLVE_REDUCED, _build_reduced(model, s)
end


# ---- Reduced-model materialization -----------------------------------------

function _build_reduced(model::ScalarModel, s::_BasicScratch{T}) where {T}
    var_map = findall(s.var_keep)
    con_map = findall(s.con_keep)
    c_red = s.c[var_map]
    lvar_red = s.lvar[var_map]
    uvar_red = s.uvar[var_map]
    lcon_red = s.lcon[con_map]
    ucon_red = s.ucon[con_map]
    A_red = _slice_csc(s.A, con_map, var_map)
    A_src = _match_source(operator_sparse_matrix(model.data.A), A_red)
    reduced = _construct_reduced(
        model,
        s,
        A_src,
        c_red,
        lvar_red,
        uvar_red,
        lcon_red,
        ucon_red,
        var_map,
    )
    return BasicPresolveResult{T,typeof(reduced),typeof(s.Q)}(
        reduced,
        var_map,
        con_map,
        s.fixed_idx,
        s.fixed_val,
        s.free_col_ops,
        s.forcing_ops,
        s.minimize,
        s.c_orig,
        s.A,
        s.Q,
        length(s.var_keep),
        length(s.con_keep),
    )
end

_slice_csc(A::SparseMatrixCSC, rows::Vector{Int}, cols::Vector{Int}) = A[rows, cols]

function _build_solved(s::_BasicScratch{T}) where {T}
    @inbounds for i in eachindex(s.con_keep)
        s.con_keep[i] || continue
        tol = _row_tol(s, i)
        (s.lcon[i] > tol || s.ucon[i] < -tol) && return :infeasible
    end
    return BasicSolvedResult{T,typeof(s.Q)}(
        copy(s.fixed_idx),
        copy(s.fixed_val),
        copy(s.free_col_ops),
        copy(s.forcing_ops),
        s.minimize,
        s.c_orig,
        s.A,
        s.Q,
        length(s.var_keep),
        length(s.con_keep),
        s.c0,
    )
end

_match_source(::SparseMatrixCSC, A_red::SparseMatrixCSC) = A_red
function _match_source(::SparseMatrixCOO, A_red::SparseMatrixCSC{T}) where {T}
    rows, cols, vals = findnz(A_red)
    return SparseMatrixCOO(size(A_red)..., rows, cols, vals)
end

function _construct_reduced(
    model::LinearModel{T},
    s::_BasicScratch{T},
    A_src,
    c_red,
    lvar_red,
    uvar_red,
    lcon_red,
    ucon_red,
    _,
) where {T}
    data = LPData(
        A_src,
        c_red;
        lcon = lcon_red,
        ucon = ucon_red,
        lvar = lvar_red,
        uvar = uvar_red,
        c0 = s.c0,
    )
    return LinearModel(data; minimize = model.meta.minimize, name = model.meta.name)
end

function _construct_reduced(
    model::QuadraticModel{T},
    s::_BasicScratch{T},
    A_src,
    c_red,
    lvar_red,
    uvar_red,
    lcon_red,
    ucon_red,
    var_map,
) where {T}
    Q_red = _slice_csc(s.Q::SparseMatrixCSC{T,Int}, var_map, var_map)
    Q_src = _match_source(operator_sparse_matrix(model.data.Q), Q_red)
    data = QPData(
        A_src,
        c_red,
        Q_src;
        lcon = lcon_red,
        ucon = ucon_red,
        lvar = lvar_red,
        uvar = uvar_red,
        c0 = s.c0,
    )
    return QuadraticModel(data; minimize = model.meta.minimize, name = model.meta.name)
end


# ---- Solution mapping -------------------------------------------------------

function recover_solution(
    r::BasicPresolveResult{T},
    x_red::AbstractVector,
    y_red::AbstractVector,
) where {T}
    x = zeros(T, r.n_orig)
    y = zeros(T, r.m_orig)
    @inbounds x[r.var_map] .= x_red
    @inbounds y[r.con_map] .= y_red
    @inbounds for (j, v) in zip(r.fixed_var_idx, r.fixed_var_val)
        x[j] = v
    end
    _replay_free_col_ops!(x, y, r.free_col_ops)
    _replay_forcing_ops!(x, y, r)
    return x, y
end

function recover_solution(
    r::BasicSolvedResult{T},
    ::AbstractVector,
    ::AbstractVector,
) where {T}
    x = zeros(T, r.n_orig)
    y = zeros(T, r.m_orig)
    @inbounds for (j, v) in zip(r.fixed_var_idx, r.fixed_var_val)
        x[j] = v
    end
    _replay_free_col_ops!(x, y, r.free_col_ops)
    _replay_forcing_ops!(x, y, r)
    return x, y
end

# Multiplier of a forcing row i from the reduced costs of the variables it
# pinned. With r_j = c_j + (Q x)_j - Σ_{k ≠ i} a_kj y_k, stationarity reads
# r_j - a_ij y_i = z_j^l - z_j^u; a variable pinned at its upper bound needs
# r_j - a_ij y_i <= 0 and one at its lower bound >= 0. For a row forced to lcon
# that is y_i >= r_j / a_ij for every pinned j (symmetrically <= for ucon), so
# y_i is the largest (smallest) ratio, clamped to the sign an inequality side
# requires. Rows are replayed newest-first so each multiplier is formed against
# the rows that were still present when the row was eliminated: rows dropped
# earlier still carry y = 0 at that point, while rows dropped by free-singleton
# column ops carry their constant multiplier, which is exactly the cost
# adjustment those ops folded into `c`. Requires x to be fully recovered.
# A maximization is handled as the minimization of -f: the ratios and the
# sign clamp are formed for that problem and the multiplier is flipped back,
# matching the `c[j]/a[i,j]` convention of the free-singleton-column ops.
function _replay_forcing_ops!(
    x::AbstractVector{T},
    y::AbstractVector{T},
    r::Union{BasicPresolveResult{T},BasicSolvedResult{T}},
) where {T}
    ops = r.forcing_ops
    isempty(ops) && return nothing
    A = r.A
    σ = r.minimize ? one(T) : -one(T)
    qx = r.Q === nothing ? nothing : Symmetric(r.Q, :L) * x
    @inbounds for k = length(ops):-1:1
        op = ops[k]
        yi = op.at_lower ? T(-Inf) : T(Inf)
        for (j, a) in zip(op.col_idx, op.col_val)
            rc = r.c[j]
            qx === nothing || (rc += qx[j])
            for p in nzrange(A, j)
                i = A.rowval[p]
                i == op.i && continue
                rc -= A.nzval[p] * y[i]
            end
            ratio = σ * rc / a
            yi = op.at_lower ? max(yi, ratio) : min(yi, ratio)
        end
        isempty(op.col_idx) && (yi = zero(T))
        if !op.equality
            yi = op.at_lower ? max(yi, zero(T)) : min(yi, zero(T))
        end
        y[op.i] = σ * yi
    end
    return nothing
end

# Back-substitute free-singleton-column eliminations in reverse chronological
# order: a stored row may reference variables removed by a *later* op, whose
# values must be recovered first. Variables fixed before the op were already
# folded into `activity`, so they never appear in `row_idx`.
function _replay_free_col_ops!(x, y, ops::Vector{_FreeSingletonColOp{T}}) where {T}
    @inbounds for k = length(ops):-1:1
        op = ops[k]
        acc = zero(T)
        for (l, a) in zip(op.row_idx, op.row_val)
            acc += a * x[l]
        end
        x[op.j] = (op.activity - acc) / op.aij
        y[op.i] = op.yi
    end
    return nothing
end
