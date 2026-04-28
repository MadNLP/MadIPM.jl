mutable struct BatchView{VI32}
    batch_size_root::Int
    layer::Int
    n::Int
    local_to_root::Vector{Int32}
    local_to_slot::Vector{Int32}
    local_to_root_dev::VI32
end

mutable struct BatchViewState{V<:BatchView}
    views::Vector{V}
    active_layer::Int
    selected_local_buffer::Vector{Int32}
end

batch_size_root(view::BatchView) = view.batch_size_root
view_layer(view::BatchView) = view.layer
local_batch_size(view::BatchView) = view.n
local_to_root_dev(view::BatchView) = view.local_to_root_dev
active_view(state::BatchViewState) = state.views[state.active_layer]
root_view(state::BatchViewState) = state.views[1]

function is_identity_view(view::BatchView)
    view.n == view.batch_size_root || return false
    @inbounds for i in 1:view.n
        view.local_to_root[i] == i || return false
    end
    return true
end

function _sync_local_to_root_dev!(view::BatchView)
    copyto!(view.local_to_root_dev, view.local_to_root)
    return view
end

function _init_batch_view!(view::BatchView, batch_size::Int)
    view.n = batch_size
    @inbounds for i in 1:batch_size
        idx = Int32(i)
        view.local_to_root[i] = idx
        view.local_to_slot[i] = idx
    end
    return _sync_local_to_root_dev!(view)
end

function _select_local!(child::BatchView, parent::BatchView, selected_local, nselected::Int, reset_slots::Bool)
    child.n = nselected
    roots = parent.local_to_root
    slots = parent.local_to_slot
    @inbounds for j in 1:nselected
        parent_j = Int(selected_local[j])
        child.local_to_root[j] = roots[parent_j]
        child.local_to_slot[j] = reset_slots ? Int32(j) : slots[parent_j]
    end
    return _sync_local_to_root_dev!(child)
end

function reset_active_view!(state::BatchViewState)
    state.active_layer = 1
    return state
end

function restore_state!(state::BatchViewState, old_state::BatchView)
    state.active_layer = view_layer(old_state)
    return state
end

function select_local!(
    state::BatchViewState,
    selected_local::AbstractVector{<:Integer},
    nselected::Int = length(selected_local);
    reset_slots::Bool = false,
)
    parent = active_view(state)
    layer = view_layer(parent) + 1
    @assert layer <= length(state.views) "BatchViewState max_layers exceeded"
    old_state = parent
    state.active_layer = layer
    _select_local!(active_view(state), old_state, selected_local, nselected, reset_slots)
    return old_state
end

function select_local!(
    state::BatchViewState,
    keep::AbstractVector{Bool};
    reset_slots::Bool = false,
)
    current = active_view(state)
    @assert length(keep) == current.n
    nselected = 0
    @inbounds for i in 1:current.n
        if keep[i]
            nselected += 1
            state.selected_local_buffer[nselected] = i
        end
    end
    return select_local!(state, state.selected_local_buffer, nselected; reset_slots=reset_slots)
end

function exclude_local!(state::BatchViewState, exclude_mask::AbstractVector{Bool})
    current = active_view(state)
    @assert length(exclude_mask) == current.n
    nselected = 0
    @inbounds for i in 1:current.n
        if !exclude_mask[i]
            nselected += 1
            state.selected_local_buffer[nselected] = i
        end
    end
    return select_local!(state, state.selected_local_buffer, nselected)
end

function BatchViewState(bcb, batch_size::Int; max_layers::Int = 4)
    sample_dev = MadNLP.create_array(bcb, Int32, batch_size)
    views = Vector{BatchView{typeof(sample_dev)}}(undef, max_layers)
    @inbounds for layer in 1:max_layers
        views[layer] = BatchView(
            batch_size,
            layer,
            0,
            Vector{Int32}(undef, batch_size),
            Vector{Int32}(undef, batch_size),
            layer == 1 ? sample_dev : MadNLP.create_array(bcb, Int32, batch_size),
        )
    end
    _init_batch_view!(views[1], batch_size)
    return BatchViewState(views, 1, Vector{Int32}(undef, batch_size))
end

function fill_batch_view_mask!(mask::AbstractMatrix{T}, batch_view::BatchView) where T
    @assert size(mask, 1) == 1
    @assert size(mask, 2) == batch_view.batch_size_root
    fill!(mask, zero(T))
    @inbounds for j in 1:batch_view.n
        mask[1, batch_view.local_to_root[j]] = one(T)
    end
    return mask
end

function gather_batch_view_columns!(
    dst::AbstractMatrix{TD},
    src::AbstractMatrix{TS},
    batch_view::BatchView,
) where {TD, TS}
    roots = batch_view.local_to_root
    @inbounds for j in 1:batch_view.n
        copyto!(view(dst, :, j), view(src, :, roots[j]))
    end
    return dst
end

function compact_active_columns_inplace!(dst::AbstractMatrix{T}, batch_view::BatchView, scratch=nothing) where T
    roots = batch_view.local_to_root
    @inbounds for j in 1:batch_view.n
        src_j = roots[j]
        src_j < j && error("active view must be root-ordered")
        src_j != j && copyto!(view(dst, :, j), view(dst, :, src_j))
    end
    return dst
end

function scatter_batch_view_columns!(
    dst::AbstractMatrix{TD},
    src::AbstractMatrix{TS},
    batch_view::BatchView,
) where {TD, TS}
    roots = batch_view.local_to_root
    @inbounds for j in 1:batch_view.n
        copyto!(view(dst, :, roots[j]), view(src, :, j))
    end
    return dst
end
