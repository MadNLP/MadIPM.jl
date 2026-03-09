abstract type AbstractBatchKKTSystem{T} end

@inbounds function _kktmul!(
    w::BatchUnreducedKKTVector,
    x::BatchUnreducedKKTVector,
    reg, du_diag_val, l_lower, u_lower, l_diag, u_diag,
    alpha, beta,
)
    MadNLP.primal(w) .+= alpha .* reg .* MadNLP.primal(x)
    MadNLP.dual(w) .+= alpha .* du_diag_val .* MadNLP.dual(x)
    xp_lr(w) .-= alpha .* MadNLP.dual_lb(x)
    xp_ur(w) .+= alpha .* MadNLP.dual_ub(x)
    MadNLP.dual_lb(w) .= beta .* MadNLP.dual_lb(w) .+ alpha .* (xp_lr(x) .* l_lower .- MadNLP.dual_lb(x) .* l_diag)
    MadNLP.dual_ub(w) .= beta .* MadNLP.dual_ub(w) .+ alpha .* (xp_ur(x) .* u_lower .+ MadNLP.dual_ub(x) .* u_diag)
    return
end

include("Sparse/augmented.jl")
