struct NormalKKTSystem{T, VT, MT, VI, VI32, LS} <: MadNLP.AbstractKKTSystem{T, VT, MT, MadNLP.ExactHessian{T, VT}}
    # Augmented system
    aug_com::MT
    # Jacobian
    A::MadNLP.SparseMatrixCOO{T,Int32,VT, VI32}
    AT::MT
    A_csr_map::Union{Nothing, VI}
    jac::VT

    reg::VT
    pr_diag::VT
    du_diag::VT
    l_diag::VT
    u_diag::VT
    l_lower::VT
    u_lower::VT
    buffer_n::VT
    buffer_m::VT
    # LinearSolver
    linear_solver::LS
    # Info
    ind_ineq::VI
    ind_lb::VI
    ind_ub::VI
    n::Int
    m::Int
end

function MadNLP.create_kkt_system(
    ::Type{NormalKKTSystem},
    cb::MadNLP.SparseCallback{T,VT},
    ind_cons,
    linear_solver::Type;
    opt_linear_solver=MadNLP.default_options(linear_solver),
    hessian_approximation=MadNLP.ExactHessian,
    qn_options=MadNLP.QuasiNewtonOptions(),
) where {T,VT}
    n = cb.nvar
    m = cb.ncon
    ind_ineq = ind_cons.ind_ineq
    n_slack = length(ind_ineq)
    nlb = length(ind_cons.ind_lb)
    nub = length(ind_cons.ind_ub)

    if cb.nnzh > 0
        error("The KKT system NormalKKTSystem supports only linear programs.
               The problem has a positive number of nonzero in its Hessian.")
    end

    # Evaluate sparsity pattern
    jac_sparsity_I = MadNLP.create_array(cb, Int32, cb.nnzj)
    jac_sparsity_J = MadNLP.create_array(cb, Int32, cb.nnzj)
    MadNLP._jac_sparsity_wrapper!(cb, jac_sparsity_I, jac_sparsity_J)

    nnzj = length(jac_sparsity_I)
    ntot = n + n_slack

    reg     = VT(undef, ntot)
    pr_diag = VT(undef, ntot)
    du_diag = VT(undef, m)
    l_diag  = VT(undef, nlb)
    u_diag  = VT(undef, nub)
    l_lower = VT(undef, nlb)
    u_lower = VT(undef, nub)

    buffer_n = VT(undef, ntot)
    buffer_m = VT(undef, m)

    # Build Jacobian with slack variable
    I = MadNLP.create_array(cb, Int32, nnzj + n_slack)
    J = MadNLP.create_array(cb, Int32, nnzj + n_slack)
    V = VT(undef, nnzj + n_slack)
    # Assemble sparsity pattern
    I[1:nnzj] .= jac_sparsity_I
    J[1:nnzj] .= jac_sparsity_J
    I[nnzj+1:nnzj+n_slack] .= ind_ineq
    J[nnzj+1:nnzj+n_slack] .= (n+1:n+n_slack)
    # Build final COO matrix
    A_coo = MadNLP.SparseMatrixCOO(m, ntot, I, J, V)
    # Get a view to build original LHS
    jac = MadNLP._madnlp_unsafe_wrap(V, nnzj, 1)

    # Fill values with continuous range to get mapping
    A_coo.V .= 1:(nnzj+n_slack)
    # Convert Jacobian to CSR
    Ap, Aj, Ax = coo_to_csr(A_coo)

    A_csr_map = convert.(Int, Ax)

    # Store transposed matrix in CSC format
    CSC = sparse_csc_format(VT)
    AT = CSC <: SparseArrays.SparseMatrixCSC ? CSC(ntot, m, Ap, Aj, Ax) : CSC(Ap, Aj, Ax, (ntot, m))

    # Assemble normal KKT system in CSR format
    if CSC <: SparseArrays.SparseMatrixCSC
        AAp, AAj = build_normal_system(m, ntot, Ap, Aj)
    else
        # Target sparse gemm in CUSPARSE
        # S = SparseArrays.sparse(transpose(AT)) * AT
        # S = tril(S)
        # AAp = S.colPtr
        # AAj = S.rowVal

        AAp_h, AAj_h = build_normal_system(m, ntot, Vector(Ap), Vector(Aj))
        VII = typeof(Ap)
        AAp = VII(AAp_h)
        AAj = VII(AAj_h)
    end
    AAx = VT(undef, length(AAj))

    aug_com = CSC <: SparseArrays.SparseMatrixCSC ? CSC(m, m, AAp, AAj, AAx) : CSC(AAp, AAj, AAx, (m, m))

    _linear_solver = linear_solver(
        aug_com; opt = opt_linear_solver
    )

    fill!(jac, zero(T))

    return NormalKKTSystem(
        aug_com,
        A_coo,
        AT,
        A_csr_map,
        jac,
        reg,
        pr_diag,
        du_diag,
        l_diag,
        u_diag,
        l_lower,
        u_lower,
        buffer_n,
        buffer_m,
        _linear_solver,
        ind_ineq,
        ind_cons.ind_lb,
        ind_cons.ind_ub,
        ntot, m,
    )
end

MadNLP.num_variables(kkt::NormalKKTSystem) = length(kkt.pr_diag)
MadNLP.get_jacobian(kkt::NormalKKTSystem) = kkt.jac
MadNLP.get_hessian(kkt::NormalKKTSystem) = Float64[]

function MadNLP.is_inertia_correct(kkt::NormalKKTSystem, num_pos, num_zero, num_neg)
    return (num_zero == 0) && (num_pos == kkt.m)
end

function MadNLP.initialize!(kkt::NormalKKTSystem{T}) where T
    fill!(kkt.reg, one(T))
    fill!(kkt.pr_diag, one(T))
    fill!(kkt.du_diag, zero(T))
    fill!(kkt.l_lower, zero(T))
    fill!(kkt.u_lower, zero(T))
    fill!(kkt.l_diag, one(T))
    fill!(kkt.u_diag, one(T))
    fill!(kkt.buffer_m, zero(T))
    fill!(kkt.buffer_n, zero(T))
    return
end

function MadNLP.compress_jacobian!(kkt::NormalKKTSystem)
    n_slack = length(kkt.ind_ineq)
    kkt.A.V[end-n_slack+1:end] .= -1.0
    # Transfer to the matrix A stored in CSC format
    fill!(kkt.AT.nzval, 0.0)
    for i in eachindex(kkt.A_csr_map)
        kkt.AT.nzval[i] = kkt.A.V[kkt.A_csr_map[i]]
    end
    return
end

MadNLP.compress_hessian!(kkt::NormalKKTSystem) = nothing

function MadNLP.jtprod!(y::AbstractVector, kkt::NormalKKTSystem, x::AbstractVector)
    return mul!(y, kkt.AT, x)
end

function MadNLP.build_kkt!(kkt::NormalKKTSystem)
    m, n = kkt.m, kkt.n
    D = kkt.buffer_n
    Cp = _colptr(kkt.aug_com)
    Cj = _rowval(kkt.aug_com)
    Cx = _nzval(kkt.aug_com)
    Ap = _colptr(kkt.AT)
    Aj = _rowval(kkt.AT)
    Ax = _nzval(kkt.AT)

    # Build normal matrix A Σ⁻¹ Aᵀ
    D .= 1.0 ./ kkt.pr_diag
    assemble_normal_system!(m, n, Ap, Aj, Ax, Cp, Cj, Cx, D)
    return
end

function MadNLP.solve!(kkt::NormalKKTSystem, w::MadNLP.AbstractKKTVector)
    MadNLP.reduce_rhs!(w.xp_lr, MadNLP.dual_lb(w), kkt.l_diag, w.xp_ur, MadNLP.dual_ub(w), kkt.u_diag)
    r1 = kkt.buffer_n
    r2 = kkt.buffer_m
    Σ = kkt.pr_diag

    wx = MadNLP.primal(w)
    wy = MadNLP.dual(w)

    # Build RHS
    r1 .= wx ./ Σ                          # Σ⁻¹ r₁
    r2 .= wy                               # r₂
    mul!(r2, kkt.AT', r1, 1.0, -1.0)       # A Σ⁻¹ r₁ - r₂
    # Solve normal KKT system
    MadNLP.solve!(kkt.linear_solver, r2)   # Δy
    # Unpack solution
    wy .= r2                               # Δy
    r1 .= wx                               # r₁
    mul!(r1, kkt.AT, wy, -1.0, 1.0)        # r₁ - Aᵀ Δy
    wx .= r1 ./ Σ                          # Σ⁻¹ (r₁ - Aᵀ Δy)

    MadNLP.finish_aug_solve!(kkt, w)
    return w
end

function MadNLP.mul!(w::MadNLP.AbstractKKTVector{T}, kkt::NormalKKTSystem, v::MadNLP.AbstractKKTVector, alpha = one(T), beta = zero(T)) where {T}
    wx = MadNLP.primal(w)
    wy = MadNLP.dual(w)

    vx = MadNLP.primal(v)
    vy = MadNLP.dual(v)

    mul!(wx, kkt.AT, vy, alpha, beta)
    mul!(wy, kkt.AT', vx, alpha, beta)

    MadNLP._kktmul!(w,v,kkt.reg, kkt.du_diag, kkt.l_lower, kkt.u_lower, kkt.l_diag, kkt.u_diag, alpha, beta)
    return w
end
