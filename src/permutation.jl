function permutation_schur(qp::QuadraticModel)
  n = qp.meta.nvar
  m = qp.meta.ncon
  p_AtA = AMD.symamd(qp.data.A)
  p = Vector{Cint}(undef, n+m)
  for i = 1:n
    p[i] = i
  end
  for i = n+1:n+m
    p[i] = p_AtA[i-n] + n
  end
  return p
end
