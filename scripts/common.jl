using QuadraticModels
using QPSReader
using GZip
using CodecBzip2
using HSL
using NLPModels
using SparseArrays

import QuadraticModels: SparseMatrixCOO

"""
    import_mps(filename::String)

Import instance from the file whose path is specified in `filename`.

The function parses the file's extension to adapt the import. If the extension
is `.mps`, `.sif` or `.SIF`, it directly reads the file. If the extension
is `.gz` or `.bz2`, it decompresses the file using gzip or bzip2, respectively.

"""
function import_mps(filename)
    ext = match(r"(.*)\.(.*)", filename).captures[2]
    data = if ext ∈ ("mps", "sif", "SIF")
        readqps(filename)
    elseif ext == "gz"
        GZip.open(filename, "r") do gz
            readqps(gz)
        end
    elseif ext == "bz2"
        open(filename, "r") do io
            stream = Bzip2DecompressorStream(io)
            readqps(stream)
        end
    end
    return data
end

function _scale_coo!(A, Dr, Dc)
    k = 1
    for (i, j) in zip(A.rows, A.cols)
        A.vals[k] = A.vals[k] / (Dr[i] * Dc[j])
        k += 1
    end
end

"""
    scale_qp(qp::QuadraticModel)

Scale QP using Ruiz' equilibration method.

The function scales the Jacobian ``A`` as ``As = Dr * A * Dc``, with ``As``
a matrix whose rows and columns have an infinite norm close to 1.

The scaling is computed using `HSL.mc77`, implementing the Ruiz equilibration method.

"""
function scale_qp(qp::QuadraticModel)
    A = qp.data.A
    m, n = size(A)

    if !LIBHSL_isfunctional()
        return qp
    end

    A_csc = sparse(A.rows, A.cols, A.vals, m, n)
    Dr, Dc = HSL.mc77(A_csc, 0)

    Hs = copy(qp.data.H)
    As = copy(qp.data.A)
    _scale_coo!(Hs, Dc, Dc)
    _scale_coo!(As, Dr, Dc)

    data = QuadraticModels.QPData(
        qp.data.c0,
        qp.data.c ./ Dc,
        qp.data.v,
        Hs,
        As,
    )

    return QuadraticModel(
        NLPModelMeta(
            qp.meta.nvar;
            ncon=qp.meta.ncon,
            lvar=qp.meta.lvar .* Dc,
            uvar=qp.meta.uvar .* Dc,
            lcon=qp.meta.lcon ./ Dr,
            ucon=qp.meta.ucon ./ Dr,
            x0=qp.meta.x0 .* Dc,
            y0=qp.meta.y0 ./ Dr,
            nnzj=qp.meta.nnzj,
            lin_nnzj=qp.meta.nnzj,
            lin=qp.meta.lin,
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        Counters(),
        data,
    )
end

