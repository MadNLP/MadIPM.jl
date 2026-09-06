# NVTX instrumentation for profiling. Installed into src/ by benchmark/profile.sh
# (not part of the package): every `function` definition in src/ (outside the
# CPU model code in src/models/) and ext/MadIPMCUDAExt/ is wrapped in an NVTX
# range named after the function, in a per-module domain, so Nsight Systems
# shows the solver's call structure over the CUDA timeline.
#
# By default the ranges only mark host-side spans, since GPU work is
# asynchronous. Set
#     MadIPM._PROFILE_SYNC_HOOK[] = CUDA.synchronize
# to synchronize on entry and exit of every annotated function, so each range
# covers exactly its own GPU work (at the cost of the overlap a normal run has).
const _PROFILE_SYNC_HOOK = Ref{Any}(() -> nothing)

_annotate_name(sig) =
    sig isa Expr && sig.head == :where ? _annotate_name(sig.args[1]) :
    sig isa Expr && sig.head == :call  ? string(sig.args[1]) :
    string(sig)

macro sync_annotate(ex)
    ok = ex isa Expr && ex.head == :function && length(ex.args) == 2 &&
         ex.args[1] isa Expr && ex.args[1].head in (:call, :where)
    ok || return esc(ex)   # e.g. `function f end`: nothing to wrap
    fsig, body = ex.args
    message = _annotate_name(fsig)
    color = hash(message) % UInt32
    mod = __module__
    quote
        $(esc(fsig)) = begin
            MadIPM._PROFILE_SYNC_HOOK[]()
            _isactive = NVTX.isactive()
            _rangeid = _isactive ? NVTX.range_start(NVTX.Domain($mod); message=$message, color=$color) : nothing
            try
                $(esc(body))
            finally
                MadIPM._PROFILE_SYNC_HOOK[]()
                _isactive && NVTX.range_end(_rangeid)
            end
        end
    end
end
