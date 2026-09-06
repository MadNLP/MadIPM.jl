#!/usr/bin/env bash
# Start benchmarks on the box set up by setup_pod.sh, detached, with one log
# per process under ~/bench and a DONE marker at the end. Each process logs
# "[ Info: <CASE>" as it starts an instance, which is what to tail for
# progress:  ssh <host> tail -f bench/2-benchmark-netlib-gpu.log
#
#   run_pod.sh                       every script, GPU side and CPU side as two
#                                    concurrent processes (the CPU baseline
#                                    never leaves the GPU idle), then merged
#   run_pod.sh <script.jl>           both sides of one script, concurrently
#   run_pod.sh <script.jl> <args...> one process with exactly these arguments
#
# CPU_ARGS (default "--cpu-max-batch=4 --cpu-time-budget=300") is appended to
# the CPU side. Stop a run by PID (~/bench/*.pid); never `pkill -f julia`
# over ssh.
set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$HOME/MadIPM/benchmark"
mkdir -p "$HOME/bench"
rm -f "$HOME/bench/DONE"

export CPU_ARGS=${CPU_ARGS:-"--cpu-max-batch=4 --cpu-time-budget=300"}

# both_sides <stem> <script> [common args...]: GPU and CPU processes side by
# side, then merge the two CSVs.
both_sides() {
    local stem=$1 script=$2; shift 2
    julia --project=. "$script" "$@" --side=gpu > ~/bench/$stem-gpu.log 2>&1 &
    julia --project=. "$script" "$@" --side=cpu $CPU_ARGS > ~/bench/$stem-cpu.log 2>&1 &
    wait
    julia --project=. merge_results.jl results/$stem >> ~/bench/merge.log 2>&1 || true
}
export -f both_sides

stem_of() {
    case "$1" in
        1-scalability.jl) echo 1-scalability-netlib ;;
        2-performance.jl) echo 2-benchmark-netlib ;;
        3-dcopf.jl)       echo 3-benchmark-dcopf ;;
        *) echo "unknown script $1" >&2; exit 1 ;;
    esac
}
extra_of() {
    case "$1" in
        2-performance.jl) echo "--benchmark=netlib" ;;
        3-dcopf.jl)       echo "--job=benchmark" ;;
        *) echo "" ;;
    esac
}

if [ $# -eq 0 ]; then
    nohup bash -c '
        both_sides 2-benchmark-netlib 2-performance.jl --benchmark=netlib
        both_sides 1-scalability-netlib 1-scalability.jl
        echo "ALL DONE $(date -u +%FT%TZ)" > ~/bench/DONE
    ' > "$HOME/bench/runner.log" 2>&1 &
    echo $! > "$HOME/bench/runner.pid"
    echo "started runner, pid $(cat "$HOME/bench/runner.pid"), logs in ~/bench"
elif [ $# -eq 1 ]; then
    script=$1; stem=$(stem_of "$script"); extra=$(extra_of "$script")
    nohup bash -c "both_sides $stem $script $extra; echo \"DONE \$(date -u +%FT%TZ)\" > ~/bench/DONE" > "$HOME/bench/runner.log" 2>&1 &
    echo $! > "$HOME/bench/runner.pid"
    echo "started both sides of $script, runner pid $(cat "$HOME/bench/runner.pid"), logs ~/bench/$stem-{gpu,cpu}.log"
else
    script=$1; shift
    log="$HOME/bench/${script%.jl}.log"
    nohup bash -c "julia --project=. $script $* > $log 2>&1; echo \"DONE \$(date -u +%FT%TZ)\" > ~/bench/DONE" > /dev/null 2>&1 &
    echo $! > "$HOME/bench/${script%.jl}.pid"
    echo "started $script $*, pid $(cat "$HOME/bench/${script%.jl}.pid"), log $log"
fi
