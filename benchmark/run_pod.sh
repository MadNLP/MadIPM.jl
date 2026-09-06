#!/usr/bin/env bash
# Start a benchmark on the box set up by setup_pod.sh, detached, with one log
# per script under ~/bench and a DONE marker at the end. Each script logs
# "[ Info: <CASE>" as it starts an instance, which is what to tail for progress:
#   ssh root@<ip> tail -f bench/2-performance.log
#
#   run_pod.sh                          2-performance (netlib) then 1-scalability
#   run_pod.sh <script.jl> [args...]    one script with its own arguments
#
# Stop a run by PID (~/bench/*.pid); never `pkill -f julia` over ssh.
set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$HOME/MadIPM/benchmark"
mkdir -p "$HOME/bench"
rm -f "$HOME/bench/DONE"

if [ $# -eq 0 ]; then
    nohup bash -c '
        julia --project=. 2-performance.jl --benchmark=netlib --cpu-max-batch=4 --cpu-time-budget=300 > ~/bench/2-performance.log 2>&1
        julia --project=. 1-scalability.jl > ~/bench/1-scalability.log 2>&1
        echo "ALL DONE $(date -u +%FT%TZ)" > ~/bench/DONE
    ' > "$HOME/bench/runner.log" 2>&1 &
    echo $! > "$HOME/bench/runner.pid"
    echo "started runner, pid $(cat "$HOME/bench/runner.pid"), logs in ~/bench"
else
    script=$1; shift
    log="$HOME/bench/${script%.jl}.log"
    nohup bash -c "julia --project=. $script $* > $log 2>&1; echo DONE > ~/bench/DONE" > /dev/null 2>&1 &
    echo $! > "$HOME/bench/${script%.jl}.pid"
    echo "started $script, pid $(cat "$HOME/bench/${script%.jl}.pid"), log $log"
fi
