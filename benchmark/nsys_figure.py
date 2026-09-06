#!/usr/bin/env python3
"""Draw a time window of an Nsight Systems report as a GPU timeline figure.

    python nsys_figure.py profiles/CRE-B-bs128.sqlite --start 150.893 --end 151.310 -o crebs128.pdf

Times are seconds on the Nsight timeline ruler. Everything renders locally from
the SQLite export of the report (`REPORT.sqlite`, written by profile.sh next to
the report; or `nsys export --type sqlite REPORT.nsys-rep`; or File > Export in
the GUI). A `.nsys-rep` is accepted when `nsys` is on the PATH.

The top row is the GUI's "CUDA HW > MadIPM" view: NVTX ranges projected onto
the GPU, each spanning from the first to the last GPU operation (kernel,
memcpy, memset) launched by the CUDA API calls made inside the range on its
thread. Host-side range times are not used; they mostly measure where the host
waits for the asynchronous GPU. The row holds one level: by default the solver
phases in PHASES, whichever nesting level they sit at (`--ranges` picks other
names; `--depth N` shows one nesting level instead, e.g. `--depth 5` in a
MadIPM profile is the level with factorize_system!/prediction_step!/...).
Below it, one thin row each for host-to-device and device-to-host copies,
drawn as ticks since they last microseconds. Bars are
labelled inside when they fit and with a leader above the row otherwise; the
legend gives each phase's GPU time and share of the window. Needs matplotlib.
"""
import argparse
import bisect
import os
import shutil
import sqlite3
import subprocess
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# default row: one bar per call of these functions (module prefixes ignored)
PHASES = ["factorize_system!", "solve_system!", "update_termination_status!", "evaluate_model!"]
PHASE_COLORS = {                          # light fills, so the black labels stay readable
    "factorize_system!": "#9ecae1",
    "solve_system!": "#a1d99b",
    "update_termination_status!": "#bcbddc",
    "evaluate_model!": "#fdae6b",
}
FALLBACK_CMAP = "Set2"                    # other range names
IDLE_COLOR = "#ececec"
MEMORY_ROWS = [(1, "HtoD", "#d62728"), (2, "DtoH", "#7b3294")]   # copyKind ids of the export   # copyKind ids of the export


# ---------------------------------------------------------------- database ----

def sqlite_path(report):
    if report.endswith(".sqlite"):
        if not os.path.exists(report):
            sys.exit(f"{report}: no such file (the .sqlite exports of the reports live next to the .nsys-rep files)")
        return report
    stem, _ = os.path.splitext(report)
    db = stem + ".sqlite"
    if os.path.exists(db):
        return db
    nsys = shutil.which("nsys")
    if nsys is None:
        sys.exit(f"{db} not found and no `nsys` on the PATH; export the report once with "
                 f"`nsys export --type sqlite {report}` (or File > Export in the GUI)")
    subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite=true", "-o", db, report], check=True)
    return db


def table_exists(con, name):
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


def load(db, t0, t1, lag):
    """NVTX ranges with their GPU projection, and the memory operations in the window.

    Host ranges up to `lag` ns before the window are considered, since the GPU
    runs behind the host; a range's projection is the span of the GPU
    operations launched by the CUDA API calls issued inside it on its thread."""
    con = sqlite3.connect(db)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    ranges = []
    for start, end, text, text_id, tid in con.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
            "WHERE eventType IN (59, 60) AND end IS NOT NULL AND end > ? AND start < ?", (t0 - lag, t1)):
        ranges.append(dict(start=start, end=end, tid=tid, name=text if text else strings.get(text_id, "?")))

    ops = {}                                  # correlationId -> GPU (start, end)
    memory = []                               # memory copies in the window, GPU times
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        if not table_exists(con, table):
            continue
        for cid, start, end in con.execute(f"SELECT correlationId, start, end FROM {table} "
                                           "WHERE start >= ? AND start < ?", (t0 - lag, t1 + lag)):
            ops[cid] = (start, end)
        if table == "CUPTI_ACTIVITY_KIND_MEMCPY":
            for start, end, k, nbytes in con.execute(f"SELECT start, end, copyKind, bytes FROM {table} "
                                                     "WHERE end > ? AND start < ?", (t0, t1)):
                memory.append(dict(start=start, end=end, kind=k, bytes=nbytes))

    by_tid = defaultdict(list)
    if table_exists(con, "CUPTI_ACTIVITY_KIND_RUNTIME"):
        for start, tid, cid in con.execute("SELECT start, globalTid, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                                           "WHERE start >= ? AND start < ? ORDER BY start", (t0 - lag, t1)):
            by_tid[tid].append((start, cid))
    con.close()
    starts = {tid: [s for s, _ in calls] for tid, calls in by_tid.items()}
    for r in ranges:
        calls, ss = by_tid.get(r["tid"], []), starts.get(r["tid"], [])
        i, j = bisect.bisect_left(ss, r["start"]), bisect.bisect_left(ss, r["end"])
        found = [ops[cid] for _, cid in calls[i:j] if cid in ops]
        r["gstart"] = min(s for s, _ in found) if found else None
        r["gend"] = max(e for _, e in found) if found else None
        r["nops"] = len(found)
    return ranges, memory


# ---------------------------------------------------------------- selection ---

def short_name(name):
    return name.replace("MadNLP.", "").replace("MadIPM.", "").replace("LinearAlgebra.", "")


def assign_depths(ranges):
    """Nesting depth of each range, per thread (ranges are properly nested)."""
    ranges.sort(key=lambda r: (r["tid"], r["start"], -r["end"]))
    stacks = defaultdict(list)
    for r in ranges:
        st = stacks[r["tid"]]
        while st and st[-1] <= r["start"]:
            st.pop()
        r["depth"] = len(st)
        st.append(r["end"])


def select_row(ranges, names, depth, t0, t1):
    """The ranges of one row (one nesting level, or the named functions) whose GPU
    projection touches the window; of two nested picks the outer one is kept."""
    assign_depths(ranges)
    picked = [r for r in ranges if (r["depth"] == depth if depth is not None else short_name(r["name"]) in names)
              and r["gstart"] is not None and r["gend"] > t0 and r["gstart"] < t1]
    picked.sort(key=lambda r: (r["start"], -r["end"]))
    row, end = [], -1
    for r in picked:
        if r["start"] >= end:
            row.append(r)
            end = r["end"]
    row.sort(key=lambda r: r["gstart"])
    return row, len(picked) - len(row)


def fmt_bytes(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


def union_length(intervals):
    total, cur_a, cur_b = 0, None, None
    for a, b in sorted(intervals):
        if cur_b is None or a > cur_b:
            total += 0 if cur_b is None else cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    return total + (0 if cur_b is None else cur_b - cur_a)


# ---------------------------------------------------------------- drawing -----

def draw(row, memory, t0, t1, title, out, width, show_memory=True):
    span = t1 - t0
    ms = lambda ns: (ns - t0) / 1e6           # window-relative milliseconds
    gclip = lambda r: (max(r["gstart"], t0), min(r["gend"], t1))

    # geometry in inches: fixed margins, rows of fixed height, the axes fill the rest
    left_in, right_in, bottom_in = 0.4, 0.06, 0.38
    top_in = 0.22 if title else 0.06
    gpu_in = 0.3 if width <= 5 else 0.45      # GPU row height; memory rows are 0.42 of it
    axes_w = width - left_in - right_in
    pts_per_ms = axes_w * 72 / (span / 1e6)
    text_w = lambda text, size: len(text) * size * 0.58
    fits = lambda x0, x1, text, size: (x1 - x0) * pts_per_ms >= text_w(text, size)

    # bars too narrow for their label get a leader label above the row, staggered
    # over up to four levels when neighbours are close
    inside, leaders, placed = set(), {}, []   # bar indices labelled inside; index -> level; (x, level) placed
    for i, r in enumerate(row):
        a, b = gclip(r)
        if fits(ms(a), ms(b), short_name(r["name"]), 7):
            inside.add(i)
            continue
        if (ms(b) - ms(a)) * pts_per_ms < 1.5:
            continue                          # too thin to point at; the legend lists it
        xc = (ms(a) + ms(b)) / 2
        near = {lv for xp, lv in placed if abs(xp - xc) * pts_per_ms < 70}
        free = [lv for lv in range(4) if lv not in near]
        if free:
            leaders[i] = free[0]
            placed.append((xc, free[0]))
    top_pad = 0.75 + 0.38 * max(leaders.values()) if leaders else 0.15

    # rows, top to bottom: the projected ranges, then one thin row per memory copy kind
    by_kind = defaultdict(list)
    for m in memory:
        by_kind[m["kind"]].append(m)
    mem_rows = MEMORY_ROWS
    lanes = [("gpu", "GPU", 1.0)] + ([(k, lbl, 0.42) for k, lbl, _ in mem_rows] if show_memory else [])
    gap, y_of, cur = 0.22, {}, 0.0
    for key, _, h in reversed(lanes):
        y_of[key] = cur
        cur += h + gap
    units = cur - gap + gap + top_pad         # data units from the bottom gap to the top of the label space
    order, totals = [], defaultdict(float)
    for r in row:
        n = short_name(r["name"])
        if n not in order:
            order.append(n)
        a, b = gclip(r)
        totals[n] += b - a
    ncol = 1 if width <= 5 else 2
    n_lines = max(-(-(len(order) + 1) // ncol), len(mem_rows) if show_memory else 0)
    legend_in = 0.125 * (n_lines + 1) + 0.06  # legend area below the x label: title + entries at 6 pt
    axes_h = gpu_in * units
    fig_h = top_in + axes_h + bottom_in + legend_in
    fig, ax = plt.subplots(figsize=(width, fig_h))
    fig.subplots_adjust(left=left_in / width, right=1 - right_in / width,
                        bottom=(bottom_in + legend_in) / fig_h, top=1 - top_in / fig_h)
    ax.set_xlim(0, span / 1e6)
    ax.set_ylim(-gap, units - gap)
    min_w = 0.8 / pts_per_ms                  # ticks at least 0.8 pt wide

    # projected ranges
    fallback = [plt.get_cmap(FALLBACK_CMAP)(i) for i in range(7)]   # without Set2's grey (the idle fill)
    color = {n: PHASE_COLORS.get(n, fallback[i % len(fallback)]) for i, n in enumerate(order)}
    y, h = y_of["gpu"], 1.0
    ax.add_patch(Rectangle((0, y), span / 1e6, h, facecolor=IDLE_COLOR, edgecolor="none"))
    for i, r in enumerate(row):
        a, b = gclip(r)
        x0, x1, n = ms(a), ms(b), short_name(r["name"])
        ax.add_patch(Rectangle((x0, y), max(x1 - x0, min_w), h, facecolor=color[n], edgecolor="white", linewidth=0.5))
        if i in leaders:
            xc = (x0 + x1) / 2
            half = text_w(n, 6.5) / 2 / pts_per_ms      # keep the text inside the axes
            xt = min(max(xc, half), span / 1e6 - half)
            ax.annotate(n, xy=(xc, y + h), xytext=(xt, y + h + 0.3 + 0.38 * leaders[i]),
                        ha="center", va="bottom", fontsize=6.5, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkA=0, shrinkB=0))
        elif i in inside:
            ax.text((x0 + x1) / 2, y + h / 2, n, ha="center", va="center", fontsize=7, clip_on=True)
    idle = span - union_length([gclip(r) for r in row])

    # memory copies as ticks
    if show_memory:
        for k, lbl, col in mem_rows:
            y, h = y_of[k], 0.42
            for m in by_kind.get(k, []):
                x0, x1 = ms(max(m["start"], t0)), ms(min(m["end"], t1))
                ax.add_patch(Rectangle((x0, y), max(x1 - x0, min_w), h, facecolor=col, edgecolor="none"))

    # axes, titles
    ax.set_yticks([y_of[k] + h / 2 for k, _, h in lanes])
    ax.set_yticklabels([lbl for _, lbl, _ in lanes], fontsize=7)
    ax.set_xlabel("time (ms)", fontsize=7, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.5, pad=2)
    ax.tick_params(axis="y", length=0, pad=3)
    if title:
        ax.set_title(title, fontsize=8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)

    # legends in the reserved area below the x label: GPU time per phase (left), copies per kind (right)
    ytop = legend_in / fig_h                  # top of the legend area, figure fraction
    style = dict(fontsize=6, title_fontsize=6.5, frameon=False, handlelength=1.2, handletextpad=0.5,
                 borderaxespad=0, labelspacing=0.3)
    by_time = sorted(order, key=lambda n: -totals[n])
    fmt_ms = lambda ns: f"{ns / 1e6:.1f} ms" if ns < 10e6 else f"{ns / 1e6:.0f} ms"
    handles = [Rectangle((0, 0), 1, 1, facecolor=color[n]) for n in by_time] + [Rectangle((0, 0), 1, 1, facecolor=IDLE_COLOR)]
    texts = [f"{n}  {fmt_ms(totals[n])} ({100 * totals[n] / span:.0f}%)" for n in by_time] \
        + [f"idle or other  {fmt_ms(idle)} ({100 * idle / span:.0f}%)"]
    fig.legend(handles, texts, title="GPU time", loc="upper left", bbox_to_anchor=(0.01, ytop), ncol=ncol, **style)
    if show_memory:
        fig.legend([Line2D([], [], color=col, linewidth=1.2) for _, _, col in mem_rows],
                   [f"{lbl}  ×{len(by_kind.get(k, []))}, {fmt_bytes(sum(m['bytes'] for m in by_kind.get(k, [])))}"
                    for k, lbl, _ in mem_rows],
                   title="memory copies", loc="upper right", bbox_to_anchor=(0.99, ytop), **style)
    fig.savefig(out, dpi=300)
    print(f"wrote {out} ({width} in wide): {len(row)} projected bars ({', '.join(f'{n} {totals[n] / 1e6:.1f} ms' for n in order)}), "
          f"idle {idle / 1e6:.1f} ms; window starts at {t0 / 1e9:.3f} s; copies: "
          + ", ".join(f"{lbl} ×{len(by_kind.get(k, []))}" for k, lbl, _ in mem_rows))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", help=".sqlite export, or .nsys-rep when `nsys` is on the PATH")
    p.add_argument("--start", type=float, required=True, help="window start, seconds on the timeline")
    p.add_argument("--end", type=float, required=True, help="window end, seconds on the timeline")
    p.add_argument("-o", "--out", default=None, help="output figure (.pdf/.png/.svg); default <report>-<start>-<end>.pdf")
    p.add_argument("--ranges", default=",".join(PHASES),
                   help="comma-separated function names for the row (default: the solver phases)")
    p.add_argument("--depth", type=int, default=None, help="show one NVTX nesting level instead of --ranges")
    p.add_argument("--lag", type=float, default=5.0, help="seconds before the window to look for host ranges whose GPU work falls in it")
    p.add_argument("--title", default=None, help="title above the axes (default none: the caption says it)")
    p.add_argument("--width", type=float, default=3.4, help="figure width in inches (default: one column of a two-column paper)")
    p.add_argument("--no-memory", action="store_true", help="omit the memory copy rows")
    a = p.parse_args()

    db = sqlite_path(a.report)
    t0, t1 = int(a.start * 1e9), int(a.end * 1e9)
    ranges, memory = load(db, t0, t1, int(a.lag * 1e9))
    row, nested = select_row(ranges, [n.strip() for n in a.ranges.split(",")], a.depth, t0, t1)
    if not row:
        sys.exit("no matching NVTX ranges with GPU work in that window")
    if nested:
        print(f"note: {nested} nested ranges hidden behind their outer range")
    stem = os.path.splitext(os.path.basename(a.report))[0]
    out = a.out or f"{stem}-{a.start:.3f}-{a.end:.3f}.pdf"
    draw(row, memory, t0, t1, a.title, out, a.width, show_memory=not a.no_memory)


if __name__ == "__main__":
    main()
