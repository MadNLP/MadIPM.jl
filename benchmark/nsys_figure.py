#!/usr/bin/env python3
"""Draw a time window of an Nsight Systems report as a two-row timeline.

    python nsys_figure.py profiles/CRE-B-bs128.sqlite --start 150.893 --end 151.310 -o crebs128.pdf

Times are seconds on the Nsight timeline ruler. Everything renders locally from
the SQLite export of the report (`REPORT.sqlite`, written by profile.sh next to
the report; or `nsys export --type sqlite REPORT.nsys-rep`; or File > Export in
the GUI). A `.nsys-rep` is accepted when `nsys` is on the PATH.

Rows: the host, as one row of NVTX ranges (by default the solver phases in
PHASES, whichever level they sit at; `--ranges` picks other names, `--depth`
shows one nesting level instead), then the memory copies as markers, host to
device and device to host highlighted, device to device in grey. Bars are
labelled inside when they fit and with a leader above the row otherwise. The
legends give each phase's host time and share of the window and the copies per
kind with their byte totals. Needs matplotlib.
"""
import argparse
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

# default host row: one bar per call of these functions (module prefixes ignored)
PHASES = ["factorize_system!", "solve_system!", "update_termination_status!", "evaluate_model!"]
PHASE_COLORS = {
    "factorize_system!": "#08306b",
    "solve_system!": "#4292c6",
    "update_termination_status!": "#f16913",
    "evaluate_model!": "#41ab5d",
}
OTHER_COLOR = "#e6e6e6"
COPY_KINDS = {1: ("host to device", "#d62728", "v"), 2: ("device to host", "#7b3294", "^"),
              8: ("device to device", "0.6", "|")}


# ---------------------------------------------------------------- database ----

def sqlite_path(report):
    if report.endswith(".sqlite"):
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


def load(db, t0, t1):
    """NVTX ranges and memory copies overlapping [t0, t1] (ns)."""
    con = sqlite3.connect(db)
    strings = dict(con.execute("SELECT id, value FROM StringIds"))
    ranges = []
    if table_exists(con, "NVTX_EVENTS"):
        for start, end, text, text_id, tid in con.execute(
                "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS "
                "WHERE eventType IN (59, 60) AND end IS NOT NULL AND end > ? AND start < ?", (t0, t1)):
            ranges.append(dict(start=start, end=end, tid=tid, name=text if text else strings.get(text_id, "?")))
    memcpys = []
    if table_exists(con, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for start, end, kind, nbytes in con.execute(
                "SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                "WHERE end > ? AND start < ?", (t0, t1)):
            memcpys.append(dict(start=start, end=end, kind=kind, bytes=nbytes))
    con.close()
    return ranges, memcpys


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


def select_row(ranges, names, depth):
    """The ranges of one row: one nesting level, or the named functions; when
    two picked ranges nest, the outer one is kept."""
    assign_depths(ranges)
    if depth is not None:
        picked = [r for r in ranges if r["depth"] == depth]
    else:
        picked = [r for r in ranges if short_name(r["name"]) in names]
    picked.sort(key=lambda r: (r["start"], -r["end"]))
    row, end = [], -1
    for r in picked:
        if r["start"] >= end:
            row.append(r)
            end = r["end"]
    return row, len(picked) - len(row)


def fmt_bytes(n):
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000


# ---------------------------------------------------------------- drawing -----

def draw(row, memcpys, t0, t1, title, out, width):
    span = t1 - t0
    ms = lambda ns: (ns - t0) / 1e6           # window-relative milliseconds
    clip = lambda e: (max(e["start"], t0), min(e["end"], t1))

    lanes = ["host"] + (["memcpy"] if memcpys is not None else [])
    lane_h, gap = 1.0, 0.3
    pts_per_ms = width * 72 * 0.84 / (span / 1e6)
    fits = lambda x0, x1, text, size: (x1 - x0) * pts_per_ms >= len(text) * size * 0.58

    # bars too narrow for their label get a leader label above the row, staggered
    # over up to three levels when neighbours are close
    leaders, placed = {}, []
    for i, r in enumerate(row):
        a, b = clip(r)
        if not fits(ms(a), ms(b), short_name(r["name"]), 7):
            xc = (ms(a) + ms(b)) / 2
            leaders[i] = min(sum(abs(xp - xc) * pts_per_ms < 70 for xp in placed), 2)
            placed.append(xc)
    top_pad = 0.5 + 0.38 * max(leaders.values()) if leaders else 0.15
    units = len(lanes) * (lane_h + gap) + gap + top_pad
    y_of = {l: (len(lanes) - 1 - i) * (lane_h + gap) for i, l in enumerate(lanes)}
    fig_w, fig_h = width, 0.45 * units / 0.77   # lanes 0.45 in tall; the axes fill ~77% of the height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, span / 1e6)
    ax.set_ylim(-gap, units - gap)

    def label(i, x0, x1, y, text):
        if i in leaders:
            xc = (x0 + x1) / 2
            ax.annotate(text, xy=(xc, y + lane_h), xytext=(xc, y + lane_h + 0.3 + 0.38 * leaders[i]),
                        ha="center", va="bottom", fontsize=6.5, annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", lw=0.5, color="0.4", shrinkA=0, shrinkB=0))
        else:
            ax.text((x0 + x1) / 2, y + lane_h / 2, text, ha="center", va="center", fontsize=7, clip_on=True)

    # host row
    order, totals = [], defaultdict(float)
    for r in row:
        n = short_name(r["name"])
        if n not in order:
            order.append(n)
        a, b = clip(r)
        totals[n] += b - a
    tab10 = plt.get_cmap("tab10")
    color = {n: PHASE_COLORS.get(n, tab10(i % 10)) for i, n in enumerate(order)}
    y = y_of["host"]
    ax.add_patch(Rectangle((0, y), span / 1e6, lane_h, facecolor=OTHER_COLOR, edgecolor="none"))
    for i, r in enumerate(row):
        a, b = clip(r)
        x0, x1, n = ms(a), ms(b), short_name(r["name"])
        ax.add_patch(Rectangle((x0, y), x1 - x0, lane_h, facecolor=color[n], edgecolor="white", linewidth=0.3))
        label(i, x0, x1, y, n)
    other = span - sum(totals.values())

    # memory copies: markers at the copy start (they last microseconds)
    copies = defaultdict(lambda: [0, 0])
    if memcpys is not None:
        y = y_of["memcpy"]
        for m in sorted(memcpys, key=lambda m: m["kind"] != 8):   # grey D2D first, under the rest
            lbl, col, marker = COPY_KINDS.get(m["kind"], ("other copy", "0.8", "|"))
            copies[m["kind"]][0] += 1
            copies[m["kind"]][1] += m["bytes"]
            x = ms(m["start"])
            if m["kind"] in (1, 2):
                ax.vlines(x, y, y + lane_h, color=col, linewidth=1.0, zorder=3)
                ax.plot([x], [y + lane_h / 2], marker=marker, color=col, markersize=5, markeredgecolor="none", zorder=4)
            else:
                ax.vlines(x, y + 0.15, y + lane_h - 0.15, color=col, linewidth=0.9, zorder=2)

    # lanes, axes, titles
    ax.set_yticks([y_of[l] + lane_h / 2 for l in lanes])
    ax.set_yticklabels(["host" if l == "host" else "memory copies" for l in lanes], fontsize=7)
    ax.set_xlabel(f"time (ms) from {t0 / 1e9:.3f} s on the Nsight timeline", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=9)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

    # legends below the axes: phases (left), copies (right)
    below = -0.55 / (0.77 * fig_h)            # ~0.55 in under the axes, in axes fraction
    handles = [Rectangle((0, 0), 1, 1, facecolor=color[n]) for n in order] + [Rectangle((0, 0), 1, 1, facecolor=OTHER_COLOR)]
    labels = [f"{n}  {totals[n] / 1e6:.1f} ms ({100 * totals[n] / span:.0f}%)" for n in order] \
        + [f"other  {other / 1e6:.1f} ms ({100 * other / span:.0f}%)"]
    lh = ax.legend(handles, labels, title="host", fontsize=6.5, title_fontsize=7, loc="upper left",
                   bbox_to_anchor=(0.0, below), frameon=False, ncol=2)
    ax.add_artist(lh)
    if copies:
        kinds = sorted(copies, key=lambda k: (k not in COPY_KINDS, k))
        ax.legend([Line2D([], [], color=COPY_KINDS.get(k, ("", "0.8", "|"))[1], marker=COPY_KINDS.get(k, ("", "", "|"))[2],
                          markersize=5, markeredgecolor="none", linewidth=1.0) for k in kinds],
                  [f"{COPY_KINDS.get(k, ('other copy',))[0]}  ×{copies[k][0]}, {fmt_bytes(copies[k][1])}" for k in kinds],
                  title="memory copies", fontsize=6.5, title_fontsize=7, loc="upper right",
                  bbox_to_anchor=(1.0, below), frameon=False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}: {len(row)} host bars ({', '.join(order)}), other {other / 1e6:.1f} ms; copies: "
          + ", ".join(f"{COPY_KINDS.get(k, ('other',))[0]} ×{c[0]} {fmt_bytes(c[1])}" for k, c in sorted(copies.items())))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("report", help=".sqlite export, or .nsys-rep when `nsys` is on the PATH")
    p.add_argument("--start", type=float, required=True, help="window start, seconds on the timeline")
    p.add_argument("--end", type=float, required=True, help="window end, seconds on the timeline")
    p.add_argument("-o", "--out", default=None, help="output figure (.pdf/.png/.svg); default <report>-<start>-<end>.pdf")
    p.add_argument("--ranges", default=",".join(PHASES),
                   help="comma-separated function names for the host row (default: the solver phases)")
    p.add_argument("--depth", type=int, default=None, help="show one NVTX nesting level instead of --ranges")
    p.add_argument("--title", default=None)
    p.add_argument("--width", type=float, default=9.0, help="figure width in inches")
    p.add_argument("--no-memcpy", action="store_true", help="omit the memory copy row")
    a = p.parse_args()

    db = sqlite_path(a.report)
    t0, t1 = int(a.start * 1e9), int(a.end * 1e9)
    ranges, memcpys = load(db, t0, t1)
    row, nested = select_row(ranges, [n.strip() for n in a.ranges.split(",")], a.depth)
    if not row:
        sys.exit("no matching NVTX ranges in that window")
    if nested:
        print(f"note: {nested} nested ranges hidden behind their outer range")
    stem = os.path.splitext(os.path.basename(a.report))[0]
    out = a.out or f"{stem}-{a.start:.3f}-{a.end:.3f}.pdf"
    title = a.title or f"{stem}: {a.start:.3f} s to {a.end:.3f} s"
    draw(row, None if a.no_memcpy else memcpys, t0, t1, title, out, a.width)


if __name__ == "__main__":
    main()
