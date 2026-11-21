"""Redraw Fig. 3b-d from the November publication data and statistics."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs/figures/figure3'
OUTPUT.mkdir(parents=True,exist_ok=True)
STATS = ROOT / 'data/figures/figure3'
TIMING = STATS
INK = "#262626"
DOT = "#777777"
BLUE = "#2268A6"
GUIDE = "#DFE2E5"
PATTERNS = ("ampa", "nmda", "bhigh", "sync")
GROUP_LABELS = ("Uni\nAMPA", "Uni\nNMDA", "HF", "Sync")
DYNAMICS = (
    ("c", "HF NMDA/AMPA", "bhigh", 7, (0, 18, 36, 42)),
    ("d", "Sync NMDA/AMPA", "sync", 0, (0, 10, 11, 32)),
)


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def axis_style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set(linewidth=.75, color=INK)
    ax.tick_params(length=2.8, width=.75, pad=3, labelsize=8, colors=INK)


def dot_offsets(ax, values):
    """Spread nearby observations laterally within the group's dot strip."""
    points_per_pixel = 72 / ax.figure.dpi
    origin = ax.transData.transform((0, 0))
    points_per_x, points_per_y = (ax.transData.transform((1, 1)) - origin) * points_per_pixel
    lanes = np.array([.26, .20, .32, .14, .38, .08, .44])
    offsets = np.empty(len(values))
    placed = []
    for idx in np.argsort(values, kind="stable"):
        y = values[idx] * points_per_y
        nearby = [(x, other_y) for x, other_y in placed if y - other_y < 2.3]
        if nearby:
            xs, ys = np.asarray(nearby).T
            separation = np.min((lanes[:, None] * points_per_x - xs)**2 + (y - ys)**2, axis=1)
            allowed = np.flatnonzero(separation >= 2.3**2)
            lane = allowed[0] if allowed.size else np.argmax(separation)
        else:
            lane = 0
        offsets[idx] = lanes[lane]
        placed.append((lanes[lane] * points_per_x, y))
    return offsets


def draw_boxplot():
    source = STATS / "timing_statistics_all_completed.csv"
    audit = json.loads((STATS / "Fig3b_statistics.json").read_text(encoding="utf-8"))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == audit["source_sha256"]
    rows = read_csv(source)
    assert len({(row["pattern"], row["case_id"]) for row in rows}) == len(rows) == 200
    values = {p: np.array([float(row["gini_30"]) for row in rows if row["pattern"] == p]) for p in PATTERNS}
    comparisons = audit["comparisons"]
    assert all(len(group) == 50 and np.all(np.isfinite(group)) for group in values.values())
    assert [row["condition"] for row in comparisons] == list(PATTERNS[:-1])
    for row in comparisons:
        assert row["reference"] == "sync" and row["p_holm"] < 1e-4 and row["stars"] == "****"
        assert np.isclose(np.median(values[row["condition"]]), row["median_condition"])
        assert np.isclose(np.median(values["sync"]), row["median_sync"])

    fig = plt.figure(figsize=(3.15, 3.20))
    fig.text(.06, .955, "b", fontsize=11, weight="bold", va="top")
    fig.text(.13, .955, "Temporal concentration", fontsize=9.5, va="top")
    ax = fig.add_axes([.207, .18, .727, .566])
    centers = np.arange(4)
    box_centers = centers - .15
    ax.set(xlim=(-.51, 3.51), ylim=(0, 1), yticks=[0, .5, 1])
    ax.set_xticks(centers, GROUP_LABELS)
    ax.set_ylabel("Timing Gini", fontsize=8.5, labelpad=4)
    axis_style(ax)

    stroke = {"color": INK, "linewidth": .82}
    ax.boxplot([values[p] for p in PATTERNS], positions=box_centers, widths=.37,
               manage_ticks=False, whis=1.5, showfliers=False, patch_artist=True,
               boxprops={**stroke, "facecolor": "white"}, whiskerprops=stroke,
               capprops=stroke, medianprops={**stroke, "linewidth": 1.15})
    for i, p in enumerate(PATTERNS):
        group = values[p]
        dots = ax.scatter(i + dot_offsets(ax, group), group, s=3.4, color=DOT,
                          alpha=.86, linewidths=0, zorder=3)
        assert len(dots.get_offsets()) == 50

    for i in (2, 1, 0):
        x0, x1 = box_centers[i], box_centers[-1]
        height = 1.025 + (2 - i) * .089
        ax.plot([x0, x0, x1, x1], [height, height+.023, height+.023, height],
                transform=ax.get_xaxis_transform(), color=INK, lw=.75, clip_on=False)
        ax.text((x0+x1)/2, height+.032, comparisons[i]["stars"],
                transform=ax.get_xaxis_transform(), fontsize=8, ha="center", va="bottom")
    return fig


def load_timing(pattern, case_id):
    with np.load(TIMING / f"{pattern}_example_trajectory.npz") as source:
        counts = source["counts"]
        time = source["time_ms"]
        soma = source["voltage"][:, -1]
        candidates = source["ans"]
    schedule = [row for row in read_csv(TIMING / "full_selection_schedule_v5.csv") if row["pattern"] == pattern]
    states = [row for row in read_csv(TIMING / "recorded_states_and_next_onsets_v5.csv") if row["pattern"] == pattern]
    assert all(int(row["case_id"]) == case_id for row in schedule + states)
    assert np.array_equal(counts, [int(row["n_added"]) for row in states])
    assert np.array_equal(candidates, [int(row["next_candidate_id"]) for row in schedule])
    steps = np.array([int(row["n_already_added"]) for row in schedule])
    onsets = np.array([float(row["next_onset_ms"]) for row in schedule])
    assert np.array_equal(steps, np.arange(counts[-1]))
    assert np.array_equal(onsets, 100 + 10 * (candidates % 31))
    spikes = {int(row["n_added"]): [float(t) for t in row["spike_times_ms"].split(";") if t] for row in states}
    assert all(len(spikes[int(row["n_added"])]) == int(row["spike_count"]) for row in states)
    window = (time >= 100) & (time <= 400)
    # Match the original 0 mV upward-crossing rule used to export spike times.
    for n, voltage in zip(counts, soma):
        crossing = np.flatnonzero((voltage[:-1] < 0) & (voltage[1:] >= 0)) + 1
        assert np.array_equal(time[crossing[window[crossing]]], spikes[n])
    assert not spikes[counts[-1]]
    return counts, steps, onsets, spikes, time[window], soma[:, window]


def draw_timing(fig, left, width, letter, title, pattern, case_id, trace_steps):
    def xpos(x):
        return left + width*x

    counts, steps, onsets, spikes, time, soma = load_timing(pattern, case_id)
    final_step = int(counts[-1])
    fig.text(xpos(.062), .955, letter, fontsize=11, weight="bold", va="top")
    fig.text(xpos(.134), .955, title, fontsize=9.5, va="top")
    ax = fig.add_axes([xpos(.196), .493, .756*width, .345])
    ax.set(xlim=(96, 404), ylim=(final_step+1.05, -1.05))
    major = [0, 18, 36, 42] if pattern == "bhigh" else [0, 10, 20, 32]
    ax.set_yticks(major)
    ax.set_yticks(np.setdiff1d(counts, major), minor=True)
    ax.set_ylabel("Inhibitory increments added", fontsize=7.5, labelpad=3)
    axis_style(ax)
    ax.tick_params(axis="y", which="minor", length=1.7, width=.5, color=DOT)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.spines["bottom"].set_color(GUIDE)
    for n in trace_steps:
        ax.axhline(n, color=GUIDE, lw=.55, zorder=0)

    ax.plot(onsets, steps, marker="o", ls="none", ms=1.55, mew=0, color=BLUE, zorder=2)
    for n, times in spikes.items():
        if times:
            ax.vlines(times, n-.43, n+.43, color=INK, lw=.95, zorder=3)
    ax.text(399, final_step-.40, "0 spikes", fontsize=6.4, color="#565656", ha="right", va="bottom")
    handles = [
        Line2D([], [], ls="none", marker="o", ms=2.3, mew=0, color=BLUE, label="Next inhibitory onset"),
        Line2D([], [], ls="none", marker="|", ms=5, mew=.95, color=INK, label="Somatic spike"),
    ]
    ax.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.035),
              borderaxespad=0, borderpad=0, handlelength=.85, handletextpad=.35,
              columnspacing=1.20, fontsize=6.6, frameon=False)

    fig.text(xpos(.196), .437, "Somatic voltage", fontsize=7.4, va="bottom")
    va = fig.add_axes([xpos(.196), .155, .756*width, .263], sharex=ax)
    offsets = np.arange(len(trace_steps)-1, -1, -1) * 128
    for n, offset in zip(trace_steps, offsets):
        voltage = soma[np.flatnonzero(counts == n).item()]
        va.plot(time, voltage + 80 + offset, color=INK, lw=.78, solid_capstyle="round")
    va.set(ylim=(-9, offsets[0]+122), xticks=[100, 200, 300, 400])
    va.set_yticks(offsets+15, trace_steps)
    axis_style(va)
    va.tick_params(axis="y", length=0)
    va.spines["left"].set_visible(False)
    va.set_xlabel("Time (ms)", fontsize=8.5, labelpad=3)
    scale_transform = va.get_yaxis_transform()
    va.plot([-.132, -.132], [0, 40], color=INK, lw=.9, transform=scale_transform, clip_on=False)
    va.text(-.158, 20, "40 mV", fontsize=6.7, rotation=90, ha="right", va="center", transform=scale_transform)


def save(fig, stem, vector=False):
    if vector:
        with plt.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
            for ext in ("svg", "pdf"):
                path = OUTPUT / f"{stem}.{ext}"
                fig.savefig(path, facecolor="white")
                print(path)
    else:
        path = OUTPUT / f"{stem}_preview.png"
        fig.savefig(path, dpi=400, facecolor="white")
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector", action="store_true", help="Export Fig. 3b and Fig. 3c-d to SVG and PDF.")
    args = parser.parse_args()
    plt.rcParams.update({"font.family": "Arial", "text.color": INK, "axes.labelcolor": INK})
    save(draw_boxplot(), "Fig3b_boxplot", vector=args.vector)
    if not args.vector:
        for item in DYNAMICS:
            fig = plt.figure(figsize=(2.93, 3.34))
            draw_timing(fig, 0, 1, *item)
            save(fig, f"Fig3{item[0]}_timing")
    fig = plt.figure(figsize=(5.86, 3.34))
    for i, item in enumerate(DYNAMICS):
        draw_timing(fig, i/2, .5, *item)
    save(fig, "Fig3cd_timing", vector=args.vector)


if __name__ == "__main__":
    main()
