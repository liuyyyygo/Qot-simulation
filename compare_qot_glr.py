"""Compare baseline SP+First-Fit RWA with QoT-GLR.

The comparison keeps the physical-layer configuration and traffic requests
identical for both strategies, then evaluates established lightpaths over the
same QoT observation window.
"""

import argparse
import copy
import csv
import math
import os
from statistics import mean, median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_figures import build_default_config
from qot_simulation.simulation import QoTSimulation, create_demo_requests


def _safe(values, fn, default=float("nan")):
    return fn(values) if values else default


def _percentile(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def _log10_values(values):
    return [math.log10(max(v, 1e-300)) for v in values]


def run_strategy(config, requests, label, observe_duration_s, observe_step_s):
    sim = QoTSimulation(config)
    results = sim.run_requests(requests, verbose=False)
    samples = sim.evaluate_established_lightpaths_over_time(
        results,
        duration_s=observe_duration_s,
        step_s=observe_step_s,
    )
    accepted = [r for r in results if r.accepted]
    threshold = (
        config.ber_threshold_with_fec
        if config.use_fec
        else config.ber_threshold_no_fec
    )
    osnr_init = [r.osnr_dB for r in accepted]
    ber_init = [r.ber for r in accepted]
    hops = [r.total_hops for r in accepted]
    osnr_time = [s.osnr_dB for s in samples]
    ber_time = [s.ber for s in samples]

    metrics = {
        "strategy": label,
        "requests": len(requests),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(requests) if requests else 0.0,
        "initial_osnr_median_dB": _safe(osnr_init, median),
        "initial_osnr_p5_dB": _percentile(osnr_init, 5),
        "initial_osnr_min_dB": _safe(osnr_init, min),
        "initial_log10ber_median": _safe(_log10_values(ber_init), median),
        "initial_ber_gt_fec_rate": (
            sum(1 for b in ber_init if b > threshold) / len(ber_init)
            if ber_init else float("nan")
        ),
        "time_osnr_median_dB": _safe(osnr_time, median),
        "time_osnr_p5_dB": _percentile(osnr_time, 5),
        "time_osnr_min_dB": _safe(osnr_time, min),
        "time_log10ber_median": _safe(_log10_values(ber_time), median),
        "time_ber_gt_1e7_rate": (
            sum(1 for b in ber_time if b > 1.0e-7) / len(ber_time)
            if ber_time else float("nan")
        ),
        "time_ber_gt_fec_rate": (
            sum(1 for b in ber_time if b > threshold) / len(ber_time)
            if ber_time else float("nan")
        ),
        "mean_hops": _safe(hops, mean),
        "median_hops": _safe(hops, median),
    }
    path_rows = []
    for result in results:
        path_rows.append(
            {
                "strategy": label,
                "request_id": result.request_id,
                "accepted": result.accepted,
                "hops": result.total_hops,
                "wavelength": result.wavelength_channel,
                "initial_osnr_dB": result.osnr_dB,
                "initial_ber": result.ber,
                "q_factor": result.q_factor,
            }
        )

    sample_rows = []
    for sample in samples:
        sample_rows.append(
            {
                "strategy": label,
                "request_id": sample.request_id,
                "time_seconds": sample.time_seconds,
                "osnr_dB": sample.osnr_dB,
                "ber": sample.ber,
                "log10_ber": math.log10(max(sample.ber, 1e-300)),
                "min_sun_angle_deg": sample.min_sun_angle_deg,
                "max_link_distance_km": sample.max_link_distance_km,
            }
        )
    return metrics, results, samples, path_rows, sample_rows


def plot_comparison(rows, outdir):
    levels = sorted(set(row["load"] for row in rows), key=["Low", "Medium", "High"].index)
    strategies = ["Baseline", "QoT-GLR"]
    colours = {"Baseline": "#555555", "QoT-GLR": "#0072B2"}

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    metrics = [
        ("acceptance_rate", "Acceptance rate", lambda v: 100 * v, "%"),
        ("time_osnr_p5_dB", "5th-percentile OSNR", lambda v: v, "dB"),
        ("time_ber_gt_1e7_rate", "BER > 1e-7 samples", lambda v: 100 * v, "%"),
        ("time_ber_gt_fec_rate", "BER > FEC samples", lambda v: 100 * v, "%"),
    ]
    x = np.arange(len(levels))
    width = 0.36
    for ax, (key, title, transform, unit) in zip(axes.flat, metrics):
        for offset, strategy in [(-width / 2, "Baseline"), (width / 2, "QoT-GLR")]:
            vals = []
            for level in levels:
                row = next(r for r in rows if r["load"] == level and r["strategy"] == strategy)
                vals.append(transform(float(row[key])))
            ax.bar(x + offset, vals, width, label=strategy, color=colours[strategy])
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylabel(unit)
        ax.grid(axis="y", lw=0.3, alpha=0.3)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Baseline SP+FF versus QoT-GLR under identical traffic")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "qot_glr_vs_baseline_summary.svg"))
    plt.close(fig)


def plot_publication_comparison(metric_rows, sample_rows, path_rows, outdir):
    """Create a publication-ready multi-panel comparison figure."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
    colours = {"Baseline": "#555555", "QoT-GLR": "#0F4D92"}
    loads = ["Low", "Medium", "High"]
    strategies = ["Baseline", "QoT-GLR"]

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(
        2, 3,
        left=0.07,
        right=0.985,
        top=0.89,
        bottom=0.10,
        wspace=0.36,
        hspace=0.48,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    def panel_label(ax, label):
        ax.text(
            -0.18, 1.05, label,
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=8,
            va="bottom",
            ha="left",
        )

    row_by_key = {(r["load"], r["strategy"]): r for r in metric_rows}
    x = np.arange(len(loads))
    width = 0.34

    # a. Acceptance and hop cost.
    ax = axes[0]
    for offset, strategy in [(-width / 2, "Baseline"), (width / 2, "QoT-GLR")]:
        vals = [100.0 * row_by_key[(load, strategy)]["acceptance_rate"] for load in loads]
        ax.bar(x + offset, vals, width, color=colours[strategy], label=strategy)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Accepted requests (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.set_title("Connection establishment")
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    panel_label(ax, "a")

    # b. Time-window OSNR percentile ranges.
    ax = axes[1]
    positions = []
    pos = 0
    for load in loads:
        for strategy in strategies:
            vals = [
                float(s["osnr_dB"]) for s in sample_rows
                if s["load"] == load and s["strategy"] == strategy
            ]
            p5, p25, p50, p75, p95 = np.percentile(vals, [5, 25, 50, 75, 95])
            positions.append(pos)
            ax.plot([pos, pos], [p5, p95], color=colours[strategy], lw=1.0, alpha=0.75)
            ax.plot([pos - 0.10, pos + 0.10], [p5, p5], color=colours[strategy], lw=0.8)
            ax.plot([pos - 0.10, pos + 0.10], [p95, p95], color=colours[strategy], lw=0.8)
            ax.add_patch(
                plt.Rectangle(
                    (pos - 0.12, p25),
                    0.24,
                    p75 - p25,
                    facecolor=colours[strategy],
                    edgecolor=colours[strategy],
                    alpha=0.78,
                    linewidth=0.7,
                )
            )
            ax.scatter(pos, p50, s=11, color="white", edgecolor=colours[strategy], linewidth=0.7, zorder=3)
            pos += 0.42
        pos += 0.36
    centers = [(positions[i * 2] + positions[i * 2 + 1]) / 2 for i in range(len(loads))]
    ax.set_xticks(centers)
    ax.set_xticklabels(loads)
    ax.axhline(10.0, color="#B64342", lw=0.7, ls=":", zorder=0)
    ax.text(positions[-1] + 0.15, 10.0, "10 dB", va="center", ha="left", fontsize=6, color="#B64342")
    ax.set_ylabel("Temporal OSNR (dB)")
    ax.set_title("QoT distribution over holding time")
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    panel_label(ax, "b")

    # c. 5th percentile OSNR improvement.
    ax = axes[2]
    for yi, load in enumerate(loads):
        b = row_by_key[(load, "Baseline")]["time_osnr_p5_dB"]
        g = row_by_key[(load, "QoT-GLR")]["time_osnr_p5_dB"]
        ax.plot([b, g], [yi, yi], color="#A8A8A8", lw=1.2, zorder=1)
        ax.scatter(b, yi, color=colours["Baseline"], s=18, zorder=2)
        ax.scatter(g, yi, color=colours["QoT-GLR"], s=18, zorder=2)
        ax.text(g + 1.0, yi, f"+{g - b:.1f} dB", va="center", fontsize=6.5)
    ax.set_yticks(range(len(loads)))
    ax.set_yticklabels(loads)
    ax.set_xlabel("5th-percentile OSNR (dB)")
    ax.set_title("Tail-OSNR gain")
    ax.grid(axis="x", lw=0.25, alpha=0.25)
    panel_label(ax, "c")

    # d. BER violation rates.
    ax = axes[3]
    categories = []
    values = []
    bar_colors = []
    xpos = []
    pos = 0
    for load in loads:
        for strategy in strategies:
            categories.append(load if strategy == "Baseline" else "")
            values.append(100.0 * row_by_key[(load, strategy)]["time_ber_gt_fec_rate"])
            bar_colors.append(colours[strategy])
            xpos.append(pos)
            pos += 0.42
        pos += 0.36
    ax.bar(xpos, values, width=0.30, color=bar_colors)
    for xp, val in zip(xpos, values):
        ax.text(xp, val + 0.35, f"{val:.1f}", ha="center", va="bottom", fontsize=6)
    ax.set_xticks([(xpos[i * 2] + xpos[i * 2 + 1]) / 2 for i in range(len(loads))])
    ax.set_xticklabels(loads)
    ax.set_ylabel("BER > FEC samples (%)")
    ax.set_title("Hard QoT-risk events")
    ax.set_ylim(0, max(values) * 1.35 if values else 1)
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    panel_label(ax, "d")

    # e. Pre-FEC sensitivity threshold.
    ax = axes[4]
    for offset, strategy in [(-width / 2, "Baseline"), (width / 2, "QoT-GLR")]:
        vals = [100.0 * row_by_key[(load, strategy)]["time_ber_gt_1e7_rate"] for load in loads]
        ax.bar(x + offset, vals, width, color=colours[strategy])
    ax.set_ylabel("BER > 1e-7 samples (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.set_title("Pre-FEC sensitivity events")
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    panel_label(ax, "e")

    # f. Routing cost.
    ax = axes[5]
    for offset, strategy in [(-width / 2, "Baseline"), (width / 2, "QoT-GLR")]:
        vals = [row_by_key[(load, strategy)]["mean_hops"] for load in loads]
        ax.bar(x + offset, vals, width, color=colours[strategy])
    ax.set_ylabel("Mean hop count")
    ax.set_xticks(x)
    ax.set_xticklabels(loads)
    ax.set_title("Routing-cost preservation")
    ax.set_ylim(0, max(row["mean_hops"] for row in metric_rows) + 1.0)
    ax.grid(axis="y", lw=0.25, alpha=0.25)
    panel_label(ax, "f")

    handles = [
        plt.Line2D([0], [0], color=colours["Baseline"], lw=5),
        plt.Line2D([0], [0], color=colours["QoT-GLR"], lw=5),
    ]
    fig.legend(
        handles,
        ["Baseline SP+FF", "QoT-GLR"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.985),
        frameon=False,
    )
    fig.suptitle(
        "QoT-GLR suppresses temporal QoT tail risk without reducing connection establishment",
        y=0.94,
        fontsize=8.5,
    )
    base = os.path.join(outdir, "qot_glr_vs_baseline_publication")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)


def aggregate_temporal_metrics(sample_rows):
    grouped = {}
    for row in sample_rows:
        key = (row["load"], row["strategy"], float(row["time_seconds"]))
        grouped.setdefault(key, []).append(row)

    temporal_rows = []
    for (load, strategy, time_seconds), rows in sorted(grouped.items()):
        osnrs = [float(r["osnr_dB"]) for r in rows]
        bers = [float(r["ber"]) for r in rows]
        log_bers = [math.log10(max(b, 1e-300)) for b in bers]
        temporal_rows.append(
            {
                "load": load,
                "strategy": strategy,
                "time_seconds": time_seconds,
                "time_min": time_seconds / 60.0,
                "mean_osnr_dB": float(np.mean(osnrs)),
                "p5_osnr_dB": float(np.percentile(osnrs, 5)),
                "mean_log10_ber": float(np.mean(log_bers)),
                "ber_gt_1e7_rate": float(np.mean([b > 1.0e-7 for b in bers])),
                "ber_gt_fec_rate": float(np.mean([b > 2.0e-3 for b in bers])),
            }
        )
    return temporal_rows


def plot_temporal_curves(sample_rows, outdir):
    """Plot six temporal curves for each aggregate QoT metric."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.linewidth": 0.7,
    })
    temporal_rows = aggregate_temporal_metrics(sample_rows)
    loads = ["Low", "Medium", "High"]
    strategies = ["Baseline", "QoT-GLR"]
    colours = {"Baseline": "#555555", "QoT-GLR": "#0F4D92"}
    styles = {"Low": "-", "Medium": "--", "High": ":"}
    markers = {"Low": "o", "Medium": "s", "High": "^"}

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(
        2, 2,
        left=0.075,
        right=0.985,
        top=0.89,
        bottom=0.10,
        wspace=0.24,
        hspace=0.30,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    panel_specs = [
        ("mean_osnr_dB", "Mean OSNR", "Mean OSNR (dB)", "a"),
        ("p5_osnr_dB", "Tail OSNR", "5th-percentile OSNR (dB)", "b"),
        ("mean_log10_ber", "Mean BER level", "Mean log10(BER)", "c"),
        ("ber_gt_fec_rate", "Hard QoT-risk events", "BER > FEC samples (%)", "d"),
    ]

    def select(load, strategy):
        rows = [
            r for r in temporal_rows
            if r["load"] == load and r["strategy"] == strategy
        ]
        return sorted(rows, key=lambda r: r["time_seconds"])

    handles = []
    labels = []
    for ax, (metric, title, ylabel, panel) in zip(axes, panel_specs):
        for strategy in strategies:
            for load in loads:
                rows = select(load, strategy)
                x = [r["time_min"] for r in rows]
                y = [r[metric] for r in rows]
                if metric == "ber_gt_fec_rate":
                    y = [100.0 * v for v in y]
                line, = ax.plot(
                    x,
                    y,
                    color=colours[strategy],
                    linestyle=styles[load],
                    marker=markers[load],
                    markersize=2.5,
                    linewidth=1.1,
                    markevery=max(1, len(x) // 6),
                    label=f"{strategy}, {load}",
                )
                if ax is axes[0]:
                    handles.append(line)
                    labels.append(f"{strategy}-{load}")

        if metric in {"mean_osnr_dB", "p5_osnr_dB"}:
            ax.axhline(10.0, color="#B64342", lw=0.7, ls=":", zorder=0)
            ax.text(0.99, 10.0, "10 dB", transform=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=5.8, color="#B64342")
        if metric == "mean_log10_ber":
            ax.axhline(math.log10(1.0e-7), color="#767676", lw=0.7, ls=":", zorder=0)
            ax.axhline(math.log10(2.0e-3), color="#B64342", lw=0.7, ls=":", zorder=0)
            ax.text(0.99, math.log10(1.0e-7), "1e-7",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=5.8, color="#767676")
            ax.text(0.99, math.log10(2.0e-3), "FEC",
                    transform=ax.get_yaxis_transform(), ha="right", va="bottom",
                    fontsize=5.8, color="#B64342")
        ax.set_xlabel("Time after setup (min)")
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.25, alpha=0.25)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
        columnspacing=1.2,
        handlelength=2.4,
    )
    base = os.path.join(outdir, "qot_glr_vs_baseline_temporal_curves")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)
    return temporal_rows


def write_markdown(rows, outdir):
    path = os.path.join(outdir, "qot_glr_vs_baseline_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# QoT-GLR versus baseline comparison\n\n")
        f.write("| Load | Strategy | Accepted | Acc. rate | Time OSNR p5 (dB) | ")
        f.write("Time OSNR min (dB) | BER>1e-7 | BER>FEC | Mean hops | Median hops |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            f.write(
                f"| {row['load']} | {row['strategy']} | {int(row['accepted'])}/"
                f"{int(row['requests'])} | {100*row['acceptance_rate']:.1f}% | "
                f"{row['time_osnr_p5_dB']:.2f} | {row['time_osnr_min_dB']:.2f} | "
                f"{100*row['time_ber_gt_1e7_rate']:.1f}% | "
                f"{100*row['time_ber_gt_fec_rate']:.1f}% | "
                f"{row['mean_hops']:.2f} | {row['median_hops']:.1f} |\n"
            )
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./figures_qot_glr_compare")
    parser.add_argument("--low", type=int, default=10)
    parser.add_argument("--medium", type=int, default=50)
    parser.add_argument("--high", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observe-duration", type=float, default=3600.0)
    parser.add_argument("--observe-step", type=float, default=300.0)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base = build_default_config()
    base.qot_observation_duration_s = args.observe_duration
    base.qot_observation_step_s = args.observe_step

    rows = []
    all_path_rows = []
    all_sample_rows = []
    for load, n_req, seed_offset in [
        ("Low", args.low, 0),
        ("Medium", args.medium, 100),
        ("High", args.high, 200),
    ]:
        request_sim = QoTSimulation(copy.deepcopy(base))
        requests = create_demo_requests(
            request_sim.constellation.total_sats,
            num_requests=n_req,
            seed=args.seed + seed_offset,
        )
        for strategy_name, strategy_value in [
            ("Baseline", "baseline"),
            ("QoT-GLR", "qot_glr"),
        ]:
            cfg = copy.deepcopy(base)
            cfg.routing_strategy = strategy_value
            metrics, _, _, path_rows, sample_rows = run_strategy(
                cfg,
                requests,
                strategy_name,
                args.observe_duration,
                args.observe_step,
            )
            metrics["load"] = load
            rows.append(metrics)
            for row in path_rows:
                row["load"] = load
                all_path_rows.append(row)
            for row in sample_rows:
                row["load"] = load
                all_sample_rows.append(row)
            print(
                f"{load:>6} {strategy_name:>8}: "
                f"acc={metrics['accepted']}/{metrics['requests']}, "
                f"OSNR_p5={metrics['time_osnr_p5_dB']:.2f} dB, "
                f"BER>FEC={100*metrics['time_ber_gt_fec_rate']:.1f}%"
            )

    csv_path = os.path.join(args.outdir, "qot_glr_vs_baseline_metrics.csv")
    fieldnames = ["load"] + [k for k in rows[0].keys() if k != "load"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    path_csv = os.path.join(args.outdir, "qot_glr_vs_baseline_paths.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_path_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_path_rows)

    sample_csv = os.path.join(args.outdir, "qot_glr_vs_baseline_time_samples.csv")
    with open(sample_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_sample_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_sample_rows)

    temporal_rows = aggregate_temporal_metrics(all_sample_rows)
    temporal_csv = os.path.join(args.outdir, "qot_glr_vs_baseline_temporal_metrics.csv")
    with open(temporal_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(temporal_rows[0].keys()))
        writer.writeheader()
        writer.writerows(temporal_rows)

    md_path = write_markdown(rows, args.outdir)
    plot_comparison(rows, args.outdir)
    plot_publication_comparison(rows, all_sample_rows, all_path_rows, args.outdir)
    plot_temporal_curves(all_sample_rows, args.outdir)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {path_csv}")
    print(f"Wrote {sample_csv}")
    print(f"Wrote {temporal_csv}")
    print(f"Wrote {md_path}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_summary.svg')}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_publication.svg')}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_temporal_curves.svg')}")


if __name__ == "__main__":
    main()
