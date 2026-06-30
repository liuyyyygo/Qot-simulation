"""Offered-load sweep for load-sensitive QoT/RWA metrics."""

import argparse
import copy
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run_figures import build_default_config
from qot_simulation.simulation import QoTSimulation, create_demo_requests
from compare_impairment_decomposition import link_channel_terms


def collect_load_metrics(config, offered_load, seed, strategy_name, strategy_value):
    cfg = copy.deepcopy(config)
    cfg.routing_strategy = strategy_value
    sim = QoTSimulation(cfg)
    requests = create_demo_requests(
        sim.constellation.total_sats,
        num_requests=offered_load,
        seed=seed,
    )
    results = sim.run_requests(requests, verbose=False)
    accepted = [r for r in results if r.accepted]
    sim.compute_all_impairments()

    link_occ_counts = np.array(
        [len(chs) for chs in sim.rwa.wavelength_occupancy.values()],
        dtype=float,
    )
    total_slots = len(link_occ_counts) * cfg.num_channels
    occupied_slots = float(np.sum(link_occ_counts))

    same_interferers = []
    adjacent_counts = []
    xt_noise_dbm = []
    for result in accepted:
        for link_id in result.path_links:
            lisl = next(l for l in sim.constellation.topology if l.link_id == link_id)
            channel = result.wavelength_channel
            same_interferers.append(
                sim._count_same_wavelength_oxc_interferers(lisl, channel)
            )
            occupied = sim.rwa.wavelength_occupancy.get(link_id, set())
            adjacent_counts.append(
                sum(1 for ch in occupied if abs(ch - channel) == 1)
            )
            terms = link_channel_terms(sim, link_id, channel)
            xt_noise_w = terms["xt_noise_W"]
            xt_noise_dbm.append(
                10.0 * math.log10(max(xt_noise_w, 1e-30) / 1e-3)
            )

    threshold = cfg.ber_threshold_with_fec if cfg.use_fec else cfg.ber_threshold_no_fec
    return {
        "strategy": strategy_name,
        "offered_load": offered_load,
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / offered_load if offered_load else 0.0,
        "capacity_utilization_pct": 100.0 * occupied_slots / total_slots if total_slots else 0.0,
        "mean_link_occupation_pct": 100.0 * float(np.mean(link_occ_counts)) / cfg.num_channels,
        "p95_link_occupation_pct": 100.0 * float(np.percentile(link_occ_counts, 95)) / cfg.num_channels,
        "max_link_occupation_pct": 100.0 * float(np.max(link_occ_counts)) / cfg.num_channels,
        "mean_same_wavelength_interferers": float(np.mean(same_interferers)) if same_interferers else 0.0,
        "mean_adjacent_channel_occupancy": float(np.mean(adjacent_counts)) if adjacent_counts else 0.0,
        "p95_xt_noise_dBm": float(np.percentile(xt_noise_dbm, 95)) if xt_noise_dbm else -300.0,
        "median_xt_noise_dBm": float(np.median(xt_noise_dbm)) if xt_noise_dbm else -300.0,
        "ber_gt_fec_rate_pct": (
            100.0 * sum(1 for r in accepted if r.ber > threshold) / len(accepted)
            if accepted else 0.0
        ),
        "mean_hops": float(np.mean([r.total_hops for r in accepted])) if accepted else 0.0,
    }


def plot_load_sensitive_metrics(rows, outdir):
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
    })
    colours = {"Baseline": "#555555", "QoT-GLR": "#0F4D92"}
    strategies = ["Baseline", "QoT-GLR"]
    metrics = [
        ("capacity_utilization_pct", "Network slot utilization", "Occupied link-wavelength slots (%)", "a"),
        ("p95_link_occupation_pct", "95th-percentile link occupation", "Occupied wavelengths per link (%)", "b"),
        ("mean_same_wavelength_interferers", "Same-wavelength OXC contention", "Mean interferers per path-hop", "c"),
        ("mean_adjacent_channel_occupancy", "Adjacent-channel contention", "Mean adjacent channels per path-hop", "d"),
        ("p95_xt_noise_dBm", "95th-percentile crosstalk noise", "Noise power (dBm)", "e"),
        ("ber_gt_fec_rate_pct", "QoT-risk event rate at setup", "Accepted paths with BER > FEC (%)", "f"),
    ]
    fig = plt.figure(figsize=(7.2, 5.7))
    gs = fig.add_gridspec(
        2, 3,
        left=0.075,
        right=0.985,
        top=0.88,
        bottom=0.10,
        wspace=0.36,
        hspace=0.46,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    for ax, (metric, title, ylabel, panel) in zip(axes, metrics):
        for strategy in strategies:
            strategy_rows = sorted(
                [r for r in rows if r["strategy"] == strategy],
                key=lambda r: r["offered_load"],
            )
            x = [r["offered_load"] for r in strategy_rows]
            y = [r[metric] for r in strategy_rows]
            ax.plot(
                x,
                y,
                marker="o",
                markersize=3.0,
                linewidth=1.3,
                color=colours[strategy],
                label=strategy,
            )
        ax.set_title(title)
        ax.set_xlabel("Number of lightpath requests")
        ax.set_ylabel(ylabel)
        ax.grid(True, lw=0.25, alpha=0.25)
        ax.text(-0.16, 1.04, panel, transform=ax.transAxes,
                fontweight="bold", fontsize=8, va="bottom", ha="left")

    handles = [
        plt.Line2D([0], [0], color=colours["Baseline"], lw=2, marker="o"),
        plt.Line2D([0], [0], color=colours["QoT-GLR"], lw=2, marker="o"),
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
        "Load-sensitive contention and QoT-risk metrics under increasing offered load",
        y=0.925,
        fontsize=8.5,
    )
    base = os.path.join(outdir, "qot_glr_vs_baseline_load_sensitive_metrics")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="./figures_qot_glr_compare")
    parser.add_argument("--loads", nargs="+", type=int, default=[10, 50, 100, 200, 400])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config = build_default_config()
    rows = []
    for offered_load in args.loads:
        for strategy_name, strategy_value in [
            ("Baseline", "baseline"),
            ("QoT-GLR", "qot_glr"),
        ]:
            row = collect_load_metrics(
                config,
                offered_load,
                args.seed + offered_load,
                strategy_name,
                strategy_value,
            )
            rows.append(row)
            print(
                f"{strategy_name:>8} load={offered_load}: "
                f"util={row['capacity_utilization_pct']:.1f}%, "
                f"same={row['mean_same_wavelength_interferers']:.2f}, "
                f"BER>FEC={row['ber_gt_fec_rate_pct']:.1f}%"
            )

    csv_path = os.path.join(args.outdir, "qot_glr_vs_baseline_load_sensitive_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_load_sensitive_metrics(rows, args.outdir)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_load_sensitive_metrics.svg')}")


if __name__ == "__main__":
    main()
