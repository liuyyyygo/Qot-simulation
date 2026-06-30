"""Temporal curves for independent impairment exposure by strategy and load."""

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
from compare_impairment_decomposition import link_channel_terms, path_osnr_from_lookup


def collect_temporal_impairment_rows(sim, results, load, strategy, duration_s, step_s):
    established = [r for r in results if r.accepted]
    time_points = []
    t = 0.0
    while t <= duration_s + 1e-9:
        time_points.append(round(t, 9))
        t += step_s

    setup_osnr_by_request = {}
    rows = []
    original_time = sim.config.sim_time_seconds
    for time_s in time_points:
        sim._refresh_impairments_at_time(time_s)
        link_osnr_db = sim._build_link_osnr_lookup()

        path_fsl_sum = []
        path_solar_max = []
        path_doppler_max = []
        path_edfa_gain_max = []
        path_xt_max = []
        path_osnr = []
        path_osnr_abs_delta = []

        for result in established:
            osnr_db = path_osnr_from_lookup(sim, result, link_osnr_db)
            path_osnr.append(osnr_db)
            if result.request_id not in setup_osnr_by_request:
                setup_osnr_by_request[result.request_id] = osnr_db
            path_osnr_abs_delta.append(abs(osnr_db - setup_osnr_by_request[result.request_id]))

            fsl_sum = 0.0
            solar_max = 0.0
            doppler_max = 0.0
            edfa_gain_max = 0.0
            xt_max_dbm = -300.0
            for link_id in result.path_links:
                imp = sim._impairment_cache[link_id]
                terms = link_channel_terms(sim, link_id, result.wavelength_channel)
                fsl_sum += imp.free_space_loss_dB
                solar_max = max(solar_max, terms["solar_penalty_dB"])
                doppler_max = max(doppler_max, 1000.0 * imp.doppler_filter_penalty_dB)
                edfa_gain_max = max(edfa_gain_max, imp.edfa_gain_degradation_dB)
                xt_dbm = 10.0 * math.log10(max(terms["xt_noise_W"], 1e-30) / 1e-3)
                xt_max_dbm = max(xt_max_dbm, xt_dbm)

            path_fsl_sum.append(fsl_sum)
            path_solar_max.append(solar_max)
            path_doppler_max.append(doppler_max)
            path_edfa_gain_max.append(edfa_gain_max)
            path_xt_max.append(xt_max_dbm)

        rows.append(
            {
                "load": load,
                "strategy": strategy,
                "time_seconds": time_s,
                "time_min": time_s / 60.0,
                "mean_path_fsl_sum_dB": float(np.mean(path_fsl_sum)) if path_fsl_sum else 0.0,
                "mean_max_solar_penalty_dB": float(np.mean(path_solar_max)) if path_solar_max else 0.0,
                "mean_max_doppler_penalty_mdB": float(np.mean(path_doppler_max)) if path_doppler_max else 0.0,
                "mean_max_edfa_gain_deg_dB": float(np.mean(path_edfa_gain_max)) if path_edfa_gain_max else 0.0,
                "mean_max_xt_noise_dBm": float(np.mean(path_xt_max)) if path_xt_max else -300.0,
                "mean_osnr_dB": float(np.mean(path_osnr)) if path_osnr else 0.0,
                "mean_abs_osnr_delta_dB": float(np.mean(path_osnr_abs_delta)) if path_osnr_abs_delta else 0.0,
            }
        )

    sim._refresh_impairments_at_time(original_time)
    return rows


def run_strategy(config, requests, load, strategy_name, strategy_value, duration_s, step_s):
    cfg = copy.deepcopy(config)
    cfg.routing_strategy = strategy_value
    sim = QoTSimulation(cfg)
    results = sim.run_requests(requests, verbose=False)
    return collect_temporal_impairment_rows(
        sim,
        results,
        load,
        strategy_name,
        duration_s,
        step_s,
    )


def plot_temporal_impairment_curves(rows, outdir):
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
        "legend.fontsize": 6.1,
        "axes.linewidth": 0.7,
    })
    loads = ["Low", "Medium", "High"]
    strategies = ["Baseline", "QoT-GLR"]
    colours = {"Baseline": "#555555", "QoT-GLR": "#0F4D92"}
    styles = {"Low": "-", "Medium": "--", "High": ":"}
    markers = {"Low": "o", "Medium": "s", "High": "^"}
    panels = [
        ("mean_path_fsl_sum_dB", "Distance-related loss", "Mean cumulative FSL (dB)", "a"),
        ("mean_max_solar_penalty_dB", "Solar-background exposure", "Mean max penalty (dB)", "b"),
        ("mean_max_doppler_penalty_mdB", "Doppler filtering exposure", "Mean max penalty (mdB)", "c"),
        ("mean_max_edfa_gain_deg_dB", "Radiation-induced EDFA degradation", "Mean max gain loss (dB)", "d"),
        ("mean_max_xt_noise_dBm", "Crosstalk exposure", "Mean max noise power (dBm)", "e"),
        ("mean_abs_osnr_delta_dB", "Temporal QoT fluctuation", "Mean |OSNR(t)-OSNR(0)| (dB)", "f"),
    ]

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(
        2, 3,
        left=0.075,
        right=0.985,
        top=0.90,
        bottom=0.10,
        wspace=0.30,
        hspace=0.34,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    handles = []
    labels = []

    for ax, (metric, title, ylabel, panel) in zip(axes, panels):
        for strategy in strategies:
            for load in loads:
                selected = sorted(
                    [
                        r for r in rows
                        if r["strategy"] == strategy and r["load"] == load
                    ],
                    key=lambda r: r["time_seconds"],
                )
                x = [r["time_min"] for r in selected]
                y = [r[metric] for r in selected]
                line, = ax.plot(
                    x,
                    y,
                    color=colours[strategy],
                    linestyle=styles[load],
                    marker=markers[load],
                    markersize=2.4,
                    markevery=max(1, len(x) // 6),
                    linewidth=1.05,
                    label=f"{strategy}-{load}",
                )
                if ax is axes[0]:
                    handles.append(line)
                    labels.append(f"{strategy}-{load}")
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
    base = os.path.join(outdir, "qot_glr_vs_baseline_temporal_impairment_curves")
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".png", dpi=600)
    plt.close(fig)


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
    config = build_default_config()
    all_rows = []
    for load, n_req, seed_offset in [
        ("Low", args.low, 0),
        ("Medium", args.medium, 100),
        ("High", args.high, 200),
    ]:
        request_sim = QoTSimulation(copy.deepcopy(config))
        requests = create_demo_requests(
            request_sim.constellation.total_sats,
            num_requests=n_req,
            seed=args.seed + seed_offset,
        )
        for strategy_name, strategy_value in [
            ("Baseline", "baseline"),
            ("QoT-GLR", "qot_glr"),
        ]:
            rows = run_strategy(
                config,
                requests,
                load,
                strategy_name,
                strategy_value,
                args.observe_duration,
                args.observe_step,
            )
            all_rows.extend(rows)
            print(f"{load:>6} {strategy_name:>8}: {len(rows)} time aggregates")

    csv_path = os.path.join(args.outdir, "qot_glr_vs_baseline_temporal_impairment_curves.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_temporal_impairment_curves(all_rows, args.outdir)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_temporal_impairment_curves.svg')}")


if __name__ == "__main__":
    main()
