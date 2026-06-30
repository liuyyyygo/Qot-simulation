"""Independent impairment comparison for baseline SP+FF and QoT-GLR."""

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


def w_to_dbm(power_w):
    return 10.0 * math.log10(max(power_w, 1e-30) / 1e-3)


def path_osnr_from_lookup(sim, result, link_osnr_db):
    inv_sum = 0.0
    for link_id in result.path_links:
        osnr_db = link_osnr_db.get((link_id, result.wavelength_channel), 15.0)
        inv_sum += 1.0 / max(10.0 ** (osnr_db / 10.0), 1e-12)
    osnr_lin = 1.0 / inv_sum if inv_sum > 0 else 0.0
    return 10.0 * math.log10(max(osnr_lin, 1e-30))


def link_channel_terms(sim, link_id, channel):
    lisl = next(l for l in sim.constellation.topology if l.link_id == link_id)
    imp = sim._impairment_cache[link_id]
    effective_gain_db = sim.config.nominal_gain_dB - imp.edfa_gain_degradation_dB
    signal_after_edfa_w = imp.received_signal_power_W * 10.0 ** (effective_gain_db / 10.0)
    occupied = list(sim.rwa.wavelength_occupancy.get(link_id, set()))

    inter_xt_w = sim.crosstalk.compute_inter_wavelength_noise_var(
        target_channel=channel,
        occupied_channels=occupied,
        received_power_per_channel_W=imp.received_signal_power_W,
        doppler_shifts={ch: imp.doppler_shift_GHz for ch in occupied},
    )
    intra_xt_w = sim.crosstalk.compute_intra_wavelength_noise_var(
        num_interfering_ports=sim._count_same_wavelength_oxc_interferers(lisl, channel),
        received_power_W=imp.received_signal_power_W,
    )
    sky_noise_w = (
        sim.celestial.L_sky
        * sim.celestial.A_Omega_DeltaLambda
        * sim.config.optical_efficiency
    )
    _, full_osnr_db, _ = sim.osnr_calc.compute_single_link_osnr(
        signal_power_W=signal_after_edfa_w,
        ase_noise_W=imp.ase_noise_power_W,
        celestial_noise_W=imp.celestial_noise_power_W,
        inter_xt_noise_var=inter_xt_w,
        intra_xt_noise_var=intra_xt_w,
    )
    _, sky_osnr_db, _ = sim.osnr_calc.compute_single_link_osnr(
        signal_power_W=signal_after_edfa_w,
        ase_noise_W=imp.ase_noise_power_W,
        celestial_noise_W=sky_noise_w,
        inter_xt_noise_var=inter_xt_w,
        intra_xt_noise_var=intra_xt_w,
    )
    return {
        "solar_penalty_dB": max(0.0, sky_osnr_db - full_osnr_db),
        "xt_noise_W": inter_xt_w + intra_xt_w,
    }


def collect_impairment_rows(sim, results, load, strategy, duration_s, step_s):
    established = [r for r in results if r.accepted]
    time_points = []
    t = 0.0
    while t <= duration_s + 1e-9:
        time_points.append(round(t, 9))
        t += step_s

    per_path = {
        r.request_id: {
            "load": load,
            "strategy": strategy,
            "request_id": r.request_id,
            "hops": r.total_hops,
            "max_distance_km": 0.0,
            "max_fsl_dB": 0.0,
            "max_path_fsl_sum_dB": 0.0,
            "max_solar_penalty_dB": 0.0,
            "max_doppler_penalty_mdB": 0.0,
            "max_edfa_gain_deg_dB": 0.0,
            "max_edfa_nf_inc_dB": 0.0,
            "max_xt_noise_dBm": -300.0,
            "min_osnr_dB": 300.0,
            "osnr_values": [],
        }
        for r in established
    }

    original_time = sim.config.sim_time_seconds
    for time_s in time_points:
        sim._refresh_impairments_at_time(time_s)
        link_osnr_db = sim._build_link_osnr_lookup()
        for result in established:
            row = per_path[result.request_id]
            osnr_db = path_osnr_from_lookup(sim, result, link_osnr_db)
            row["osnr_values"].append(osnr_db)
            row["min_osnr_dB"] = min(row["min_osnr_dB"], osnr_db)
            path_fsl_sum_dB = 0.0
            for link_id in result.path_links:
                imp = sim._impairment_cache[link_id]
                terms = link_channel_terms(sim, link_id, result.wavelength_channel)
                path_fsl_sum_dB += imp.free_space_loss_dB
                row["max_distance_km"] = max(row["max_distance_km"], imp.link_distance_km)
                row["max_fsl_dB"] = max(row["max_fsl_dB"], imp.free_space_loss_dB)
                row["max_solar_penalty_dB"] = max(
                    row["max_solar_penalty_dB"], terms["solar_penalty_dB"]
                )
                row["max_doppler_penalty_mdB"] = max(
                    row["max_doppler_penalty_mdB"],
                    1000.0 * imp.doppler_filter_penalty_dB,
                )
                row["max_edfa_gain_deg_dB"] = max(
                    row["max_edfa_gain_deg_dB"], imp.edfa_gain_degradation_dB
                )
                row["max_edfa_nf_inc_dB"] = max(
                    row["max_edfa_nf_inc_dB"], imp.edfa_nf_increase_dB
                )
                row["max_xt_noise_dBm"] = max(
                    row["max_xt_noise_dBm"], w_to_dbm(terms["xt_noise_W"])
                )
            row["max_path_fsl_sum_dB"] = max(row["max_path_fsl_sum_dB"], path_fsl_sum_dB)

    sim._refresh_impairments_at_time(original_time)

    rows = []
    for row in per_path.values():
        osnrs = row.pop("osnr_values")
        row["osnr_range_dB"] = max(osnrs) - min(osnrs) if osnrs else 0.0
        row["osnr_std_dB"] = float(np.std(osnrs)) if osnrs else 0.0
        rows.append(row)
    return rows


def run_load_strategy(base_config, requests, load, strategy_name, strategy_value, duration_s, step_s):
    cfg = copy.deepcopy(base_config)
    cfg.routing_strategy = strategy_value
    sim = QoTSimulation(cfg)
    results = sim.run_requests(requests, verbose=False)
    return collect_impairment_rows(sim, results, load, strategy_name, duration_s, step_s)


def plot_impairment_decomposition(rows, outdir):
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
        "legend.fontsize": 6.4,
        "axes.linewidth": 0.7,
    })
    loads = ["Low", "Medium", "High"]
    strategies = ["Baseline", "QoT-GLR"]
    colours = {"Baseline": "#555555", "QoT-GLR": "#0F4D92"}
    metrics = [
        ("max_path_fsl_sum_dB", "Cumulative free-space loss", "Path loss (dB)", "a"),
        ("max_solar_penalty_dB", "Maximum solar-background penalty", "OSNR penalty (dB)", "b"),
        ("max_doppler_penalty_mdB", "Maximum Doppler filtering loss", "Penalty (mdB)", "c"),
        ("max_edfa_gain_deg_dB", "Maximum EDFA gain degradation", "Gain loss (dB)", "d"),
        ("max_xt_noise_dBm", "Maximum crosstalk noise", "Noise power (dBm)", "e"),
        ("osnr_range_dB", "Temporal QoT fluctuation", "OSNR range (dB)", "f"),
    ]

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = fig.add_gridspec(
        2, 3,
        left=0.07,
        right=0.985,
        top=0.91,
        bottom=0.10,
        wspace=0.30,
        hspace=0.34,
    )
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    for ax, (metric, title, ylabel, panel) in zip(axes, metrics):
        data = []
        positions = []
        colors = []
        pos = 0.0
        for load in loads:
            for strategy in strategies:
                vals = [
                    float(r[metric]) for r in rows
                    if r["load"] == load and r["strategy"] == strategy
                ]
                data.append(vals)
                positions.append(pos)
                colors.append(colours[strategy])
                pos += 0.42
            pos += 0.36
        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.28,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "white", "linewidth": 0.9},
            whiskerprops={"linewidth": 0.7},
            capprops={"linewidth": 0.7},
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor(color)
            patch.set_alpha(0.82)
        centers = [(positions[i * 2] + positions[i * 2 + 1]) / 2 for i in range(len(loads))]
        ax.set_xticks(centers)
        ax.set_xticklabels(loads)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", lw=0.25, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=colours["Baseline"], lw=5),
        plt.Line2D([0], [0], color=colours["QoT-GLR"], lw=5),
    ]
    fig.legend(
        handles,
        ["Baseline SP+FF", "QoT-GLR"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.995),
        frameon=False,
    )
    base = os.path.join(outdir, "qot_glr_vs_baseline_impairment_decomposition")
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
    base_config = build_default_config()
    all_rows = []
    for load, n_req, seed_offset in [
        ("Low", args.low, 0),
        ("Medium", args.medium, 100),
        ("High", args.high, 200),
    ]:
        request_sim = QoTSimulation(copy.deepcopy(base_config))
        requests = create_demo_requests(
            request_sim.constellation.total_sats,
            num_requests=n_req,
            seed=args.seed + seed_offset,
        )
        for strategy_name, strategy_value in [
            ("Baseline", "baseline"),
            ("QoT-GLR", "qot_glr"),
        ]:
            rows = run_load_strategy(
                base_config,
                requests,
                load,
                strategy_name,
                strategy_value,
                args.observe_duration,
                args.observe_step,
            )
            all_rows.extend(rows)
            print(f"{load:>6} {strategy_name:>8}: {len(rows)} accepted lightpaths")

    csv_path = os.path.join(args.outdir, "qot_glr_vs_baseline_impairment_decomposition.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    plot_impairment_decomposition(all_rows, args.outdir)
    print(f"\nWrote {csv_path}")
    print(f"Wrote {os.path.join(args.outdir, 'qot_glr_vs_baseline_impairment_decomposition.svg')}")


if __name__ == "__main__":
    main()
