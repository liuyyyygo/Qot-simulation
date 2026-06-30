"""
Paper-quality figure generation for QoT simulation results.

Generates figures organized by three analysis dimensions:
  I.   Physical Layer Link Analysis (Figures 1-7)
  II.  Network Performance Analysis (Figures 8-12)
  III. QoT Constraint Analysis (Figures 13-16)

Uses Nature-journal visual standards: semantic colours, SVG output,
multi-panel architecture where appropriate.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, LogFormatter
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional
import math
import os

# ─── Nature-style plot defaults ───────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.format": "svg",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.2,
    "lines.markersize": 3,
})

# Nature semantic colour palette (colourblind-friendly)
C_INTRA = "#0072B2"
C_INTER = "#D55E00"
C_LOW   = "#009E73"
C_MED   = "#E69F00"
C_HIGH  = "#CC79A7"
C_REF   = "#000000"
C_SAA   = "#D41159"
C_SUN   = "#F0E442"
C_BER   = "#56B4E9"
COLOURS = [C_INTRA, C_INTER, C_LOW, C_MED, C_HIGH, C_SAA, C_SUN, C_BER]


class SimulationDataCollector:
    """Collects and organises simulation output for plotting."""

    def __init__(self, traffic_label: str = ""):
        self.label = traffic_label

        self.link_distances_km: List[float] = []
        self.link_fsl_dB: List[float] = []
        self.link_doppler_GHz: List[float] = []
        self.link_doppler_penalty_dB: List[float] = []
        self.link_sun_angle_deg: List[float] = []
        self.link_sun_blocked: List[bool] = []
        self.link_saa_risk: List[float] = []
        self.link_edfa_gain_deg_dB: List[float] = []
        self.link_edfa_nf_inc_dB: List[float] = []
        self.link_ase_noise_W: List[float] = []
        self.link_celestial_noise_W: List[float] = []
        self.link_osnr_dB: List[float] = []
        self.link_types: List[str] = []

        self.path_accepted: List[bool] = []
        self.path_osnr_dB: List[float] = []
        self.path_ber: List[float] = []
        self.path_q_factor: List[float] = []
        self.path_hops: List[int] = []
        self.path_wavelength: List[int] = []
        self.path_cumulative_dose_krad: List[float] = []
        self.path_per_link_osnr: List[List[float]] = []
        self.path_per_link_dist: List[List[float]] = []
        self.path_links: List[List[int]] = []
        self.path_satellites: List[List[int]] = []
        self.path_reject_reasons: List[str] = []
        self.time_qot_samples = []

        self.wavelength_occupancy: Dict[int, set] = defaultdict(set)

        self.sat_lat_deg: List[float] = []
        self.sat_lon_deg: List[float] = []
        self.sat_saa_risk: List[float] = []
        self.sat_dose_rate: List[float] = []
        self.sat_cumulative_dose: List[float] = []

    def collect_from_simulation(self, sim):
        """Collect all data from a completed QoTSimulation."""
        for lisl in sim.constellation.topology:
            imp = sim._impairment_cache.get(lisl.link_id)
            if imp is None:
                continue
            self.link_distances_km.append(imp.link_distance_km)
            self.link_fsl_dB.append(imp.free_space_loss_dB)
            self.link_doppler_GHz.append(imp.doppler_shift_GHz)
            self.link_doppler_penalty_dB.append(imp.doppler_filter_penalty_dB)
            self.link_sun_angle_deg.append(imp.sun_angle_deg)
            self.link_sun_blocked.append(imp.is_sun_blocked)
            self.link_saa_risk.append(imp.saa_risk_factor)
            self.link_edfa_gain_deg_dB.append(imp.edfa_gain_degradation_dB)
            self.link_edfa_nf_inc_dB.append(imp.edfa_nf_increase_dB)
            self.link_ase_noise_W.append(imp.ase_noise_power_W)
            self.link_celestial_noise_W.append(imp.celestial_noise_power_W)
            self.link_osnr_dB.append(imp.osnr_dB)
            self.link_types.append(lisl.link_type)

        for link_id, ch_set in sim.rwa.wavelength_occupancy.items():
            self.wavelength_occupancy[link_id] = set(ch_set)

        for sat_id in range(sim.constellation.total_sats):
            sat = sim.constellation.get_satellite(sat_id)
            self.sat_lat_deg.append(sat.lat_deg)
            self.sat_lon_deg.append(sat.lon_deg)
            dose_rate = sim.rwa_sat_to_dose_rate.get(sat_id, 0)
            self.sat_dose_rate.append(dose_rate)
            self.sat_saa_risk.append(
                sim.saa.compute_saa_risk_factor(sat.lat_deg, sat.lon_deg)
            )
            self.sat_cumulative_dose.append(sat.cumulative_dose_krad)

    def collect_results(self, results: list):
        """Collect lightpath results."""
        for r in results:
            self.path_accepted.append(r.accepted)
            self.path_reject_reasons.append(
                r.reject_reason.value if not r.accepted else "Accepted"
            )
            if r.accepted:
                self.path_osnr_dB.append(r.osnr_dB)
                self.path_ber.append(r.ber)
                self.path_q_factor.append(r.q_factor)
                self.path_hops.append(r.total_hops)
                self.path_wavelength.append(r.wavelength_channel)
                self.path_cumulative_dose_krad.append(r.cumulative_radiation_krad)
                self.path_per_link_osnr.append(r.per_link_osnr_dB)
                self.path_per_link_dist.append(r.per_link_distance_km)
                self.path_links.append(r.path_links)
                self.path_satellites.append(r.path_satellites)

    def collect_time_qot_samples(self, samples: list):
        """Collect QoT samples observed after lightpath establishment."""
        self.time_qot_samples = list(samples)

    @property
    def num_links(self) -> int:
        return len(self.link_distances_km)

    @property
    def num_accepted(self) -> int:
        return sum(self.path_accepted)

    @property
    def num_rejected(self) -> int:
        return len(self.path_accepted) - self.num_accepted

    @property
    def acceptance_rate(self) -> float:
        total = len(self.path_accepted)
        return self.num_accepted / total if total > 0 else 0.0

    def get_rejection_reasons(self) -> Counter:
        return Counter(self.path_reject_reasons)


# ═══════════════════════════════════════════════════════════════════════════
# I. PHYSICAL LAYER LINK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def _loaded_link_osnr_samples(data: SimulationDataCollector, sim) -> List[dict]:
    """Return occupied link-channel OSNR samples with load-aware crosstalk."""
    if sim is None or not data.wavelength_occupancy:
        return []

    link_obj = {lisl.link_id: lisl for lisl in sim.constellation.topology}
    samples = []

    def count_same_wavelength_interferers(target_lisl, channel: int) -> int:
        endpoint_sats = {target_lisl.sat_i, target_lisl.sat_j}
        interfering_links = set()
        for other_lisl in sim.constellation.topology:
            if other_lisl.link_id == target_lisl.link_id:
                continue
            if not ({other_lisl.sat_i, other_lisl.sat_j} & endpoint_sats):
                continue
            if channel in data.wavelength_occupancy.get(other_lisl.link_id, set()):
                interfering_links.add(other_lisl.link_id)
        return len(interfering_links)

    for link_id, channels in data.wavelength_occupancy.items():
        lisl = link_obj.get(link_id)
        imp = sim._impairment_cache.get(link_id)
        if lisl is None or imp is None:
            continue

        effective_gain_dB = sim.config.nominal_gain_dB - imp.edfa_gain_degradation_dB
        signal_after_edfa_W = imp.received_signal_power_W * 10.0 ** (effective_gain_dB / 10.0)
        occupied = list(channels)

        for channel in sorted(channels):
            inter_xt_var = sim.crosstalk.compute_inter_wavelength_noise_var(
                target_channel=channel,
                occupied_channels=occupied,
                received_power_per_channel_W=imp.received_signal_power_W,
                doppler_shifts={ch: imp.doppler_shift_GHz for ch in occupied},
            )
            intra_xt_var = sim.crosstalk.compute_intra_wavelength_noise_var(
                num_interfering_ports=count_same_wavelength_interferers(lisl, channel),
                received_power_W=imp.received_signal_power_W,
            )
            _, osnr_dB, _ = sim.osnr_calc.compute_single_link_osnr(
                signal_power_W=signal_after_edfa_W,
                ase_noise_W=imp.ase_noise_power_W,
                celestial_noise_W=imp.celestial_noise_power_W,
                inter_xt_noise_var=inter_xt_var,
                intra_xt_noise_var=intra_xt_var,
            )
            samples.append({
                "link_id": link_id,
                "channel": channel,
                "type": lisl.link_type,
                "distance_km": imp.link_distance_km,
                "sun_angle_deg": imp.sun_angle_deg,
                "osnr_dB": osnr_dB,
            })

    return samples


def fig1_link_osnr_distribution(data: SimulationDataCollector, sim=None, outdir: str = "."):
    """Figure 1: Single-link OSNR CDF - intra vs inter orbit."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    loaded_samples = _loaded_link_osnr_samples(data, sim)
    if loaded_samples:
        intra_osnr = [s["osnr_dB"] for s in loaded_samples if s["type"] == "intra_orbit"]
        inter_osnr = [s["osnr_dB"] for s in loaded_samples if s["type"] == "inter_orbit"]
    else:
        intra_osnr = [o for o, t in zip(data.link_osnr_dB, data.link_types) if t == "intra_orbit"]
        inter_osnr = [o for o, t in zip(data.link_osnr_dB, data.link_types) if t == "inter_orbit"]

    for vals, label, colour in [
        (intra_osnr, "Intra-orbit", C_INTRA),
        (inter_osnr, "Inter-orbit", C_INTER),
    ]:
        if not vals:
            continue
        sorted_vals = np.sort(vals)
        cdf = np.linspace(0, 1, len(sorted_vals))
        ax.plot(sorted_vals, cdf, color=colour, label=f"{label} (n={len(vals)})", lw=1.5)

    ax.set_xlabel("Single-link OSNR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title(f"a  Single-link OSNR distribution - {data.label}")
    ax.legend(loc="lower right", frameon=False)
    all_vals = intra_osnr + inter_osnr
    if all_vals:
        xmin = math.floor(min(all_vals) / 5.0) * 5.0
        xmax = math.ceil(max(all_vals) / 5.0) * 5.0
        ax.set_xlim(xmin, xmax)
        if xmin < 0 < xmax:
            ax.axvline(0.0, color=C_REF, ls=":", lw=0.6, alpha=0.7)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_link_osnr_cdf.svg"))
    plt.close(fig)


def fig2_distance_vs_osnr(data: SimulationDataCollector, sim=None, outdir: str = "."):
    """Figure 2: Link distance vs OSNR scatter with sun angle colour."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    loaded_samples = _loaded_link_osnr_samples(data, sim)
    if loaded_samples:
        link_types = [s["type"] for s in loaded_samples]
        dists = np.array([s["distance_km"] for s in loaded_samples])
        osnrs = np.array([s["osnr_dB"] for s in loaded_samples])
        sun_angles = np.array([s["sun_angle_deg"] for s in loaded_samples])
    else:
        link_types = data.link_types
        dists = np.array(data.link_distances_km)
        osnrs = np.array(data.link_osnr_dB)
        sun_angles = np.array(data.link_sun_angle_deg)

    intra_mask = np.array([t == "intra_orbit" for t in link_types])
    inter_mask = ~intra_mask

    if inter_mask.sum() > 0:
        sc = ax.scatter(
            dists[inter_mask], osnrs[inter_mask],
            c=sun_angles[inter_mask], cmap="plasma",
            s=12, alpha=0.7, edgecolors="none", label="Inter-orbit",
            vmin=0, vmax=90,
        )
        cbar = fig.colorbar(sc, ax=ax, label="Sun angle (deg)")
        cbar.ax.tick_params(labelsize=6)

    if intra_mask.sum() > 0:
        ax.scatter(
            dists[intra_mask], osnrs[intra_mask],
            c=C_INTRA, s=12, alpha=0.6, edgecolors="none",
            marker="s", label="Intra-orbit",
        )

    d_ref = np.linspace(500, 6000, 100)
    osnr_ref = 28.0 - 40.0 * np.log10(d_ref / 1000.0)
    ax.plot(d_ref, osnr_ref, "--", color=C_REF, lw=0.8,
            label=r"FSL scaling ($-40\log_{10}d$)")

    ax.set_xlabel("Link distance (km)")
    ax.set_ylabel("Single-link OSNR (dB)")
    ax.set_title(f"b  Link distance vs OSNR - {data.label}")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.set_xlim(0, 6000)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_distance_vs_osnr.svg"))
    plt.close(fig)


def fig3_doppler_distribution(data: SimulationDataCollector, outdir: str = "."):
    """Figure 3: Doppler shift histogram - intra vs inter orbit."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    for ax, link_type, colour, title in [
        (ax1, "intra_orbit", C_INTRA, "Intra-orbit"),
        (ax2, "inter_orbit", C_INTER, "Inter-orbit"),
    ]:
        dopplers = [abs(d) for d, t in zip(data.link_doppler_GHz, data.link_types) if t == link_type]
        if dopplers:
            ax.hist(dopplers, bins=20, color=colour, alpha=0.7, edgecolor="white", lw=0.3)
        ax.set_xlabel("|Doppler shift| (GHz)")
        ax.set_ylabel("Link count")
        ax.set_title(title, fontsize=8)

    fig.suptitle(f"c  Doppler frequency shift distribution - {data.label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_doppler_distribution.svg"))
    plt.close(fig)


def fig4_doppler_penalty_vs_shift(
    data: SimulationDataCollector,
    sim_config=None,
    outdir: str = ".",
):
    """Figure 4: Doppler filter penalty vs frequency shift."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    shifts = np.array([abs(d) for d in data.link_doppler_GHz])
    penalties_mdB = 1000.0 * np.array(data.link_doppler_penalty_dB)
    filter_bw = getattr(sim_config, "optical_filter_BW_GHz", 12.5)

    df_theory = np.linspace(0, max(shifts) * 1.1 if len(shifts) > 0 else 10, 100)
    b_eff = filter_bw / math.log(2.0) ** (1.0 / 4.0)
    penalty_theory_mdB = 1000.0 * -10.0 * np.log10(
        np.maximum(np.exp(-(df_theory / b_eff) ** 4), 1e-12)
    )

    ax.scatter(shifts, penalties_mdB, s=10, alpha=0.5, c=C_INTER, edgecolors="none")
    ax.plot(df_theory, penalty_theory_mdB, "--", color=C_REF, lw=0.8,
            label=f"Super-Gaussian (BW={filter_bw:g} GHz)")

    ax.set_xlabel("|Doppler shift| (GHz)")
    ax.set_ylabel("Filter power penalty (mdB)")
    ax.set_title(f"d  Doppler filter penalty - {data.label}")
    ax.legend(loc="lower right", frameon=False)
    ax.set_xlim(left=0)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_doppler_penalty.svg"))
    plt.close(fig)


def fig5_saa_radiation_field(data: SimulationDataCollector, sim_config, outdir: str = "."):
    """Figure 5: SAA radiation risk field heatmap with satellite positions."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    lons = np.linspace(-180, 180, 360)
    lats = np.linspace(-90, 90, 180)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    saa_center_lat = sim_config.saa_center_lat_deg
    saa_center_lon = sim_config.saa_center_lon_deg
    sigma_lat = sim_config.saa_lat_window_deg
    sigma_lon = sim_config.saa_lon_window_deg

    dlat = lat_grid - saa_center_lat
    # Use modular arithmetic for correct angular distance across ±180° boundary
    dlon = (lon_grid - saa_center_lon + 180) % 360 - 180

    risk_grid = np.exp(-0.5 * (dlat / sigma_lat) ** 2 - 0.5 * (dlon / sigma_lon) ** 2)

    im = ax.contourf(lon_grid, lat_grid, risk_grid, levels=20,
                     cmap="YlOrRd", alpha=0.8)
    cbar = fig.colorbar(im, ax=ax, label="Normalised SAA risk")
    cbar.ax.tick_params(labelsize=6)

    if data.sat_lat_deg:
        ax.scatter(data.sat_lon_deg, data.sat_lat_deg,
                   s=1.5, c=C_INTRA, alpha=0.6, edgecolors="none",
                   label=f"Satellites (n={len(data.sat_lat_deg)})")

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"e  SAA radiation risk field - {data.label}")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.legend(loc="lower right", frameon=False, fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig5_saa_radiation_field.svg"))
    plt.close(fig)


def fig6_edfa_gain_degradation(data: SimulationDataCollector, outdir: str = "."):
    """Figure 6: EDFA gain and noise-figure radiation degradation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    if data.link_edfa_gain_deg_dB:
        ax1.hist(data.link_edfa_gain_deg_dB, bins=30,
                 color=C_SAA, alpha=0.7, edgecolor="white", lw=0.3)
        ax1.axvline(x=np.mean(data.link_edfa_gain_deg_dB),
                    color=C_REF, ls="--", lw=0.6)
    ax1.set_xlabel("Gain degradation (dB)")
    ax1.set_ylabel("Link count")
    ax1.set_title("a  EDFA gain loss")

    if data.link_edfa_nf_inc_dB:
        ax2.hist(data.link_edfa_nf_inc_dB, bins=30,
                 color=C_MED, alpha=0.7, edgecolor="white", lw=0.3)
        ax2.axvline(x=np.mean(data.link_edfa_nf_inc_dB),
                    color=C_REF, ls="--", lw=0.6)
    ax2.set_xlabel("NF increase (dB)")
    ax2.set_ylabel("Link count")
    ax2.set_title("b  EDFA noise-figure rise")

    fig.suptitle(f"f  EDFA radiation degradation - {data.label}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig6_edfa_degradation.svg"))
    plt.close(fig)


def fig7_sun_angle_noise(data: SimulationDataCollector, outdir: str = "."):
    """Figure 7: Sun angle vs celestial background noise."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    angles = np.array(data.link_sun_angle_deg)
    noise = np.array(data.link_celestial_noise_W)

    noise_dBm = 10.0 * np.log10(np.maximum(noise, 1e-30)) + 30.0
    blocked = np.array(data.link_sun_blocked)

    ax.scatter(angles[~blocked], noise_dBm[~blocked],
               s=10, alpha=0.5, c=C_INTRA, edgecolors="none",
               label="Clear")
    if blocked.sum() > 0:
        ax.scatter(angles[blocked], noise_dBm[blocked],
                   s=12, alpha=0.7, c=C_SAA, edgecolors="none",
                   marker="x", label="Sun-blocked")

    ax.axvline(x=3.0, color=C_REF, ls="--", lw=0.8, label="3 deg avoidance")

    ax.set_xlabel("Sun-receiver angle (deg)")
    ax.set_ylabel("Celestial noise power (dBm)")
    ax.set_title(f"g  Celestial background noise - {data.label}")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig7_sun_angle_noise.svg"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# II. NETWORK PERFORMANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def fig8_path_osnr_cdf(data_dict: Dict[str, SimulationDataCollector], outdir: str = "."):
    """Figure 8: End-to-end OSNR CDF at multiple traffic loads."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    all_osnr = []
    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_osnr_dB:
            continue
        all_osnr.extend(d.path_osnr_dB)
        sorted_osnr = np.sort(d.path_osnr_dB)
        cdf = np.linspace(0, 1, len(sorted_osnr))
        ax.plot(sorted_osnr, cdf, color=colour, lw=1.5,
                label=f"{label} load (n={len(sorted_osnr)})")

    ax.set_xlabel("End-to-end OSNR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("End-to-end lightpath OSNR distribution")
    ax.legend(loc="upper left", frameon=False)
    if all_osnr:
        xmin = math.floor(min(all_osnr) / 5.0) * 5.0
        xmax = math.ceil(max(all_osnr) / 5.0) * 5.0
        ax.set_xlim(xmin, xmax)
        if xmin < 0 < xmax:
            ax.axvline(0.0, color=C_REF, ls=":", lw=0.6, alpha=0.7)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_path_osnr_cdf.svg"))
    plt.close(fig)


def fig9_osnr_vs_hops(data_dict: Dict[str, SimulationDataCollector], outdir: str = "."):
    """Figure 9: Time-window OSNR distribution grouped by hop count."""
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), sharey=True)

    max_hops_all = 0
    plotted_any = False
    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        ax = axes[["Low", "Medium", "High"].index(label)]
        d = data_dict.get(label)
        if d is None:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="#666666")
            ax.set_title(f"{label} load")
            continue

        hop_groups = defaultdict(list)
        if d.time_qot_samples:
            for sample in d.time_qot_samples:
                hops = len(sample.path_links)
                if hops > 0 and math.isfinite(sample.osnr_dB):
                    hop_groups[hops].append(sample.osnr_dB)
        else:
            for hops, osnr in zip(d.path_hops, d.path_osnr_dB):
                if hops > 0 and math.isfinite(osnr):
                    hop_groups[hops].append(osnr)

        hop_counts = sorted(hop_groups.keys())
        if not hop_counts:
            ax.text(0.5, 0.5, "No path samples", transform=ax.transAxes,
                    ha="center", va="center", color="#666666")
            ax.set_title(f"{label} load")
            continue

        grouped_osnr = [hop_groups[h] for h in hop_counts]
        bp = ax.boxplot(
            grouped_osnr,
            positions=hop_counts,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": C_REF, "linewidth": 1.0},
            whiskerprops={"color": "#555555", "linewidth": 0.8},
            capprops={"color": "#555555", "linewidth": 0.8},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.45)
            patch.set_edgecolor("#333333")

        medians = [np.median(values) for values in grouped_osnr]
        ax.plot(hop_counts, medians, color=colour, marker="o", ms=3,
                lw=1.0, label="Median")
        for h, values in zip(hop_counts, grouped_osnr):
            q95 = np.percentile(values, 95)
            ax.text(h, q95 + 0.25, f"n={len(values)}",
                    ha="center", va="bottom", fontsize=5.5,
                    color="#555555", rotation=90)

        max_hops_all = max(max_hops_all, max(hop_counts))
        plotted_any = True
        ax.set_title(f"{label} load")
        ax.set_xlabel("Hop count")
        ax.set_xticks(hop_counts)
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        ax.legend(loc="upper right", frameon=False, fontsize=6.0)

    if max_hops_all > 0:
        h_theory = np.arange(1, max_hops_all + 1)
        osnr_theory = 25.0 - 10.0 * np.log10(h_theory)
        for ax in axes:
            ax.plot(h_theory, osnr_theory, "--", color=C_REF, lw=0.7,
                    alpha=0.55, label=r"Fixed-link $-10\log_{10}H$")
            ax.set_xlim(0.5, max_hops_all + 0.5)

    axes[0].set_ylabel("Path OSNR over observation window (dB)")
    if plotted_any:
        handles, labels = axes[-1].get_legend_handles_labels()
        if handles:
            unique = dict(zip(labels, handles))
            axes[-1].legend(unique.values(), unique.keys(),
                            loc="upper right", frameon=False, fontsize=5.8)

    fig.suptitle("b  Time-window end-to-end OSNR distribution by hop count", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig9_osnr_vs_hops.svg"))
    plt.close(fig)


def fig10_ber_distribution(
    data_dict: Dict[str, SimulationDataCollector],
    use_fec: bool = True,
    outdir: str = ".",
):
    """Figure 10: BER CDF with QoT and FEC threshold lines."""
    fig, ax = plt.subplots(figsize=(4.6, 2.8))

    fec_threshold = 2.0e-3 if use_fec else 1.0e-12
    qot_threshold = 1.0e-7
    x_floor = -15.0

    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_ber:
            continue
        log_ber = np.log10(np.maximum(np.array(d.path_ber), 1e-300))
        log_ber = np.maximum(log_ber, x_floor)
        sorted_ber = np.sort(log_ber)
        cdf = np.linspace(0, 1, len(sorted_ber))
        ax.plot(sorted_ber, cdf, color=colour, lw=1.5,
                label=f"{label} load (n={len(sorted_ber)})")

    ax.axvline(x=np.log10(fec_threshold), color=C_SAA, ls="--", lw=0.8,
               label=f"FEC threshold ({fec_threshold:.0e})")
    ax.axvline(x=np.log10(qot_threshold), color=C_BER, ls="-.", lw=0.9,
               label="QoT degradation (1e-7)")
    if use_fec:
        ax.axvline(x=np.log10(1e-12), color=C_REF, ls=":", lw=0.6,
                   label="No-FEC threshold (1e-12)")

    ax.axvline(x=x_floor, color="#888888", ls=":", lw=0.6)
    ax.text(
        x_floor + 0.15, 0.04,
        r"$\log_{10}(\mathrm{BER})<-15$ clipped",
        rotation=90, ha="left", va="bottom", fontsize=5.8, color="#666666",
    )

    ax.set_xlabel("log10(BER), right-focused and clipped at -15")
    ax.set_ylabel("CDF")
    ax.set_title("c  End-to-end BER distribution")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(x_floor, 0.2)
    ax.set_xticks([-15, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, 0])
    ax.tick_params(axis="x", labelrotation=35)
    ax.grid(axis="x", alpha=0.12, lw=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig10_ber_distribution.svg"))
    plt.close(fig)


def fig11_acceptance_rate(data_dict: Dict[str, SimulationDataCollector], outdir: str = "."):
    """Figure 11: Acceptance rate vs traffic load."""
    fig, ax = plt.subplots(figsize=(3.5, 2.4))

    labels = []
    rates = []
    colours = []
    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None:
            continue
        labels.append(f"{label}\n({len(d.path_accepted)} req)")
        rates.append(d.acceptance_rate * 100)
        colours.append(colour)

    bars = ax.bar(labels, rates, color=colours, alpha=0.85, width=0.5)

    for bar, rate, label in zip(bars, rates, labels):
        d = data_dict.get(label.split("\n")[0])
        if d:
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 1,
                    f"{d.num_accepted}/{len(d.path_accepted)}",
                    ha="center", fontsize=7)

    ax.set_ylabel("Acceptance rate (%)")
    ax.set_title("d  Lightpath acceptance rate vs load")
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color=C_REF, ls=":", lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig11_acceptance_rate.svg"))
    plt.close(fig)


def fig12_wavelength_occupancy_heatmap(
    data: SimulationDataCollector,
    outdir: str = ".",
    max_links: int = 60,
):
    """Figure 12: Wavelength occupancy heatmap."""
    fig, ax = plt.subplots(figsize=(5.0, 3.0))

    if not data.wavelength_occupancy:
        ax.text(0.5, 0.5, "No occupancy data", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig12_wavelength_occupancy.svg"))
        plt.close(fig)
        return

    occupied_links = sorted(
        [lid for lid, chs in data.wavelength_occupancy.items() if chs],
        key=lambda lid: len(data.wavelength_occupancy[lid]),
        reverse=True,
    )[:max_links]

    if not occupied_links:
        ax.text(0.5, 0.5, "All wavelengths free", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig12_wavelength_occupancy.svg"))
        plt.close(fig)
        return

    all_channels = set()
    for chs in data.wavelength_occupancy.values():
        all_channels.update(chs)
    max_ch = max(all_channels) if all_channels else 39
    num_ch_display = min(max_ch + 1, 40)

    grid = np.zeros((len(occupied_links), num_ch_display))
    for i, lid in enumerate(occupied_links):
        for ch in data.wavelength_occupancy[lid]:
            if ch < num_ch_display:
                grid[i, ch] = 1.0

    im = ax.imshow(grid, aspect="auto", cmap="Blues",
                   interpolation="nearest", vmin=0, vmax=1)

    ax.set_xlabel("Wavelength channel")
    ax.set_ylabel(f"Link ID (top {len(occupied_links)} most loaded)")
    ax.set_title(f"e  Wavelength occupancy - {data.label}")

    step = max(1, len(occupied_links) // 10)
    ax.set_yticks(range(0, len(occupied_links), step))
    ax.set_yticklabels([str(occupied_links[i]) for i in range(0, len(occupied_links), step)])

    cbar = fig.colorbar(im, ax=ax, label="Occupied", ticks=[0, 1])
    cbar.ax.set_yticklabels(["Free", "Busy"])

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig12_wavelength_occupancy.svg"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# III. QoT CONSTRAINT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def fig13_rejection_reasons(data_dict: Dict[str, SimulationDataCollector], outdir: str = "."):
    """Figure 13: Rejection reason distribution across loads."""
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

    for ax, (label, colour) in zip(
        axes, [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]
    ):
        d = data_dict.get(label)
        if d is None:
            ax.set_title(f"{label}")
            continue
        reasons = d.get_rejection_reasons()
        if "Accepted" in reasons:
            del reasons["Accepted"]

        if reasons:
            names = list(reasons.keys())
            counts = list(reasons.values())
            short_names = []
            for n in names:
                if len(n) > 35:
                    n = n[:32] + "..."
                short_names.append(n)
            ax.barh(range(len(short_names)), counts, color=colour, alpha=0.8, height=0.6)
            ax.set_yticks(range(len(short_names)))
            ax.set_yticklabels(short_names, fontsize=5.5)
        else:
            ax.text(0.5, 0.5, "All accepted", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7)

        ax.set_title(f"{label} load ({d.num_rejected} rejected)", fontsize=8)
        ax.set_xlabel("Count")

    fig.suptitle("a  Rejection reason distribution", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig13_rejection_reasons.svg"))
    plt.close(fig)


def fig14_constraint_margins(
    data: SimulationDataCollector,
    sim_config,
    outdir: str = ".",
):
    """Figure 14: Safety margins for each QoT constraint."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    if not data.path_osnr_dB:
        ax.text(0.5, 0.5, "No accepted paths", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig14_constraint_margins.svg"))
        plt.close(fig)
        return

    margins = {
        "OSNR": [],
        "Hop count": [],
        "Radiation": [],
    }

    for i, osnr_dB in enumerate(data.path_osnr_dB):
        margins["OSNR"].append(osnr_dB - 12.0)

        margins["Hop count"].append(
            sim_config.max_hops - data.path_hops[i]
        )

        if i < len(data.path_cumulative_dose_krad):
            margins["Radiation"].append(
                sim_config.max_cumulative_dose_krad - data.path_cumulative_dose_krad[i]
            )

    positions = range(len(margins))
    for pos, (name, vals) in zip(positions, margins.items()):
        if not vals:
            continue
        ax.boxplot(vals, positions=[pos], widths=0.5,
                   patch_artist=True,
                   boxprops=dict(facecolor=COLOURS[pos], alpha=0.6),
                   medianprops=dict(color=C_REF, lw=0.8),
                   flierprops=dict(markersize=3, markerfacecolor=C_REF))

    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels(margins.keys())
    ax.set_ylabel("Safety margin")
    ax.set_title(f"b  Constraint safety margins - {data.label}")
    ax.axhline(y=0, color=C_SAA, ls="--", lw=0.8, label="Constraint boundary")
    ax.legend(loc="lower left", frameon=False, fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig14_constraint_margins.svg"))
    plt.close(fig)


def fig15_hop_count_distribution(
    data_dict: Dict[str, SimulationDataCollector],
    outdir: str = ".",
):
    """Figure 15: Path hop count distribution across loads."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    all_hops = []
    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_hops:
            continue
        all_hops.extend(d.path_hops)

    max_hop = max(all_hops) if all_hops else 0
    bins = np.arange(0.5, max_hop + 2.5, 1)

    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_hops:
            continue
        ax.hist(d.path_hops, bins=bins, color=colour, alpha=0.5,
                label=f"{label} (mean={np.mean(d.path_hops):.1f})",
                edgecolor="white", lw=0.3)

    ax.axvline(x=12, color=C_SAA, ls="--", lw=0.8, label="Max hops = 12")
    ax.set_xlabel("Hop count")
    ax.set_ylabel("Path count")
    ax.set_title("c  Hop count distribution")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig15_hop_count_distribution.svg"))
    plt.close(fig)


def fig16_topology_overlay(
    data: SimulationDataCollector,
    sim,
    outdir: str = ".",
    max_paths: int = 10,
):
    """Figure 16: Constellation topology with overlaid lightpaths."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    for sat_id in range(sim.constellation.total_sats):
        sat = sim.constellation.get_satellite(sat_id)
        ax.plot(sat.lon_deg, sat.lat_deg, "o", ms=0.8,
                color="gray", alpha=0.3, mec="none")

    for lisl in sim.constellation.topology:
        sat_i = sim.constellation.get_satellite(lisl.sat_i)
        sat_j = sim.constellation.get_satellite(lisl.sat_j)
        ax.plot([sat_i.lon_deg, sat_j.lon_deg],
                [sat_i.lat_deg, sat_j.lat_deg],
                "-", lw=0.2, color="gray", alpha=0.2)

    if data.path_satellites:
        cmap = plt.cm.viridis
        displayed = 0
        for pi, path in enumerate(data.path_satellites):
            if displayed >= max_paths:
                break
            if len(path) < 2:
                continue
            lons = [sim.constellation.get_satellite(s).lon_deg for s in path]
            lats = [sim.constellation.get_satellite(s).lat_deg for s in path]

            colour = cmap(pi / min(len(data.path_satellites), max_paths))
            ax.plot(lons, lats, "-o", lw=0.8, ms=1.5, color=colour, alpha=0.7)
            displayed += 1

    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"d  Network topology with lightpaths - {data.label}")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig16_topology_overlay.svg"))
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# COMPOSITE FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def fig17_impairment_contribution(
    data: SimulationDataCollector,
    sim,
    outdir: str = ".",
):
    """Figure 17: Impairment contribution for single-hop and multi-hop paths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    if not data.path_links:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No accepted paths", transform=ax.transAxes,
                    ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig17_impairment_contribution.svg"))
        plt.close(fig)
        return

    link_imp = {}
    link_obj = {}
    for lisl in sim.constellation.topology:
        link_obj[lisl.link_id] = lisl
        imp = sim._impairment_cache.get(lisl.link_id)
        if imp is not None:
            link_imp[lisl.link_id] = imp

    loss_groups = {
        "Single-hop": defaultdict(list),
        "Multi-hop": defaultdict(list),
    }
    noise_groups = {
        "Single-hop": defaultdict(list),
        "Multi-hop": defaultdict(list),
    }

    def count_same_wavelength_interferers(target_lisl, channel: int) -> int:
        endpoint_sats = {target_lisl.sat_i, target_lisl.sat_j}
        interfering_links = set()
        for other_lisl in sim.constellation.topology:
            if other_lisl.link_id == target_lisl.link_id:
                continue
            if not ({other_lisl.sat_i, other_lisl.sat_j} & endpoint_sats):
                continue
            if channel in data.wavelength_occupancy.get(other_lisl.link_id, set()):
                interfering_links.add(other_lisl.link_id)
        return len(interfering_links)

    for path_links, wavelength in zip(data.path_links, data.path_wavelength):
        group = "Single-hop" if len(path_links) == 1 else "Multi-hop"
        fsl = doppler = edfa = fixed = 0.0
        ase = celestial = xtalk = thermal = 0.0

        for link_id in path_links:
            imp = link_imp.get(link_id)
            lisl = link_obj.get(link_id)
            if imp is None or lisl is None:
                continue
            fsl += imp.free_space_loss_dB
            doppler += imp.doppler_filter_penalty_dB
            edfa += imp.edfa_gain_degradation_dB
            fixed += sim.config.pointing_loss_dB + sim.config.oxc_insertion_loss_dB
            ase += imp.ase_noise_power_W
            celestial += imp.celestial_noise_power_W
            occupied = list(data.wavelength_occupancy.get(link_id, set()))
            inter_xt_var = sim.crosstalk.compute_inter_wavelength_noise_var(
                target_channel=wavelength,
                occupied_channels=occupied,
                received_power_per_channel_W=imp.received_signal_power_W,
                doppler_shifts={ch: imp.doppler_shift_GHz for ch in occupied},
            )
            intra_xt_var = sim.crosstalk.compute_intra_wavelength_noise_var(
                num_interfering_ports=count_same_wavelength_interferers(lisl, wavelength),
                received_power_W=imp.received_signal_power_W,
            )
            xtalk += math.sqrt(max(0.0, inter_xt_var + intra_xt_var))
            thermal += getattr(sim.osnr_calc, "thermal_noise_W", 0.0)

        loss_groups[group]["FSL"].append(fsl)
        loss_groups[group]["Doppler"].append(doppler)
        loss_groups[group]["Pointing/OXC"].append(fixed)
        loss_groups[group]["EDFA rad."].append(edfa)
        noise_groups[group]["ASE"].append(ase)
        noise_groups[group]["Celestial"].append(celestial)
        noise_groups[group]["Crosstalk"].append(xtalk)
        noise_groups[group]["Thermal"].append(thermal)

    loss_names = ["FSL", "Doppler", "Pointing/OXC", "EDFA rad."]
    noise_names = ["ASE", "Celestial", "Crosstalk", "Thermal"]
    groups = [("Single-hop", C_INTRA), ("Multi-hop", C_INTER)]
    width = 0.35

    x = np.arange(len(loss_names))
    for offset, (group, colour) in [(-width / 2, groups[0]), (width / 2, groups[1])]:
        means = [
            np.mean(loss_groups[group][name]) if loss_groups[group][name] else 0.0
            for name in loss_names
        ]
        ax1.bar(x + offset, means, width=width, color=colour, alpha=0.78, label=group)

    ax1.set_xticks(x)
    ax1.set_xticklabels(loss_names, rotation=25, ha="right")
    ax1.set_ylabel("Mean accumulated penalty (dB)")
    ax1.set_title("a  Power-loss terms")
    ax1.legend(frameon=False, fontsize=6.5)

    x2 = np.arange(len(noise_names))
    for offset, (group, colour) in [(-width / 2, groups[0]), (width / 2, groups[1])]:
        means = []
        for name in noise_names:
            vals = noise_groups[group][name]
            means.append(np.mean(vals) if vals else 0.0)
        ax2.bar(x2 + offset, means, width=width, color=colour, alpha=0.78, label=group)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(noise_names, rotation=25, ha="right")
    ax2.set_yscale("log")
    ax2.set_ylabel("Mean accumulated equivalent noise (W)")
    ax2.set_title("b  Noise-equivalent power")
    ax2.legend(frameon=False, fontsize=6.5)

    fig.suptitle("Impairment contribution in established lightpaths", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig17_impairment_contribution.svg"))
    plt.close(fig)


def fig18_celestial_osnr_penalty_heatmap(
    sim,
    outdir: str = ".",
):
    """
    Figure 18: Inter-orbit LISL solar-background risk map.

    Rows are the transmitter satellite index within an orbital plane, columns
    are transmitter orbital-plane IDs.  Under the nominal receiver FOV/noise
    floor, celestial-background OSNR loss can be numerically invisible, so the
    figure reports the solar geometry that drives that loss rather than forcing
    a near-zero OSNR-penalty heatmap.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)

    n_planes = sim.constellation.N_planes
    sats_per_plane = sim.constellation.sats_per_plane
    angle_grid = np.full((sats_per_plane, n_planes), np.nan)
    penalty_grid = np.full((sats_per_plane, n_planes), np.nan)

    for lisl in sim.constellation.topology:
        if lisl.link_type != "inter_orbit":
            continue
        imp = sim._impairment_cache.get(lisl.link_id)
        if imp is None:
            continue

        sat = sim.constellation.get_satellite(lisl.sat_i)
        noise_without_celestial = max(
            imp.total_noise_power_W - imp.celestial_noise_power_W,
            1e-30,
        )
        signal_power_for_osnr = imp.osnr_linear * imp.total_noise_power_W
        osnr_without_celestial = signal_power_for_osnr / noise_without_celestial
        osnr_without_celestial_dB = 10.0 * np.log10(max(osnr_without_celestial, 1e-30))
        penalty_dB = max(0.0, osnr_without_celestial_dB - imp.osnr_dB)

        row = sat.index_in_plane
        col = sat.plane_id
        current_angle = angle_grid[row, col]
        if not np.isfinite(current_angle) or imp.sun_angle_deg < current_angle:
            angle_grid[row, col] = imp.sun_angle_deg
        current_penalty = penalty_grid[row, col]
        if not np.isfinite(current_penalty) or penalty_dB > current_penalty:
            penalty_grid[row, col] = penalty_dB

    masked_angle = np.ma.masked_invalid(angle_grid)
    im1 = ax1.imshow(masked_angle, origin="lower", aspect="auto", cmap="viridis_r",
                     vmin=0, vmax=180)
    cbar1 = fig.colorbar(im1, ax=ax1)
    cbar1.set_label("Minimum sun-link angle (deg)")
    cbar1.ax.tick_params(labelsize=6)

    risk_window_deg = max(10.0, getattr(sim.config, "sun_avoidance_angle_deg", 3.0) + 7.0)
    risk_grid = np.where(
        np.isfinite(angle_grid),
        np.maximum(0.0, risk_window_deg - angle_grid),
        np.nan,
    )
    masked_risk = np.ma.masked_invalid(risk_grid)
    vmax_risk = max(risk_window_deg, float(np.nanmax(risk_grid)) if np.isfinite(risk_grid).any() else 1.0)
    im2 = ax2.imshow(masked_risk, origin="lower", aspect="auto", cmap="magma",
                     vmin=0, vmax=vmax_risk)
    cbar2 = fig.colorbar(im2, ax=ax2)
    cbar2.set_label(f"Solar proximity risk, max(0, {risk_window_deg:.0f} deg - angle)")
    cbar2.ax.tick_params(labelsize=6)

    if np.isfinite(angle_grid).any():
        yy, xx = np.indices(angle_grid.shape)
        close_mask = np.isfinite(angle_grid) & (angle_grid <= risk_window_deg)
        blocked_mask = np.isfinite(angle_grid) & (
            angle_grid <= getattr(sim.config, "sun_avoidance_angle_deg", 3.0)
        )
        ax1.scatter(xx[close_mask], yy[close_mask], s=12, facecolors="none",
                    edgecolors="white", linewidths=0.7, label=f"<={risk_window_deg:.0f} deg")
        ax1.scatter(xx[blocked_mask], yy[blocked_mask], s=18, marker="x",
                    color="#D41159", linewidths=0.9, label="avoidance")
        if close_mask.any() or blocked_mask.any():
            ax1.legend(frameon=True, fontsize=6, loc="upper right")

    max_nominal_penalty = (
        float(np.nanmax(penalty_grid)) if np.isfinite(penalty_grid).any() else 0.0
    )
    min_angle = float(np.nanmin(angle_grid)) if np.isfinite(angle_grid).any() else float("nan")
    ax2.text(
        0.02, 0.98,
        f"Nominal OSNR penalty max: {max_nominal_penalty:.2e} dB\n"
        f"Minimum angle: {min_angle:.1f} deg",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 2.0},
    )

    ax1.set_xlabel("Transmitter orbital-plane ID")
    ax1.set_ylabel("Transmitter satellite index in plane")
    ax1.set_title("a  Inter-orbit LISL sun angle")
    ax2.set_xlabel("Transmitter orbital-plane ID")
    ax2.set_title("b  Solar-background risk geometry")

    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(n_planes))
        ax.set_yticks(np.arange(0, sats_per_plane, max(1, sats_per_plane // 11)))
        ax.grid(False)
    if n_planes > 24:
        ticks = np.arange(0, n_planes, max(1, n_planes // 12))
        ax1.set_xticks(ticks)
        ax2.set_xticks(ticks)

    fig.suptitle("Celestial-background exposure diagnostic for inter-orbit LISLs", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig18_celestial_osnr_penalty_heatmap.svg"))
    plt.close(fig)


def fig19_solar_background_osnr_time_series(
    sim,
    outdir: str = ".",
    duration_s: Optional[float] = None,
    step_s: float = 30.0,
    n_links: int = 3,
):
    """
    Figure 19: OSNR fluctuation caused by solar background noise.

    The function scans one orbital period by default, selects the inter-orbit
    LISLs that approach the Sun most closely, and compares OSNR with the full
    celestial background against a sky-only baseline.
    """
    if duration_s is None:
        duration_s = sim.constellation.orbital_period_s

    inter_links = [lisl for lisl in sim.constellation.topology
                   if lisl.link_type == "inter_orbit"]
    if not inter_links:
        fig, ax = plt.subplots(figsize=(4.8, 2.8))
        ax.text(0.5, 0.5, "No inter-orbit LISLs", transform=ax.transAxes,
                ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig19_solar_background_osnr_timeseries.svg"))
        plt.close(fig)
        return

    original_time = sim.config.sim_time_seconds
    time_points = np.arange(0.0, duration_s + 1e-9, step_s)
    min_angle_by_link = {lisl.link_id: 180.0 for lisl in inter_links}

    for t in time_points:
        sim._refresh_impairments_at_time(float(t))
        for lisl in inter_links:
            imp = sim._impairment_cache.get(lisl.link_id)
            if imp is not None:
                min_angle_by_link[lisl.link_id] = min(
                    min_angle_by_link[lisl.link_id],
                    imp.sun_angle_deg,
                )

    excluded_link_ids = {1, 30}
    selected_ids = [
        link_id for link_id, _ in
        sorted(min_angle_by_link.items(), key=lambda item: item[1])
        if link_id not in excluded_link_ids
    ][:n_links]
    selected_links = {lisl.link_id: lisl for lisl in inter_links
                      if lisl.link_id in selected_ids}

    traces = {
        link_id: {"time_min": [], "osnr_full": [], "osnr_sky": [],
                  "penalty": [], "sun_angle": []}
        for link_id in selected_ids
    }
    sky_noise_W = (
        sim.celestial.L_sky
        * sim.celestial.A_Omega_DeltaLambda
        * sim.config.optical_efficiency
    )

    for t in time_points:
        sim._refresh_impairments_at_time(float(t))
        for link_id in selected_ids:
            imp = sim._impairment_cache.get(link_id)
            if imp is None:
                continue

            effective_gain_dB = sim.config.nominal_gain_dB - imp.edfa_gain_degradation_dB
            signal_after_edfa_W = (
                imp.received_signal_power_W
                * 10.0 ** (effective_gain_dB / 10.0)
            )
            _, osnr_sky_dB, _ = sim.osnr_calc.compute_single_link_osnr(
                signal_power_W=signal_after_edfa_W,
                ase_noise_W=imp.ase_noise_power_W,
                celestial_noise_W=sky_noise_W,
                inter_xt_noise_var=imp.inter_xt_noise_var,
                intra_xt_noise_var=imp.intra_xt_noise_var,
            )

            traces[link_id]["time_min"].append(t / 60.0)
            traces[link_id]["osnr_full"].append(imp.osnr_dB)
            traces[link_id]["osnr_sky"].append(osnr_sky_dB)
            traces[link_id]["penalty"].append(max(0.0, osnr_sky_dB - imp.osnr_dB))
            traces[link_id]["sun_angle"].append(imp.sun_angle_deg)

    sim._refresh_impairments_at_time(original_time)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(7.2, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1.0, 1.0]},
    )

    colours = [C_INTER, C_INTRA, C_HIGH, C_MED, C_LOW]
    for idx, link_id in enumerate(selected_ids):
        trace = traces[link_id]
        if not trace["time_min"]:
            continue

        lisl = selected_links[link_id]
        label = f"L{link_id}: S{lisl.sat_i}-S{lisl.sat_j}"
        colour = colours[idx % len(colours)]
        t = trace["time_min"]

        ax1.plot(t, trace["osnr_full"], color=colour, label=label)
        ax1.plot(t, trace["osnr_sky"], color=colour, ls="--", lw=0.8, alpha=0.55)
        ax2.plot(t, trace["penalty"], color=colour, label=label)
        ax3.plot(t, trace["sun_angle"], color=colour, label=label)

    fov_deg = getattr(sim.config, "sun_avoidance_angle_deg", 3.0)
    risk_deg = fov_deg + getattr(sim.config, "sun_coupling_scale_deg", 5.0)
    ax3.axhline(fov_deg, color=C_SAA, ls="-", lw=0.8, label=f"FOV core ({fov_deg:.0f} deg)")
    ax3.axhline(risk_deg, color=C_REF, ls=":", lw=0.7, label=f"tail scale ({risk_deg:.0f} deg)")

    ax1.set_ylabel("Single-link OSNR (dB)")
    ax1.set_title("a  OSNR with solar background (solid) and sky-only baseline (dashed)")
    ax2.set_ylabel("Solar-induced\nOSNR penalty (dB)")
    ax2.set_title("b  OSNR loss attributable to solar background noise")
    ax3.set_ylabel("Sun-link angle (deg)")
    ax3.set_xlabel("Time from epoch (min)")
    ax3.set_title("c  Solar approach geometry")

    for ax in (ax1, ax2, ax3):
        ax.grid(True, lw=0.3, alpha=0.25)
    ax1.legend(frameon=False, fontsize=6.2, ncol=3, loc="best")
    ax3.legend(frameon=False, fontsize=6.2, ncol=3, loc="best")

    fig.suptitle("Solar-background-induced OSNR fluctuation for selected inter-orbit LISLs",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig19_solar_background_osnr_timeseries.svg"))
    plt.close(fig)


def fig20_temporal_link_impairment_envelope(
    sim,
    outdir: str = ".",
    duration_s: float = 3600.0,
    step_s: float = 300.0,
):
    """Figure 20: Time-window statistics for all physical LISLs."""
    original_time = sim.config.sim_time_seconds
    time_points = np.arange(0.0, duration_s + 1e-9, step_s)

    time_min = []
    osnr_p5 = []
    osnr_mean = []
    osnr_p95 = []
    sun_min = []
    doppler_p95 = []

    for t in time_points:
        sim._refresh_impairments_at_time(float(t))
        imps = list(sim._impairment_cache.values())
        if not imps:
            continue
        osnrs = np.array([imp.osnr_dB for imp in imps])
        sun_angles = np.array([imp.sun_angle_deg for imp in imps])
        dopplers = np.array([abs(imp.doppler_shift_GHz) for imp in imps])

        time_min.append(t / 60.0)
        osnr_p5.append(np.percentile(osnrs, 5))
        osnr_mean.append(np.mean(osnrs))
        osnr_p95.append(np.percentile(osnrs, 95))
        sun_min.append(np.min(sun_angles))
        doppler_p95.append(np.percentile(dopplers, 95))

    sim._refresh_impairments_at_time(original_time)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(7.2, 5.0), sharex=True,
        gridspec_kw={"height_ratios": [1.25, 1.0, 1.0]},
    )

    if time_min:
        ax1.fill_between(time_min, osnr_p5, osnr_p95, color=C_BER, alpha=0.22,
                         label="5-95% envelope")
        ax1.plot(time_min, osnr_mean, color=C_INTRA, label="Mean")
        ax2.plot(time_min, sun_min, color=C_SAA)
        ax2.axhline(sim.config.sun_avoidance_angle_deg, color=C_REF, ls="--", lw=0.8,
                    label="FOV criterion")
        ax3.plot(time_min, doppler_p95, color=C_INTER)

    ax1.set_ylabel("Single-link OSNR (dB)")
    ax1.set_title("a  Full-network single-link OSNR envelope")
    ax1.legend(frameon=False, fontsize=6.5)
    ax2.set_ylabel("Minimum sun angle (deg)")
    ax2.set_title("b  Worst solar geometry across LISLs")
    ax2.legend(frameon=False, fontsize=6.5)
    ax3.set_ylabel("95th |Doppler| (GHz)")
    ax3.set_xlabel("Time after setup (min)")
    ax3.set_title("c  Doppler temporal envelope")
    fig.suptitle("Time-window statistics of physical LISL impairments", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig20_temporal_link_impairment_envelope.svg"))
    plt.close(fig)


def fig21_temporal_path_qot_distribution(
    data_dict: Dict[str, SimulationDataCollector],
    outdir: str = ".",
):
    """Figure 21: Time-window QoT statistics for established lightpaths."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.9))

    labels = []
    osnr_data = []
    logber_data = []
    for label in ["Low", "Medium", "High"]:
        data = data_dict.get(label)
        if data is None or not data.time_qot_samples:
            continue
        labels.append(label)
        osnr_data.append([sample.osnr_dB for sample in data.time_qot_samples])
        logber_data.append([
            math.log10(max(sample.ber, 1e-300))
            for sample in data.time_qot_samples
        ])

    if osnr_data:
        bp1 = ax1.boxplot(osnr_data, labels=labels, patch_artist=True, showfliers=False)
        bp2 = ax2.boxplot(logber_data, labels=labels, patch_artist=True, showfliers=False)
        for bp in (bp1, bp2):
            for patch, colour in zip(bp["boxes"], [C_LOW, C_MED, C_HIGH]):
                patch.set_facecolor(colour)
                patch.set_alpha(0.45)
                patch.set_edgecolor("#333333")
    else:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, "No time-window QoT samples",
                    transform=ax.transAxes, ha="center", va="center")

    ax1.set_ylabel("Path OSNR over time (dB)")
    ax1.set_title("a  Established-lightpath OSNR")
    ax2.set_ylabel("log10(BER) over time")
    ax2.axhline(math.log10(1e-7), color=C_BER, ls="-.", lw=0.8,
                label="QoT degradation (1e-7)")
    ax2.axhline(math.log10(2e-3), color=C_SAA, ls="--", lw=0.8,
                label="HD-FEC (2e-3)")
    ax2.set_title("b  Established-lightpath BER")
    ax2.legend(frameon=False, fontsize=6.0, loc="lower right")
    fig.suptitle("Time-window QoT distribution after lightpath establishment", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig21_temporal_path_qot_distribution.svg"))
    plt.close(fig)


def fig22_temporal_path_violation_rate(
    data_dict: Dict[str, SimulationDataCollector],
    outdir: str = ".",
    ber_qot_threshold: float = 1e-7,
    ber_fec_threshold: float = 2e-3,
):
    """Figure 22: Time-varying BER violation ratio for established lightpaths."""
    fig, ax = plt.subplots(figsize=(4.8, 2.9))

    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        data = data_dict.get(label)
        if data is None or not data.time_qot_samples:
            continue
        by_time = defaultdict(list)
        for sample in data.time_qot_samples:
            by_time[sample.time_seconds].append(sample.ber)
        times = sorted(by_time.keys())
        qot_rates = [
            100.0 * sum(ber > ber_qot_threshold for ber in by_time[t]) / len(by_time[t])
            for t in times
        ]
        fec_rates = [
            100.0 * sum(ber > ber_fec_threshold for ber in by_time[t]) / len(by_time[t])
            for t in times
        ]
        time_min = [t / 60.0 for t in times]
        ax.plot(time_min, qot_rates, color=colour, label=f"{label}: BER>1e-7")
        ax.plot(time_min, fec_rates, color=colour, ls="--", lw=0.9,
                alpha=0.75, label=f"{label}: BER>2e-3")

    ax.set_xlabel("Time after setup (min)")
    ax.set_ylabel("Established paths exceeding threshold (%)")
    ax.set_ylim(-2, 102)
    ax.set_title("Temporal QoT violation ratio for established lightpaths")
    ax.legend(frameon=False, fontsize=5.8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig22_temporal_path_violation_rate.svg"))
    plt.close(fig)


def generate_all_figures(
    data_dict: Dict[str, SimulationDataCollector],
    sim_configs: Dict[str, object],
    sim_objects: Dict[str, object],
    outdir: str = "./figures",
):
    """
    Generate all 16 figures from simulation data.

    Args:
        data_dict: {"Low": Data, "Medium": Data, "High": Data}
        sim_configs: {"Low": config, ...}
        sim_objects: {"Low": QoTSimulation, ...}
        outdir: output directory for SVG files
    """
    os.makedirs(outdir, exist_ok=True)

    primary = data_dict.get("High") or data_dict.get("Medium") or data_dict.get("Low")
    primary_sim = sim_objects.get("High") or sim_objects.get("Medium") or sim_objects.get("Low")
    primary_config = sim_configs.get("High") or sim_configs.get("Medium") or sim_configs.get("Low")

    if primary is None:
        print("No data to plot!")
        return

    print("Generating physical layer figures (1-7)...")
    fig1_link_osnr_distribution(primary, primary_sim, outdir)
    fig2_distance_vs_osnr(primary, primary_sim, outdir)
    fig3_doppler_distribution(primary, outdir)
    fig4_doppler_penalty_vs_shift(primary, primary_config, outdir)
    if primary_config:
        fig5_saa_radiation_field(primary, primary_config, outdir)
    fig6_edfa_gain_degradation(primary, outdir)
    fig7_sun_angle_noise(primary, outdir)

    print("Generating network performance figures (8-12)...")
    fig8_path_osnr_cdf(data_dict, outdir)
    fig9_osnr_vs_hops(data_dict, outdir)
    fig10_ber_distribution(data_dict,
                           use_fec=primary_config.use_fec if primary_config else True,
                           outdir=outdir)
    fig11_acceptance_rate(data_dict, outdir)
    fig12_wavelength_occupancy_heatmap(primary, outdir)

    print("Generating constraint analysis figures (13-16)...")
    fig13_rejection_reasons(data_dict, outdir)
    if primary and primary_config:
        fig14_constraint_margins(primary, primary_config, outdir)
    fig15_hop_count_distribution(data_dict, outdir)
    if primary_sim:
        fig16_topology_overlay(primary, primary_sim, outdir)
        fig17_impairment_contribution(primary, primary_sim, outdir)
        fig18_celestial_osnr_penalty_heatmap(primary_sim, outdir)
        fig19_solar_background_osnr_time_series(primary_sim, outdir)
        fig20_temporal_link_impairment_envelope(primary_sim, outdir)
        fig21_temporal_path_qot_distribution(data_dict, outdir)
        fig22_temporal_path_violation_rate(data_dict, outdir)

    print(f"All figures saved to {outdir}/")
