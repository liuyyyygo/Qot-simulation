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
        self.path_satellites: List[List[int]] = []
        self.path_reject_reasons: List[str] = []

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
                self.path_satellites.append(r.path_satellites)

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

def fig1_link_osnr_distribution(data: SimulationDataCollector, outdir: str = "."):
    """Figure 1: Single-link OSNR CDF - intra vs inter orbit."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

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
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_link_osnr_cdf.svg"))
    plt.close(fig)


def fig2_distance_vs_osnr(data: SimulationDataCollector, outdir: str = "."):
    """Figure 2: Link distance vs OSNR scatter with sun angle colour."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    intra_mask = np.array([t == "intra_orbit" for t in data.link_types])
    inter_mask = ~intra_mask

    dists = np.array(data.link_distances_km)
    osnrs = np.array(data.link_osnr_dB)
    sun_angles = np.array(data.link_sun_angle_deg)

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


def fig4_doppler_penalty_vs_shift(data: SimulationDataCollector, outdir: str = "."):
    """Figure 4: Doppler filter penalty vs frequency shift."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    shifts = np.array([abs(d) for d in data.link_doppler_GHz])
    penalties = np.array(data.link_doppler_penalty_dB)

    df_theory = np.linspace(0, max(shifts) * 1.1 if len(shifts) > 0 else 10, 100)
    b_eff_50 = 50.0 / math.log(2.0) ** (1.0 / 4.0)
    penalty_50 = -10.0 * np.log10(np.maximum(np.exp(-(df_theory / b_eff_50) ** 4), 1e-12))

    ax.scatter(shifts, penalties, s=10, alpha=0.5, c=C_INTER, edgecolors="none")
    ax.plot(df_theory, penalty_50, "--", color=C_REF, lw=0.8,
            label="Super-Gaussian (n=2, BW=50 GHz)")

    ax.set_xlabel("|Doppler shift| (GHz)")
    ax.set_ylabel("Filter power penalty (dB)")
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
    """Figure 6: EDFA gain degradation distribution."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))

    if data.link_edfa_gain_deg_dB:
        ax.hist(data.link_edfa_gain_deg_dB, bins=30,
                color=C_SAA, alpha=0.7, edgecolor="white", lw=0.3)
        ax.set_xlabel("EDFA gain degradation (dB)")
        ax.set_ylabel("Link count")
        if data.link_edfa_gain_deg_dB:
            ax.axvline(x=np.mean(data.link_edfa_gain_deg_dB),
                       color=C_REF, ls="--", lw=0.6)

    ax.set_title(f"f  EDFA gain degradation - {data.label}")
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

    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_osnr_dB:
            continue
        sorted_osnr = np.sort(d.path_osnr_dB)
        cdf = np.linspace(0, 1, len(sorted_osnr))
        ax.plot(sorted_osnr, cdf, color=colour, lw=1.5,
                label=f"{label} load (n={len(sorted_osnr)})")

    ax.set_xlabel("End-to-end OSNR (dB)")
    ax.set_ylabel("CDF")
    ax.set_title("a  End-to-end lightpath OSNR distribution")
    ax.legend(loc="upper left", frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig8_path_osnr_cdf.svg"))
    plt.close(fig)


def fig9_osnr_vs_hops(data_dict: Dict[str, SimulationDataCollector], outdir: str = "."):
    """Figure 9: OSNR degradation with hop count."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    max_hops_all = 0
    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_hops:
            continue

        hop_groups = defaultdict(list)
        for hops, osnr in zip(d.path_hops, d.path_osnr_dB):
            hop_groups[hops].append(osnr)

        hop_counts = sorted(hop_groups.keys())
        means = [np.mean(hop_groups[h]) for h in hop_counts]
        stds = [np.std(hop_groups[h]) for h in hop_counts]

        ax.errorbar(hop_counts, means, yerr=stds,
                    color=colour, marker="o", ms=4,
                    lw=1.2, capsize=2, capthick=0.6,
                    label=f"{label} load", alpha=0.85)
        max_hops_all = max(max_hops_all, max(hop_counts))

    if max_hops_all > 0:
        h_theory = np.arange(1, max_hops_all + 2)
        osnr_theory = 25.0 - 10.0 * np.log10(h_theory)
        ax.plot(h_theory, osnr_theory, "--", color=C_REF, lw=0.8,
                label=r"Theoretical ($-10\log_{10}H$)")

    ax.set_xlabel("Hop count")
    ax.set_ylabel("End-to-end OSNR (dB)")
    ax.set_title("b  OSNR vs hop count")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    ax.set_xlim(0.5, max_hops_all + 1.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig9_osnr_vs_hops.svg"))
    plt.close(fig)


def fig10_ber_distribution(
    data_dict: Dict[str, SimulationDataCollector],
    use_fec: bool = True,
    outdir: str = ".",
):
    """Figure 10: BER CDF with FEC threshold lines."""
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    fec_threshold = 2.0e-3 if use_fec else 1.0e-12

    for label, colour in [("Low", C_LOW), ("Medium", C_MED), ("High", C_HIGH)]:
        d = data_dict.get(label)
        if d is None or not d.path_ber:
            continue
        log_ber = np.log10(np.array(d.path_ber))
        sorted_ber = np.sort(log_ber)
        cdf = np.linspace(0, 1, len(sorted_ber))
        ax.plot(sorted_ber, cdf, color=colour, lw=1.5,
                label=f"{label} load (n={len(sorted_ber)})")

    ax.axvline(x=np.log10(fec_threshold), color=C_SAA, ls="--", lw=0.8,
               label=f"FEC threshold ({fec_threshold:.0e})")
    if use_fec:
        ax.axvline(x=np.log10(1e-12), color=C_REF, ls=":", lw=0.6,
                   label="No-FEC threshold (1e-12)")

    ax.set_xlabel("log10(BER)")
    ax.set_ylabel("CDF")
    ax.set_title("c  End-to-end BER distribution")
    ax.legend(loc="upper left", frameon=False, fontsize=6)
    ax.set_ylim(0, 1.02)

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
    fig1_link_osnr_distribution(primary, outdir)
    fig2_distance_vs_osnr(primary, outdir)
    fig3_doppler_distribution(primary, outdir)
    fig4_doppler_penalty_vs_shift(primary, outdir)
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

    print(f"All figures saved to {outdir}/")
