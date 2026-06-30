"""
Multi-load simulation runner for paper figure generation.

Runs QoT simulations at low, medium, and high traffic loads,
collects all data, and generates paper-quality figures.
"""

import sys
import os
import math
import copy
import numpy as np
from collections import Counter

try:
    from qot_simulation.simulation import (
        SimulationConfig, QoTSimulation, create_demo_requests, LightpathRequest
    )
    from plotting import SimulationDataCollector, generate_all_figures
except ImportError:
    from simulation import (
        SimulationConfig, QoTSimulation, create_demo_requests, LightpathRequest
    )
    from plotting import SimulationDataCollector, generate_all_figures


def build_default_config() -> SimulationConfig:
    """Build a representative LEO constellation configuration."""
    return SimulationConfig(
        N_planes=12,
        sats_per_plane=11,
        F=1,
        altitude_km=550.0,
        inclination_deg=53.0,
        num_channels=24,
        channel_spacing_GHz=25.0,
        data_rate_Gbps=10.0,
        tx_power_dBm=30.0,
        rx_aperture_diameter_mm=80.0,
        tx_divergence_angle_urad=10.0,
        optical_efficiency=0.6,
        pointing_loss_dB=1.0,
        oxc_insertion_loss_dB=3.0,
        oxc_isolation_dB=30.0,
        optical_filter_BW_GHz=12.5,
        demux_isolation_dB=25.0,
        rx_thermal_noise_dBm=-40.0,
        osnr_ref_BW_GHz=12.5,
        rx_electrical_BW_GHz=7.5,
        nominal_gain_dB=20.0,
        nominal_nf_dB=5.0,
        gain_degradation_slope=0.10,
        nf_degradation_slope=0.20,
        background_dose_rate=0.1,
        saa_enhancement=10.0,
        saa_center_lat_deg=-25.0,
        saa_center_lon_deg=-45.0,
        saa_lat_window_deg=15.0,
        saa_lon_window_deg=30.0,
        max_hops=12,
        max_link_distance_km=5400.0,
        max_cumulative_dose_krad=50.0,
        use_fec=True,
        ber_threshold_no_fec=1.0e-12,
        ber_threshold_with_fec=2.0e-3,
        k_shortest_paths=3,
        sim_time_seconds=0.0,
        mission_age_years=3.0,
        path_duration_yr=0.1,
        sun_avoidance_angle_deg=3.0,
        sun_coupling_scale_deg=5.0,
        sun_coupling_order=4.0,
        sun_spectral_radiance=1.5e6,
        sky_spectral_radiance=3.0e-4,
        fov_solid_angle_sr=1.0e-10,
    )


def run_traffic_level(
    config: SimulationConfig,
    num_requests: int,
    label: str,
    seed: int = 42,
    observe_duration_s: float = 3600.0,
    observe_step_s: float = 300.0,
):
    """Run simulation at a specific traffic level and collect data."""
    print(f"\n{'='*70}")
    print(f"Running {label} traffic: {num_requests} requests on "
          f"{config.N_planes * config.sats_per_plane} satellites")
    print(f"{'='*70}")

    sim = QoTSimulation(config)

    requests = create_demo_requests(
        sim.constellation.total_sats,
        num_requests=num_requests,
        seed=seed,
    )

    results = sim.run_requests(requests, verbose=(num_requests <= 50))

    collector = SimulationDataCollector(label)
    collector.collect_from_simulation(sim)
    collector.collect_results(results)
    if observe_duration_s > 0 and observe_step_s > 0:
        qot_samples = sim.evaluate_established_lightpaths_over_time(
            results,
            duration_s=observe_duration_s,
            step_s=observe_step_s,
        )
        collector.collect_time_qot_samples(qot_samples)

    accepted = collector.num_accepted
    rejected = collector.num_rejected
    print(f"\n{label} load summary:")
    print(f"  Total: {num_requests} | Accepted: {accepted} | "
          f"Rejected: {rejected} | Rate: {collector.acceptance_rate*100:.1f}%")

    if rejected > 0:
        reasons = collector.get_rejection_reasons()
        if "Accepted" in reasons:
            del reasons["Accepted"]
        for reason, count in reasons.most_common():
            print(f"    {reason}: {count}")

    if accepted > 0:
        print(f"  OSNR: {min(collector.path_osnr_dB):.1f}-{max(collector.path_osnr_dB):.1f} dB "
              f"(mean {np.mean(collector.path_osnr_dB):.1f})")
        print(f"  BER: {min(collector.path_ber):.1e}-{max(collector.path_ber):.1e}")
        print(f"  Hops: {min(collector.path_hops)}-{max(collector.path_hops)} "
              f"(mean {np.mean(collector.path_hops):.1f})")

    return collector, sim


def main():
    """Run all traffic levels and generate figures."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Multi-load QoT simulation for paper figures"
    )
    parser.add_argument(
        "--outdir", type=str, default="./figures",
        help="Output directory for figures (default: ./figures)"
    )
    parser.add_argument(
        "--low", type=int, default=10,
        help="Number of requests for low load (default: 10)"
    )
    parser.add_argument(
        "--medium", type=int, default=50,
        help="Number of requests for medium load (default: 50)"
    )
    parser.add_argument(
        "--high", type=int, default=200,
        help="Number of requests for high load (default: 200)"
    )
    parser.add_argument(
        "--starlink", action="store_true",
        help="Use Starlink-scale constellation (72x22=1584 sats)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--observe-duration", type=float, default=3600.0,
        help="Time-window QoT observation duration in seconds (default: 3600)"
    )
    parser.add_argument(
        "--observe-step", type=float, default=300.0,
        help="Time-window QoT observation step in seconds (default: 300)"
    )
    args = parser.parse_args()

    if args.starlink:
        base_config = SimulationConfig(
            N_planes=72, sats_per_plane=22, F=11,
            altitude_km=550.0, inclination_deg=53.0,
            num_channels=80,
            mission_age_years=3.0,
            background_dose_rate=0.1,
        )
    else:
        base_config = build_default_config()

    print("=" * 70)
    print("MULTI-LOAD QoT SIMULATION FOR PAPER FIGURES")
    print("=" * 70)
    print(f"Constellation: {base_config.N_planes} planes x "
          f"{base_config.sats_per_plane} sats/plane = "
          f"{base_config.N_planes * base_config.sats_per_plane} satellites")
    print(f"Altitude: {base_config.altitude_km} km | "
          f"Inclination: {base_config.inclination_deg} deg")
    print(f"WDM: {base_config.num_channels} channels x "
          f"{base_config.channel_spacing_GHz} GHz")
    print(f"FEC: {'HD-FEC (BER<2e-3)' if base_config.use_fec else 'No FEC (BER<1e-12)'}")
    print(f"Mission age: {base_config.mission_age_years} yr | "
          f"Path duration: {base_config.path_duration_yr} yr")
    print(f"QoT observation window: {args.observe_duration:.0f} s | "
          f"step: {args.observe_step:.0f} s")

    data_dict = {}
    sim_dict = {}
    config_dict = {}

    for level, n_req, seed_offset, desc in [
        ("Low", args.low, 0, "light load - minimal contention"),
        ("Medium", args.medium, 100, "moderate load - emerging contention"),
        ("High", args.high, 200, "heavy load - significant blocking"),
    ]:
        collector, sim = run_traffic_level(
            copy.deepcopy(base_config),
            n_req,
            level,
            seed=args.seed + seed_offset,
            observe_duration_s=args.observe_duration,
            observe_step_s=args.observe_step,
        )
        data_dict[level] = collector
        sim_dict[level] = sim
        config_dict[level] = base_config

    print(f"\n{'='*70}")
    print("Generating paper-quality figures...")
    print(f"{'='*70}")

    generate_all_figures(
        data_dict=data_dict,
        sim_configs=config_dict,
        sim_objects=sim_dict,
        outdir=args.outdir,
    )

    print(f"\nGenerated figures in {args.outdir}/:")
    for f in sorted(os.listdir(args.outdir)):
        if f.endswith(".svg"):
            size_kb = os.path.getsize(os.path.join(args.outdir, f)) / 1024
            print(f"  {f} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
