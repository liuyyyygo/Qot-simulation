"""
QoT-Guaranteed Lightpath Routing Simulation for LEO Satellite Optical Networks.

Takes satellite constellation parameters, physical layer configuration,
and traffic requests as input, and outputs OSNR and BER for each
established lightpath under extended QoT constraints.

Usage:
  python simulation.py                          # Run with defaults
  python simulation.py --config config.yaml     # Run with config file
  python simulation.py --demo                   # Run demo with sample requests
"""

import sys
import math
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np

try:
    from .constellation import (
        WalkerDeltaConstellation,
        SatelliteState,
        LISL,
    )
    from .impairments import (
        FreeSpaceLoss,
        TelescopeGain,
        DopplerShift,
        CelestialBackground,
        WDMCrosstalk,
        SAARadiation,
        ImpairmentResult,
    )
    from .osnr_ber import (
        OSNRCalculator,
        BERCalculator,
        LinkOSNRResult,
        PathOSNRResult,
        BERResult,
        format_ber_scientific,
    )
    from .rwa import (
        RWASolver,
        LightpathRequest,
        LightpathResult,
        RejectReason,
    )
except ImportError:
    from constellation import (
        WalkerDeltaConstellation,
        SatelliteState,
        LISL,
    )
    from impairments import (
        FreeSpaceLoss,
        TelescopeGain,
        DopplerShift,
        CelestialBackground,
        WDMCrosstalk,
        SAARadiation,
        ImpairmentResult,
    )
    from osnr_ber import (
        OSNRCalculator,
        BERCalculator,
        LinkOSNRResult,
        PathOSNRResult,
        BERResult,
        format_ber_scientific,
    )
    from rwa import (
        RWASolver,
        LightpathRequest,
        LightpathResult,
        RejectReason,
    )


@dataclass
class SimulationConfig:
    """Simulation configuration loaded from YAML or defaults."""
    # Constellation
    N_planes: int = 6
    sats_per_plane: int = 6
    F: int = 1
    altitude_km: float = 550.0
    inclination_deg: float = 53.0

    # Physical layer
    wavelength_start_nm: float = 1550.12
    wavelength_end_nm: float = 1555.75
    channel_spacing_GHz: float = 25.0
    num_channels: int = 24
    carrier_freq_THz: float = 193.414
    data_rate_Gbps: float = 10.0
    tx_power_dBm: float = 30.0
    rx_aperture_diameter_mm: float = 80.0
    tx_divergence_angle_urad: float = 10.0
    optical_efficiency: float = 0.6
    pointing_loss_dB: float = 1.0
    oxc_insertion_loss_dB: float = 3.0
    oxc_isolation_dB: float = 30.0
    optical_filter_BW_GHz: float = 12.5
    demux_isolation_dB: float = 25.0
    rx_thermal_noise_dBm: float = -40.0
    osnr_ref_BW_GHz: float = 12.5
    rx_electrical_BW_GHz: float = 7.5

    # EDFA
    nominal_gain_dB: float = 20.0
    nominal_nf_dB: float = 5.0
    gain_degradation_slope: float = 0.10
    nf_degradation_slope: float = 0.20

    # SAA
    saa_center_lat_deg: float = -25.0
    saa_center_lon_deg: float = -45.0
    saa_lat_window_deg: float = 15.0
    saa_lon_window_deg: float = 30.0
    background_dose_rate: float = 0.1
    saa_enhancement: float = 10.0

    # Celestial
    sun_avoidance_angle_deg: float = 3.0
    sun_coupling_scale_deg: float = 5.0
    sun_coupling_order: float = 4.0
    sun_spectral_radiance: float = 1.5e6
    sky_spectral_radiance: float = 3.0e-4
    fov_solid_angle_sr: float = 1.0e-10

    # RWA
    max_hops: int = 12
    max_link_distance_km: float = 5400.0
    max_cumulative_dose_krad: float = 50.0
    enforce_qot_constraints: bool = False
    use_fec: bool = True
    ber_threshold_no_fec: float = 1.0e-12
    ber_threshold_with_fec: float = 2.0e-3
    k_shortest_paths: int = 3

    # Simulation
    sim_time_seconds: float = 0.0
    earth_radius_km: float = 6371.0
    mission_age_years: float = 3.0
    path_duration_yr: float = 0.1
    qot_observation_duration_s: float = 0.0
    qot_observation_step_s: float = 60.0

    # Routing strategy
    routing_strategy: str = "baseline"

    # QoT-GLR lightweight risk model
    qot_glr_risk_duration_s: float = 3600.0
    qot_glr_risk_step_s: float = 900.0
    qot_glr_pruning_threshold: float = 0.85
    qot_glr_sun_weight: float = 0.35
    qot_glr_distance_weight: float = 0.25
    qot_glr_occupation_weight: float = 0.20
    qot_glr_radiation_weight: float = 0.12
    qot_glr_doppler_weight: float = 0.08
    qot_glr_accum_weight: float = 0.45
    qot_glr_bottleneck_weight: float = 0.25
    qot_glr_temporal_weight: float = 0.20
    qot_glr_wavelength_weight: float = 0.10
    qot_glr_adjacent_xt_weight: float = 0.25


@dataclass
class LightpathQoTSample:
    """QoT sample for an already established lightpath at one time instant."""
    time_seconds: float
    request_id: int
    path_links: List[int]
    wavelength_channel: int
    osnr_dB: float
    ber: float
    q_factor: float
    min_sun_angle_deg: float
    max_link_distance_km: float


class QoTSimulation:
    """
    Main QoT simulation engine.

    Integrates constellation modeling, physical impairment calculation,
    OSNR/BER computation, and RWA with QoT constraints.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config

        # Initialize constellation
        self.constellation = WalkerDeltaConstellation(
            N_planes=config.N_planes,
            sats_per_plane=config.sats_per_plane,
            F=config.F,
            altitude_km=config.altitude_km,
            inclination_deg=config.inclination_deg,
        )

        self.constellation.propagate(config.sim_time_seconds)

        # Initialize impairment calculators
        self._init_impairment_calculators()

        # Initialize OSNR/BER calculators
        self.osnr_calc = OSNRCalculator(
            osnr_ref_BW_GHz=config.osnr_ref_BW_GHz,
            rx_electrical_BW_GHz=config.rx_electrical_BW_GHz,
            thermal_noise_dBm=config.rx_thermal_noise_dBm,
        )

        self.ber_calc = BERCalculator(
            modulation="OOK",
            osnr_ref_BW_GHz=config.osnr_ref_BW_GHz,
            rx_electrical_BW_GHz=config.rx_electrical_BW_GHz,
            ber_threshold_no_fec=config.ber_threshold_no_fec,
            ber_threshold_with_fec=config.ber_threshold_with_fec,
            use_fec=config.use_fec,
        )

        # Build link data for RWA
        self._init_rwa_data()

        # Initialize RWA solver (pass BER calculator to avoid duplication)
        self.rwa = RWASolver(
            num_satellites=self.constellation.total_sats,
            adjacency=self.constellation.get_adjacency_list(),
            link_distances_km=self.rwa_link_distances,
            topology_links=self.constellation.topology,
            num_wavelengths=config.num_channels,
            max_hops=config.max_hops,
            max_link_distance_km=config.max_link_distance_km,
            sun_avoidance_angle_deg=config.sun_avoidance_angle_deg,
            max_cumulative_dose_krad=config.max_cumulative_dose_krad,
            ber_threshold=(
                config.ber_threshold_with_fec
                if config.use_fec
                else config.ber_threshold_no_fec
            ),
            enforce_qot_constraints=config.enforce_qot_constraints,
            ber_calculator=self.ber_calc,
        )

        # Pre-computed impairment results cache
        self._impairment_cache: Dict[int, ImpairmentResult] = {}

    def _init_impairment_calculators(self):
        """Initialize all impairment model calculators."""
        self.fsl = FreeSpaceLoss(wavelength_nm=1550.0)
        self.telescope = TelescopeGain(wavelength_nm=1550.0)
        self.doppler = DopplerShift(
            carrier_freq_THz=self.config.carrier_freq_THz
        )
        carrier_wavelength_m = 299792458.0 / (self.config.carrier_freq_THz * 1e12)
        filter_spectral_BW_um = (
            carrier_wavelength_m ** 2
            * self.config.optical_filter_BW_GHz
            * 1e9
            / 299792458.0
            * 1e6
        )
        self.celestial = CelestialBackground(
            aperture_diameter_m=self.config.rx_aperture_diameter_mm / 1000.0,
            fov_solid_angle_sr=self.config.fov_solid_angle_sr,
            filter_spectral_BW_um=filter_spectral_BW_um,
            sun_spectral_radiance=self.config.sun_spectral_radiance,
            sky_spectral_radiance=self.config.sky_spectral_radiance,
            sun_avoidance_angle_deg=self.config.sun_avoidance_angle_deg,
            sun_coupling_scale_deg=self.config.sun_coupling_scale_deg,
            sun_coupling_order=self.config.sun_coupling_order,
        )
        self.crosstalk = WDMCrosstalk(
            channel_spacing_GHz=self.config.channel_spacing_GHz,
            filter_BW_GHz=self.config.optical_filter_BW_GHz,
            demux_isolation_dB=self.config.demux_isolation_dB,
            oxc_isolation_dB=self.config.oxc_isolation_dB,
            num_channels=self.config.num_channels,
        )
        self.saa = SAARadiation(
            saa_center_lat_deg=self.config.saa_center_lat_deg,
            saa_center_lon_deg=self.config.saa_center_lon_deg,
            lat_window_deg=self.config.saa_lat_window_deg,
            lon_window_deg=self.config.saa_lon_window_deg,
            background_dose_rate_krad_yr=self.config.background_dose_rate,
            saa_enhancement_factor=self.config.saa_enhancement,
            nominal_gain_dB=self.config.nominal_gain_dB,
            nominal_nf_dB=self.config.nominal_nf_dB,
            gain_degradation_slope=self.config.gain_degradation_slope,
            nf_degradation_slope=self.config.nf_degradation_slope,
            osnr_ref_BW_GHz=self.config.osnr_ref_BW_GHz,
        )

        self.G_tx_dB = self.telescope.tx_gain_dB(
            self.config.tx_divergence_angle_urad * 1e-6
        )
        self.G_rx_dB = self.telescope.rx_gain_dB(
            self.config.rx_aperture_diameter_mm / 1000.0
        )

    def _init_rwa_data(self):
        """Build RWA data structures from constellation and topology."""
        self.rwa_link_distances = {}
        self.rwa_link_to_distance = {}
        self.rwa_link_to_sun_angle = {}
        self.rwa_sat_to_dose_rate = {}

        for lisl in self.constellation.topology:
            edge = (min(lisl.sat_i, lisl.sat_j), max(lisl.sat_i, lisl.sat_j))
            self.rwa_link_distances[edge] = 0.0
            self.rwa_link_to_distance[lisl.link_id] = 0.0
            self.rwa_link_to_sun_angle[lisl.link_id] = 180.0

        for sat_id in range(self.constellation.total_sats):
            sat = self.constellation.get_satellite(sat_id)
            dose_rate = self.saa.compute_dose_rate_krad_yr(
                sat.lat_deg,
                sat.lon_deg,
                sat.altitude_km,
            )
            self.rwa_sat_to_dose_rate[sat_id] = dose_rate
            self.constellation.satellites[sat_id].cumulative_dose_krad = (
                dose_rate * self.config.mission_age_years
            )

    def _build_link_osnr_lookup(self) -> Dict[Tuple[int, int], float]:
        """
        Build wavelength-aware OSNR lookup with current occupancy.

        Computes per-link, per-wavelength OSNR accounting for:
        - EDFA gain degradation from cumulative radiation dose
        - Inter-wavelength crosstalk from occupied channels
        - Intra-wavelength crosstalk from OXC port leakage
        """
        link_osnr_db: Dict[Tuple[int, int], float] = {}

        for lisl in self.constellation.topology:
            imp = self._impairment_cache.get(lisl.link_id)
            if imp is None:
                continue

            effective_gain_dB = self.config.nominal_gain_dB - imp.edfa_gain_degradation_dB
            effective_gain_lin = 10.0 ** (effective_gain_dB / 10.0)
            signal_after_edfa_W = imp.received_signal_power_W * effective_gain_lin
            occupied = list(self.rwa.wavelength_occupancy.get(lisl.link_id, set()))

            for ch in range(self.config.num_channels):
                inter_xt_var = self.crosstalk.compute_inter_wavelength_noise_var(
                    target_channel=ch,
                    occupied_channels=occupied,
                    received_power_per_channel_W=imp.received_signal_power_W,
                    doppler_shifts={occ_ch: imp.doppler_shift_GHz for occ_ch in occupied},
                )
                num_intra_interf = self._count_same_wavelength_oxc_interferers(
                    lisl, ch
                )
                intra_xt_var = self.crosstalk.compute_intra_wavelength_noise_var(
                    num_interfering_ports=num_intra_interf,
                    received_power_W=imp.received_signal_power_W,
                )

                _, osnr_dB, _ = self.osnr_calc.compute_single_link_osnr(
                    signal_power_W=signal_after_edfa_W,
                    ase_noise_W=imp.ase_noise_power_W,
                    celestial_noise_W=imp.celestial_noise_power_W,
                    inter_xt_noise_var=inter_xt_var,
                    intra_xt_noise_var=intra_xt_var,
                )
                link_osnr_db[(lisl.link_id, ch)] = osnr_dB

        return link_osnr_db

    def _count_same_wavelength_oxc_interferers(self, target_lisl: LISL, channel: int) -> int:
        """
        Count same-wavelength OXC leakage contributors at either endpoint.

        Yang et al.'s in-band crosstalk source is other input ports carrying
        the same nominal wavelength into the wavelength-routing node.  The
        network-level equivalent used here counts occupied incident LISLs at
        the two endpoint satellites, excluding the target LISL itself.
        """
        endpoint_sats = {target_lisl.sat_i, target_lisl.sat_j}
        interfering_links = set()
        for lisl in self.constellation.topology:
            if lisl.link_id == target_lisl.link_id:
                continue
            if not ({lisl.sat_i, lisl.sat_j} & endpoint_sats):
                continue
            if channel in self.rwa.wavelength_occupancy.get(lisl.link_id, set()):
                interfering_links.add(lisl.link_id)
        return len(interfering_links)

    def compute_impairments_for_link(
        self, lisl: LISL
    ) -> ImpairmentResult:
        """
        Compute all physical impairments for a single LISL.
        """
        if lisl.link_id in self._impairment_cache:
            return self._impairment_cache[lisl.link_id]

        sat_i = self.constellation.get_satellite(lisl.sat_i)
        sat_j = self.constellation.get_satellite(lisl.sat_j)

        distance_km = self.constellation.compute_link_distance_km(lisl.sat_i, lisl.sat_j)
        distance_m = distance_km * 1000.0
        unit_vec = self.constellation.compute_link_unit_vector(lisl.sat_i, lisl.sat_j)

        # 1. Free space loss
        fsl_dB = self.fsl.compute_loss_dB(distance_m)

        # 2. Doppler shift and filter penalty
        v_radial = self.constellation.compute_relative_radial_velocity_ms(
            lisl.sat_i, lisl.sat_j
        )
        doppler_GHz = self.doppler.compute_shift_GHz(v_radial)
        doppler_penalty_dB = self.doppler.compute_filter_penalty_dB(
            doppler_GHz,
            filter_BW_GHz=self.config.optical_filter_BW_GHz,
            has_frequency_tracking=False,
        )

        # 3. Celestial background
        sun_vector = self.celestial.compute_sun_vector_ecef(
            sun_dec_deg=0.0,
            sun_ra_deg=0.0,
            time_hours=(12.0 + self.config.sim_time_seconds / 3600.0) % 24.0,
        )
        sun_angle = self.celestial.compute_bidirectional_link_sun_angle_deg(
            unit_vec,
            sun_vector,
        )
        celestial_noise_W, sun_blocked = self.celestial.compute_noise_power_W(
            sun_angle,
            optical_efficiency=self.config.optical_efficiency,
        )

        # 4. Received signal power
        rx_power_dBm = (
            self.config.tx_power_dBm
            + self.G_tx_dB
            + self.G_rx_dB
            - fsl_dB
            - self.config.pointing_loss_dB
            - self.config.oxc_insertion_loss_dB
            - doppler_penalty_dB
        )
        rx_power_W = 10.0 ** ((rx_power_dBm - 30.0) / 10.0) * self.config.optical_efficiency

        # 5. SAA radiation and EDFA degradation
        saa_risk_i = self.saa.compute_saa_risk_factor(sat_i.lat_deg, sat_i.lon_deg)
        saa_risk_j = self.saa.compute_saa_risk_factor(sat_j.lat_deg, sat_j.lon_deg)
        avg_risk = (saa_risk_i + saa_risk_j) / 2.0

        avg_dose = (sat_i.cumulative_dose_krad + sat_j.cumulative_dose_krad) / 2.0

        gain_degradation = self.saa.compute_gain_degradation_dB(avg_dose)
        nf_increase = self.saa.compute_nf_increase_dB(avg_dose)

        effective_gain_dB = self.config.nominal_gain_dB - gain_degradation
        effective_nf_dB = self.config.nominal_nf_dB + nf_increase
        effective_gain_lin = 10.0 ** (effective_gain_dB / 10.0)
        effective_nf_lin = 10.0 ** (effective_nf_dB / 10.0)

        ase_noise_W = self.saa.compute_ase_noise_power_W(
            effective_gain_lin, effective_nf_lin
        )

        # 6. WDM crosstalk (placeholder — actual values in _build_link_osnr_lookup)
        inter_xt_var = 0.0
        intra_xt_var = 0.0

        # 7. OSNR computation (after EDFA amplification)
        rx_power_after_edfa_W = rx_power_W * effective_gain_lin

        osnr_lin, osnr_dB, total_noise = self.osnr_calc.compute_single_link_osnr(
            signal_power_W=rx_power_after_edfa_W,
            ase_noise_W=ase_noise_W,
            celestial_noise_W=celestial_noise_W,
            inter_xt_noise_var=inter_xt_var,
            intra_xt_noise_var=intra_xt_var,
        )

        result = ImpairmentResult(
            link_distance_km=distance_km,
            free_space_loss_dB=fsl_dB,
            doppler_shift_GHz=doppler_GHz,
            doppler_filter_penalty_dB=doppler_penalty_dB,
            celestial_noise_power_W=celestial_noise_W,
            sun_angle_deg=sun_angle,
            is_sun_blocked=sun_blocked,
            inter_xt_noise_var=inter_xt_var,
            intra_xt_noise_var=intra_xt_var,
            saa_risk_factor=avg_risk,
            edfa_gain_degradation_dB=gain_degradation,
            edfa_nf_increase_dB=nf_increase,
            ase_noise_power_W=ase_noise_W,
            total_noise_power_W=total_noise,
            received_signal_power_W=rx_power_W,
            osnr_linear=osnr_lin,
            osnr_dB=osnr_dB,
        )

        self._impairment_cache[lisl.link_id] = result

        edge = (min(lisl.sat_i, lisl.sat_j), max(lisl.sat_i, lisl.sat_j))
        self.rwa_link_distances[edge] = distance_km
        self.rwa_link_to_distance[lisl.link_id] = distance_km
        self.rwa_link_to_sun_angle[lisl.link_id] = sun_angle

        return result

    def compute_all_impairments(self):
        """Pre-compute impairments for all LISLs in the topology."""
        for lisl in self.constellation.topology:
            self.compute_impairments_for_link(lisl)

    def _qot_glr_time_points(self) -> List[float]:
        """Return coarse trajectory sampling instants for QoT-GLR risk estimation."""
        duration_s = max(0.0, self.config.qot_glr_risk_duration_s)
        step_s = max(1.0, self.config.qot_glr_risk_step_s)
        start_s = self.config.sim_time_seconds
        points = []
        t = 0.0
        while t <= duration_s + 1e-9:
            points.append(start_s + t)
            t += step_s
        if points[-1] < start_s + duration_s:
            points.append(start_s + duration_s)
        return sorted(set(round(p, 9) for p in points))

    def _normalise_weighted_sum(self, terms: Dict[str, float]) -> float:
        weights = {
            "sun": self.config.qot_glr_sun_weight,
            "distance": self.config.qot_glr_distance_weight,
            "occupation": self.config.qot_glr_occupation_weight,
            "radiation": self.config.qot_glr_radiation_weight,
            "doppler": self.config.qot_glr_doppler_weight,
        }
        total_weight = sum(max(0.0, w) for w in weights.values())
        if total_weight <= 0:
            return 0.0
        return sum(max(0.0, weights[k]) * terms[k] for k in weights) / total_weight

    def _compute_qot_glr_link_risks(self):
        """
        Estimate model-derived link risks over the expected holding interval.

        The risk terms follow the manuscript's QoT-GLR design: distance,
        near-solar background, Doppler filtering, radiation-induced EDFA
        degradation, and wavelength occupation are normalised to [0, 1].
        """
        original_time = self.config.sim_time_seconds
        risk_series: Dict[int, List[float]] = {
            lisl.link_id: [] for lisl in self.constellation.topology
        }
        indicators: Dict[int, Dict[str, float]] = {}

        sun_core = self.config.sun_avoidance_angle_deg
        sun_tail = self.config.sun_avoidance_angle_deg + 3.0 * self.config.sun_coupling_scale_deg
        doppler_th = max(self.config.optical_filter_BW_GHz / 2.0, 1e-9)
        gain_deg_th = max(
            self.config.gain_degradation_slope * self.config.max_cumulative_dose_krad,
            1e-9,
        )

        for time_s in self._qot_glr_time_points():
            self._refresh_impairments_at_time(time_s)
            for lisl in self.constellation.topology:
                imp = self._impairment_cache[lisl.link_id]
                occupied_count = len(
                    self.rwa.wavelength_occupancy.get(lisl.link_id, set())
                )
                r_dist = min(
                    max(imp.link_distance_km / max(self.config.max_link_distance_km, 1e-9), 0.0),
                    1.0,
                )
                if imp.sun_angle_deg <= sun_core:
                    r_sun = 1.0
                elif imp.sun_angle_deg >= sun_tail:
                    r_sun = 0.0
                else:
                    x = (sun_tail - imp.sun_angle_deg) / max(sun_tail - sun_core, 1e-9)
                    r_sun = min(max(x, 0.0), 1.0)
                r_dop = min(abs(imp.doppler_shift_GHz) / doppler_th, 1.0)
                r_rad = min(max(imp.edfa_gain_degradation_dB, 0.0) / gain_deg_th, 1.0)
                r_occ = min(occupied_count / max(self.config.num_channels, 1), 1.0)
                terms = {
                    "sun": r_sun,
                    "distance": r_dist,
                    "occupation": r_occ,
                    "radiation": r_rad,
                    "doppler": r_dop,
                }
                link_risk = self._normalise_weighted_sum(terms)
                risk_series[lisl.link_id].append(link_risk)

                item = indicators.setdefault(
                    lisl.link_id,
                    {
                        "dmax": 0.0,
                        "theta_min": 180.0,
                        "doppler_max": 0.0,
                        "edfa_gain_deg_max": 0.0,
                        "occupation": occupied_count,
                    },
                )
                item["dmax"] = max(item["dmax"], imp.link_distance_km)
                item["theta_min"] = min(item["theta_min"], imp.sun_angle_deg)
                item["doppler_max"] = max(item["doppler_max"], abs(imp.doppler_shift_GHz))
                item["edfa_gain_deg_max"] = max(
                    item["edfa_gain_deg_max"], imp.edfa_gain_degradation_dB
                )
                item["occupation"] = occupied_count

        self._refresh_impairments_at_time(original_time)

        link_risks = {
            link_id: (max(values) if values else 0.0)
            for link_id, values in risk_series.items()
        }
        return link_risks, risk_series, indicators

    def _is_wavelength_available(self, path_links: List[int], channel: int) -> bool:
        return all(
            channel not in self.rwa.wavelength_occupancy.get(link_id, set())
            for link_id in path_links
        )

    def _compute_path_result_from_osnr(
        self,
        request: LightpathRequest,
        path_sats: List[int],
        path_links: List[int],
        wavelength: int,
        link_osnr_db: Dict[Tuple[int, int], float],
    ) -> LightpathResult:
        per_link_osnr = []
        per_link_dist = []
        inv_osnr_sum = 0.0
        for link_id in path_links:
            osnr_dB = link_osnr_db.get((link_id, wavelength), 15.0)
            osnr_lin = 10.0 ** (osnr_dB / 10.0)
            per_link_osnr.append(osnr_dB)
            per_link_dist.append(self.rwa_link_to_distance.get(link_id, 0.0))
            inv_osnr_sum += 1.0 / max(osnr_lin, 1e-12)

        total_osnr_lin = 1.0 / inv_osnr_sum if inv_osnr_sum > 0 else 0.0
        total_osnr_dB = 10.0 * math.log10(max(total_osnr_lin, 1e-30))
        ber_result = self.ber_calc.compute_ber_from_osnr(total_osnr_dB)

        return LightpathResult(
            request_id=request.request_id,
            src_sat=request.src_sat,
            dst_sat=request.dst_sat,
            accepted=True,
            path_satellites=path_sats,
            path_links=path_links,
            wavelength_channel=wavelength,
            total_hops=len(path_links),
            osnr_dB=total_osnr_dB,
            ber=ber_result.ber,
            q_factor=ber_result.q_factor,
            per_link_osnr_dB=per_link_osnr,
            per_link_distance_km=per_link_dist,
        )

    def _qot_glr_wavelength_term(self, path_links: List[int], channel: int) -> float:
        link_by_id = {lisl.link_id: lisl for lisl in self.constellation.topology}
        n_port = max(1, getattr(self.constellation, "max_lisl_per_sat", 4))
        if not path_links:
            return 0.0

        total = 0.0
        for link_id in path_links:
            lisl = link_by_id[link_id]
            same = self._count_same_wavelength_oxc_interferers(lisl, channel)
            occupied = self.rwa.wavelength_occupancy.get(link_id, set())
            adjacent = sum(
                1 for ch in occupied if abs(ch - channel) == 1
            )
            total += (
                same / n_port
                + self.config.qot_glr_adjacent_xt_weight
                * adjacent / max(self.config.num_channels, 1)
            )
        return total / len(path_links)

    def _assign_qot_glr_lightpath(
        self,
        request: LightpathRequest,
        link_osnr_db: Dict[Tuple[int, int], float],
    ) -> LightpathResult:
        """Assign a lightpath using the manuscript-inspired QoT-GLR score."""
        result = LightpathResult(
            request_id=request.request_id,
            src_sat=request.src_sat,
            dst_sat=request.dst_sat,
            accepted=False,
        )

        link_risks, risk_series, indicators = self._compute_qot_glr_link_risks()
        edge_weights = {}
        for lisl in self.constellation.topology:
            edge = (min(lisl.sat_i, lisl.sat_j), max(lisl.sat_i, lisl.sat_j))
            ind = indicators.get(lisl.link_id, {})
            hard_pruned = (
                ind.get("dmax", 0.0) > self.config.max_link_distance_km
                or ind.get("theta_min", 180.0) < self.config.sun_avoidance_angle_deg
                or link_risks.get(lisl.link_id, 0.0) > self.config.qot_glr_pruning_threshold
            )
            edge_weights[edge] = (
                float("inf")
                if hard_pruned
                else 1.0 + link_risks.get(lisl.link_id, 0.0)
            )

        paths = self.rwa.k_shortest_paths(
            request.src_sat,
            request.dst_sat,
            self.config.k_shortest_paths,
            link_weights=edge_weights,
        )
        paths = [
            (path_sats, path_links)
            for path_sats, path_links in paths
            if path_links and len(path_links) <= self.config.max_hops
            and all(edge_weights.get(
                (min(path_sats[i], path_sats[i + 1]), max(path_sats[i], path_sats[i + 1])),
                1.0,
            ) < float("inf") for i in range(len(path_sats) - 1))
        ]
        if not paths:
            result.reject_reason = RejectReason.NO_PATH
            return result

        best = None
        best_score = float("inf")
        for path_sats, path_links in paths:
            path_len = max(len(path_links), 1)
            accum = sum(link_risks.get(link_id, 0.0) for link_id in path_links) / path_len
            bottleneck = max(link_risks.get(link_id, 0.0) for link_id in path_links)
            temporal_values = []
            series_len = max((len(risk_series.get(link_id, [])) for link_id in path_links), default=0)
            for idx in range(series_len):
                temporal_values.append(
                    sum(risk_series.get(link_id, [0.0] * series_len)[idx]
                        for link_id in path_links) / path_len
                )
            temporal = (
                max(temporal_values) - min(temporal_values)
                if temporal_values else 0.0
            )

            for wavelength in range(self.config.num_channels):
                if not self._is_wavelength_available(path_links, wavelength):
                    continue
                wavelength_term = self._qot_glr_wavelength_term(path_links, wavelength)
                score = (
                    self.config.qot_glr_accum_weight * accum
                    + self.config.qot_glr_bottleneck_weight * bottleneck
                    + self.config.qot_glr_temporal_weight * temporal
                    + self.config.qot_glr_wavelength_weight * wavelength_term
                )
                if score < best_score:
                    best_score = score
                    best = (path_sats, path_links, wavelength, score)

        if best is None:
            result.reject_reason = RejectReason.NO_WAVELENGTH
            return result

        path_sats, path_links, wavelength, score = best
        self.rwa.allocate_wavelength(path_links, wavelength)
        result = self._compute_path_result_from_osnr(
            request,
            path_sats,
            path_links,
            wavelength,
            link_osnr_db,
        )
        result.constraint_violations = [f"QoT-GLR score={score:.4f}"]
        return result

    def process_request(
        self, request: LightpathRequest
    ) -> LightpathResult:
        """
        Process a single lightpath request.

        1. Build wavelength-aware OSNR lookup
        2. Find route and assign wavelength via RWA
        3. Verify QoT constraints
        """
        link_osnr_db = self._build_link_osnr_lookup()

        if self.config.routing_strategy.lower() == "qot_glr":
            return self._assign_qot_glr_lightpath(request, link_osnr_db)

        result = self.rwa.assign_lightpath(
            request=request,
            link_id_to_osnr=link_osnr_db,
            link_id_to_ber={},
            link_id_to_distance=self.rwa_link_to_distance,
            link_id_to_sun_angle=self.rwa_link_to_sun_angle,
            sat_id_to_dose_rate=self.rwa_sat_to_dose_rate,
            k_paths=self.config.k_shortest_paths,
            path_duration_yr=self.config.path_duration_yr,
        )

        return result

    def run_requests(
        self,
        requests: List[LightpathRequest],
        verbose: bool = True,
    ) -> List[LightpathResult]:
        """Run a batch of lightpath requests."""
        self.compute_all_impairments()

        results = []
        for req in requests:
            result = self.process_request(req)
            results.append(result)

            if verbose:
                status = "ACCEPTED" if result.accepted else "REJECTED"
                print(
                    f"[{status}] Request {req.request_id}: "
                    f"Sat {req.src_sat} -> Sat {req.dst_sat} | "
                    f"Hops: {result.total_hops} | "
                    f"Wavelength: ch{result.wavelength_channel} | "
                    f"OSNR: {result.osnr_dB:.1f} dB | "
                    f"BER: {format_ber_scientific(result.ber)} | "
                    f"Reason: {result.reject_reason.value if not result.accepted else 'OK'}"
                )

        return results

    def _refresh_impairments_at_time(self, time_seconds: float):
        """Propagate the constellation and recompute all link impairments."""
        self.config.sim_time_seconds = time_seconds
        self.constellation.propagate(time_seconds)
        self._impairment_cache.clear()
        self._init_rwa_data()
        self.compute_all_impairments()

    def evaluate_established_lightpaths_over_time(
        self,
        results: List[LightpathResult],
        duration_s: float,
        step_s: float,
    ) -> List[LightpathQoTSample]:
        """
        Evaluate QoT dynamics for already established lightpaths.

        The path and wavelength of each accepted lightpath are kept fixed. At
        each time sample, the constellation is propagated and impairments are
        recomputed, then OSNR/BER are recomputed for the existing lightpaths.
        """
        if duration_s < 0:
            raise ValueError("duration_s must be non-negative")
        if step_s <= 0:
            raise ValueError("step_s must be positive")

        established = [
            result
            for result in results
            if result.accepted and result.path_links and result.wavelength_channel >= 0
        ]
        if not established:
            return []

        samples: List[LightpathQoTSample] = []
        time_points = []
        t = 0.0
        while t <= duration_s + 1e-9:
            time_points.append(round(t, 9))
            t += step_s
        if time_points[-1] < duration_s:
            time_points.append(duration_s)

        for time_seconds in time_points:
            self._refresh_impairments_at_time(time_seconds)
            link_osnr_db = self._build_link_osnr_lookup()

            for result in established:
                inv_osnr_sum = 0.0
                min_sun_angle = 180.0
                max_distance = 0.0

                for link_id in result.path_links:
                    osnr_dB = link_osnr_db.get(
                        (link_id, result.wavelength_channel),
                        15.0,
                    )
                    osnr_lin = 10.0 ** (osnr_dB / 10.0)
                    inv_osnr_sum += 1.0 / max(osnr_lin, 1e-12)
                    min_sun_angle = min(
                        min_sun_angle,
                        self.rwa_link_to_sun_angle.get(link_id, 180.0),
                    )
                    max_distance = max(
                        max_distance,
                        self.rwa_link_to_distance.get(link_id, 0.0),
                    )

                total_osnr_lin = 1.0 / inv_osnr_sum if inv_osnr_sum > 0 else 0.0
                total_osnr_dB = 10.0 * math.log10(max(total_osnr_lin, 1e-30))
                ber_result = self.ber_calc.compute_ber_from_osnr(total_osnr_dB)

                samples.append(
                    LightpathQoTSample(
                        time_seconds=time_seconds,
                        request_id=result.request_id,
                        path_links=list(result.path_links),
                        wavelength_channel=result.wavelength_channel,
                        osnr_dB=total_osnr_dB,
                        ber=ber_result.ber,
                        q_factor=ber_result.q_factor,
                        min_sun_angle_deg=min_sun_angle,
                        max_link_distance_km=max_distance,
                    )
                )

        return samples

    def print_time_series_qot_summary(self, samples: List[LightpathQoTSample]):
        """Print aggregate QoT dynamics for established lightpaths."""
        if not samples:
            print("\nNo established lightpaths available for time-series QoT evaluation.")
            return

        ber_limit = (
            self.config.ber_threshold_with_fec
            if self.config.use_fec
            else self.config.ber_threshold_no_fec
        )
        osnrs = [s.osnr_dB for s in samples]
        bers = [s.ber for s in samples]
        sun_angles = [s.min_sun_angle_deg for s in samples]
        max_distances = [s.max_link_distance_km for s in samples]
        ber_exceeded = sum(1 for s in samples if s.ber > ber_limit)

        print(f"\n{'='*80}")
        print("TIME-SERIES QOT SUMMARY FOR ESTABLISHED LIGHTPATHS")
        print(f"{'='*80}")
        print(f"Samples: {len(samples)}")
        print(f"Time span: {min(s.time_seconds for s in samples):.1f}-"
              f"{max(s.time_seconds for s in samples):.1f} s")
        print(f"OSNR (dB): min {min(osnrs):.2f}, max {max(osnrs):.2f}, "
              f"mean {np.mean(osnrs):.2f}")
        print(f"log10(BER): min {min(np.log10(bers)):.2f}, "
              f"max {max(np.log10(bers)):.2f}, mean {np.mean(np.log10(bers)):.2f}")
        print(f"BER above threshold: {ber_exceeded}/{len(samples)} samples")
        print(f"Minimum sun angle: {min(sun_angles):.2f} deg")
        print(f"Maximum link distance: {max(max_distances):.1f} km")

    def print_impairment_summary(self):
        """Print a summary of physical impairments for all LISLs."""
        print("\n" + "=" * 80)
        print("PHYSICAL IMPAIRMENT SUMMARY (all LISLs)")
        print("=" * 80)

        distances = []
        fsls = []
        dopplers = []
        osnrs = []
        sun_angles = []
        saa_risks = []
        gain_degs = []

        for imp in self._impairment_cache.values():
            distances.append(imp.link_distance_km)
            fsls.append(imp.free_space_loss_dB)
            dopplers.append(abs(imp.doppler_shift_GHz))
            osnrs.append(imp.osnr_dB)
            sun_angles.append(imp.sun_angle_deg)
            saa_risks.append(imp.saa_risk_factor)
            gain_degs.append(imp.edfa_gain_degradation_dB)

        print(f"{'Metric':<35} {'Min':>10} {'Max':>10} {'Mean':>10}")
        print("-" * 65)
        print(f"{'Link Distance (km)':<35} {min(distances):>10.1f} {max(distances):>10.1f} {np.mean(distances):>10.1f}")
        print(f"{'Free Space Loss (dB)':<35} {min(fsls):>10.1f} {max(fsls):>10.1f} {np.mean(fsls):>10.1f}")
        print(f"{'|Doppler Shift| (GHz)':<35} {min(dopplers):>10.3f} {max(dopplers):>10.3f} {np.mean(dopplers):>10.3f}")
        print(f"{'Sun Angle (deg)':<35} {min(sun_angles):>10.1f} {max(sun_angles):>10.1f} {np.mean(sun_angles):>10.1f}")
        print(f"{'SAA Risk Factor':<35} {min(saa_risks):>10.3f} {max(saa_risks):>10.3f} {np.mean(saa_risks):>10.3f}")
        print(f"{'EDFA Gain Degradation (dB)':<35} {min(gain_degs):>10.3f} {max(gain_degs):>10.3f} {np.mean(gain_degs):>10.3f}")
        print(f"{'Single-Link OSNR (dB)':<35} {min(osnrs):>10.1f} {max(osnrs):>10.1f} {np.mean(osnrs):>10.1f}")

        below_15dB = sum(1 for o in osnrs if o < 15.0)
        below_20dB = sum(1 for o in osnrs if o < 20.0)
        print(f"\nLinks with OSNR < 15 dB: {below_15dB}/{len(osnrs)}")
        print(f"Links with OSNR < 20 dB: {below_20dB}/{len(osnrs)}")

    def print_lightpath_details(self, result: LightpathResult):
        """Print detailed information for a lightpath."""
        print(f"\n{'='*80}")
        print(f"LIGHTPATH DETAILS: Request {result.request_id}")
        print(f"{'='*80}")
        print(f"Source: Sat {result.src_sat}  ->  Destination: Sat {result.dst_sat}")
        print(f"Accepted: {result.accepted}")
        print(f"Hop Count: {result.total_hops}")
        print(f"Wavelength: Channel {result.wavelength_channel}")
        print(f"End-to-End OSNR: {result.osnr_dB:.2f} dB")
        print(f"End-to-End BER: {format_ber_scientific(result.ber)}")
        print(f"Q Factor: {result.q_factor:.2f}")

        print(f"\n{'Hop':<5} {'Link ID':<8} {'Distance':>10} {'OSNR(dB)':>10}")
        print("-" * 40)
        for h in range(result.total_hops):
            link_id = result.path_links[h] if h < len(result.path_links) else -1
            dist = (
                result.per_link_distance_km[h]
                if h < len(result.per_link_distance_km)
                else 0.0
            )
            osnr = (
                result.per_link_osnr_dB[h]
                if h < len(result.per_link_osnr_dB)
                else 0.0
            )
            print(f"{h+1:<5} {link_id:<8} {dist:>8.1f} km {osnr:>9.1f}")

        print(f"\nPath (satellites): {' -> '.join(str(s) for s in result.path_satellites)}")


def create_demo_requests(
    total_sats: int,
    num_requests: int = 10,
    seed: int = 42,
) -> List[LightpathRequest]:
    """Generate random lightpath requests for demonstration."""
    random.seed(seed)
    requests = []
    for i in range(num_requests):
        src = random.randint(0, total_sats - 1)
        dst = random.randint(0, total_sats - 1)
        while dst == src:
            dst = random.randint(0, total_sats - 1)
        requests.append(
            LightpathRequest(
                request_id=i + 1,
                src_sat=src,
                dst_sat=dst,
                bandwidth_Gbps=10.0,
            )
        )
    return requests


def build_small_config() -> SimulationConfig:
    """Build a small-scale configuration for quick testing."""
    return SimulationConfig(
        N_planes=6,
        sats_per_plane=11,
        F=1,
        altitude_km=550.0,
        inclination_deg=53.0,
        num_channels=24,
        channel_spacing_GHz=25.0,
        optical_filter_BW_GHz=12.5,
        demux_isolation_dB=25.0,
        oxc_isolation_dB=30.0,
        sim_time_seconds=0.0,
        max_link_distance_km=5400.0,
    )


def build_leo_config() -> SimulationConfig:
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
        enforce_qot_constraints=False,
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


def build_starlink_config() -> SimulationConfig:
    """Build a Starlink-scale configuration."""
    return SimulationConfig(
        N_planes=72,
        sats_per_plane=22,
        F=11,
        altitude_km=550.0,
        inclination_deg=53.0,
        num_channels=80,
        sim_time_seconds=0.0,
        mission_age_years=3.0,
        background_dose_rate=0.1,
    )


def main():
    """Main simulation entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="QoT-Guaranteed Lightpath Routing Simulation for SONs"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="YAML configuration file path"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Run demonstration with sample requests"
    )
    parser.add_argument(
        "--starlink", action="store_true",
        help="Use Starlink-scale constellation (72x22)"
    )
    parser.add_argument(
        "--leo", action="store_true",
        help="Use representative LEO constellation (12x11)"
    )
    parser.add_argument(
        "--num-requests", type=int, default=10,
        help="Number of random lightpath requests"
    )
    parser.add_argument(
        "--verbose", action="store_true", default=True,
        help="Print detailed output"
    )
    parser.add_argument(
        "--detail", type=int, default=None,
        help="Print detailed breakdown for a specific request ID"
    )
    parser.add_argument(
        "--observe-duration", type=float, default=None,
        help="Observe established-lightpath QoT dynamics for this many seconds"
    )
    parser.add_argument(
        "--observe-step", type=float, default=None,
        help="Time step in seconds for established-lightpath QoT observation"
    )

    args = parser.parse_args()

    # Build configuration
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        config = SimulationConfig()
        section_maps = {
            "constellation": {
                "N_planes": "N_planes",
                "sats_per_plane": "sats_per_plane",
                "F": "F",
                "altitude_km": "altitude_km",
                "inclination_deg": "inclination_deg",
            },
            "physical_layer": {
                "channel_spacing_GHz": "channel_spacing_GHz",
                "num_channels": "num_channels",
                "carrier_freq_THz": "carrier_freq_THz",
                "data_rate_Gbps": "data_rate_Gbps",
                "tx_power_dBm": "tx_power_dBm",
                "rx_aperture_diameter_mm": "rx_aperture_diameter_mm",
                "tx_divergence_angle_urad": "tx_divergence_angle_urad",
                "optical_efficiency": "optical_efficiency",
                "pointing_loss_dB": "pointing_loss_dB",
                "oxc_insertion_loss_dB": "oxc_insertion_loss_dB",
                "oxc_isolation_dB": "oxc_isolation_dB",
                "optical_filter_BW_GHz": "optical_filter_BW_GHz",
                "demux_isolation_dB": "demux_isolation_dB",
                "rx_thermal_noise_dBm": "rx_thermal_noise_dBm",
                "osnr_ref_BW_GHz": "osnr_ref_BW_GHz",
                "rx_electrical_BW_GHz": "rx_electrical_BW_GHz",
            },
            "edfa": {
                "nominal_gain_dB": "nominal_gain_dB",
                "nominal_nf_dB": "nominal_nf_dB",
                "gain_degradation_slope": "gain_degradation_slope",
                "nf_degradation_slope": "nf_degradation_slope",
            },
            "saa_radiation": {
                "center_lat_deg": "saa_center_lat_deg",
                "center_lon_deg": "saa_center_lon_deg",
                "lat_window_deg": "saa_lat_window_deg",
                "lon_window_deg": "saa_lon_window_deg",
                "background_dose_rate_krad_yr": "background_dose_rate",
                "saa_enhancement_factor": "saa_enhancement",
            },
            "celestial": {
                "sun_avoidance_angle_deg": "sun_avoidance_angle_deg",
                "sun_coupling_scale_deg": "sun_coupling_scale_deg",
                "sun_coupling_order": "sun_coupling_order",
                "sun_spectral_radiance": "sun_spectral_radiance",
                "sky_spectral_radiance": "sky_spectral_radiance",
                "receiver_fov_sr": "fov_solid_angle_sr",
            },
            "rwa": {
                "max_hops": "max_hops",
                "use_fec": "use_fec",
                "ber_threshold_no_fec": "ber_threshold_no_fec",
                "ber_threshold_with_fec": "ber_threshold_with_fec",
                "k_shortest_paths": "k_shortest_paths",
                "max_cumulative_dose_krad": "max_cumulative_dose_krad",
                "enforce_qot_constraints": "enforce_qot_constraints",
                "max_link_distance_km": "max_link_distance_km",
            },
            "simulation": {
                "time_step_s": "sim_time_seconds",
                "earth_radius_km": "earth_radius_km",
                "mission_age_years": "mission_age_years",
                "path_duration_yr": "path_duration_yr",
                "qot_observation_duration_s": "qot_observation_duration_s",
                "qot_observation_step_s": "qot_observation_step_s",
            },
        }
        for section, mapping in section_maps.items():
            for key, value in yaml_config.get(section, {}).items():
                attr = mapping.get(key)
                if attr and hasattr(config, attr):
                    setattr(config, attr, value)
    elif args.starlink:
        config = build_starlink_config()
    elif args.leo:
        config = build_leo_config()
    else:
        config = build_small_config()

    if args.observe_duration is not None:
        config.qot_observation_duration_s = args.observe_duration
    if args.observe_step is not None:
        config.qot_observation_step_s = args.observe_step

    print("=" * 80)
    print("QOT PERFORMANCE SIMULATION FOR TRADITIONAL RWA")
    print("LEO Satellite Optical Network with OXC Architecture")
    print("=" * 80)
    print(f"\nConstellation: {config.N_planes} planes x {config.sats_per_plane} sats/plane")
    print(f"Altitude: {config.altitude_km} km | Inclination: {config.inclination_deg} deg")
    print(f"Total satellites: {config.N_planes * config.sats_per_plane}")
    print(f"WDM channels: {config.num_channels} | Spacing: {config.channel_spacing_GHz} GHz")
    print(f"Data rate: {config.data_rate_Gbps} Gbps | TX power: {config.tx_power_dBm} dBm")
    print(f"Mission age: {config.mission_age_years} yr | "
          f"FEC: {'Enabled' if config.use_fec else 'Disabled'} | "
          f"BER threshold: {config.ber_threshold_with_fec if config.use_fec else config.ber_threshold_no_fec:.0e}")
    print(
        "Routing mode: "
        + (
            "QoT-constrained SP+First-Fit"
            if config.enforce_qot_constraints
            else "Traditional SP+First-Fit with post-route QoT evaluation"
        )
    )

    # Initialize simulation
    sim = QoTSimulation(config)

    # Generate requests
    if args.demo or not args.config:
        requests = create_demo_requests(
            sim.constellation.total_sats,
            num_requests=args.num_requests,
        )

        print(f"\nProcessing {len(requests)} lightpath requests...\n")

        results = sim.run_requests(requests, verbose=args.verbose)

        sim.print_impairment_summary()

        # Request summary
        print(f"\n{'='*80}")
        print("REQUEST SUMMARY")
        print(f"{'='*80}")
        accepted = sum(1 for r in results if r.accepted)
        rejected = len(results) - accepted
        print(f"Total: {len(results)} | Accepted: {accepted} | Rejected: {rejected}")
        print(f"Acceptance rate: {accepted/len(results)*100:.1f}%")

        from collections import Counter
        reasons = Counter(r.reject_reason.value for r in results if not r.accepted)
        if reasons:
            print(f"\nRejection reasons:")
            for reason, count in reasons.most_common():
                print(f"  {reason}: {count}")

        # Print accepted lightpath details
        if args.detail is not None:
            for r in results:
                if r.request_id == args.detail and r.accepted:
                    sim.print_lightpath_details(r)
                    break
        elif accepted > 0:
            for r in results:
                if r.accepted:
                    sim.print_lightpath_details(r)
                    break

        # Summary statistics for accepted paths
        accepted_results = [r for r in results if r.accepted]
        if accepted_results:
            osnrs = [r.osnr_dB for r in accepted_results]
            bers = [r.ber for r in accepted_results]
            hops = [r.total_hops for r in accepted_results]
            ber_limit = (
                config.ber_threshold_with_fec
                if config.use_fec
                else config.ber_threshold_no_fec
            )
            ber_exceeded = sum(1 for r in accepted_results if r.ber > ber_limit)
            sun_flagged = sum(
                1
                for r in accepted_results
                if any("Sun blocked" in v for v in r.constraint_violations)
            )

            print(f"\n{'='*80}")
            print("ACCEPTED LIGHTPATH STATISTICS")
            print(f"{'='*80}")
            print(f"{'Metric':<25} {'Min':>10} {'Max':>10} {'Mean':>10}")
            print("-" * 55)
            print(f"{'OSNR (dB)':<25} {min(osnrs):>10.1f} {max(osnrs):>10.1f} {np.mean(osnrs):>10.1f}")
            print(f"{'log10(BER)':<25} {min(np.log10(bers)):>10.1f} {max(np.log10(bers)):>10.1f} {np.mean(np.log10(bers)):>10.1f}")
            print(f"{'Hop Count':<25} {min(hops):>10} {max(hops):>10} {np.mean(hops):>10.1f}")
            if not config.enforce_qot_constraints:
                print("\nPost-route QoT observations:")
                print(f"  BER above threshold: {ber_exceeded}/{len(accepted_results)}")
                print(f"  Sun-angle flagged: {sun_flagged}/{len(accepted_results)}")

            if config.qot_observation_duration_s > 0:
                qot_samples = sim.evaluate_established_lightpaths_over_time(
                    accepted_results,
                    duration_s=config.qot_observation_duration_s,
                    step_s=config.qot_observation_step_s,
                )
                sim.print_time_series_qot_summary(qot_samples)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()
