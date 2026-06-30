"""
Physical impairment models for LEO satellite optical inter-satellite links.

Models five orbit-related impairments:
  1. Free Space Loss (FSL) - Friis transmission equation
  2. Doppler Frequency Shift & Filter Penalty
  3. Celestial Background Light
  4. WDM Crosstalk (inter-wavelength and intra-wavelength)
  5. SAA Radiation Effects on EDFA

References:
  Friis, H.T., Proc. IRE, 34(5), 254-256, 1946.
  Boumalek et al., Int. J. Satell. Commun. Netw., 42(2), 2024.
  Yang, Q., Tan, L., Ma, J., JLT, 28(6), 931-938, 2010.
  Leeb, W.R., Applied Optics, 28(16), 3443-3449, 1989.
  Facchini et al., Applied Sciences, 13(20), 11589, 2023.
  Ladaci et al., J. Applied Physics, 121(16), 163104, 2017.
  Giles, C.R., Desurvire, E., JLT, 9(2), 271-283, 1991.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import math


# Physical constants
C_LIGHT_MS = 2.99792458e8          # Speed of light (m/s)
H_PLANCK = 6.62607015e-34          # Planck constant (J·s)
BOLTZMANN = 1.380649e-23           # Boltzmann constant (J/K)


@dataclass
class ImpairmentResult:
    """Results of physical impairment calculation for a single LISL."""
    link_distance_km: float
    free_space_loss_dB: float
    doppler_shift_GHz: float
    doppler_filter_penalty_dB: float
    celestial_noise_power_W: float
    sun_angle_deg: float
    is_sun_blocked: bool
    inter_xt_noise_var: float
    intra_xt_noise_var: float
    saa_risk_factor: float
    edfa_gain_degradation_dB: float
    edfa_nf_increase_dB: float
    ase_noise_power_W: float
    total_noise_power_W: float
    received_signal_power_W: float
    osnr_linear: float
    osnr_dB: float


class FreeSpaceLoss:
    """
    Free space loss using Friis transmission equation.

    FSL_dB = 20 * log10(4 * pi * d / lambda)
    """

    def __init__(self, wavelength_nm: float = 1550.0):
        self.wavelength = wavelength_nm * 1e-9

    def compute_loss_dB(self, distance_m: float) -> float:
        if distance_m <= 0:
            return 0.0
        return 20.0 * math.log10(4.0 * math.pi * distance_m / self.wavelength)

    def compute_received_power_dBm(
        self,
        tx_power_dBm: float,
        distance_m: float,
        tx_gain_dB: float = 0.0,
        rx_gain_dB: float = 0.0,
    ) -> float:
        fsl_dB = self.compute_loss_dB(distance_m)
        return tx_power_dBm + tx_gain_dB + rx_gain_dB - fsl_dB


class TelescopeGain:
    """
    Telescope/antenna gain for optical systems.

    G_T = 16 / (theta_T)²  (transmitter gain from divergence angle)
    G_R = (pi * D_R / lambda)²  (receiver gain from aperture diameter)

    Reference: Suh & Ko, J-KICS, 50(12), 2025.
    """

    def __init__(self, wavelength_nm: float = 1550.0):
        self.wavelength_m = wavelength_nm * 1e-9

    def tx_gain_dB(self, divergence_half_angle_rad: float) -> float:
        if divergence_half_angle_rad <= 0:
            return 120.0
        gain_linear = 16.0 / (divergence_half_angle_rad ** 2)
        return 10.0 * math.log10(gain_linear)

    def rx_gain_dB(self, aperture_diameter_m: float) -> float:
        gain_linear = (math.pi * aperture_diameter_m / self.wavelength_m) ** 2
        return 10.0 * math.log10(gain_linear)


class DopplerShift:
    """
    Doppler frequency shift in LEO inter-satellite links.

    Delta_f = f_c * v_radial / c

    Reference:
      Boumalek et al., Int. J. Satell. Commun. Netw., 42(2), 2024.
      Yang, Q., Tan, L., Ma, J., JLT, 28(6), 931-938, 2010.
    """

    def __init__(self, carrier_freq_THz: float = 193.414):
        self.carrier_freq_Hz = carrier_freq_THz * 1e12

    def compute_shift_GHz(self, radial_velocity_ms: float) -> float:
        return (self.carrier_freq_Hz * radial_velocity_ms / C_LIGHT_MS) / 1e9

    def compute_filter_penalty_dB(
        self,
        doppler_shift_GHz: float,
        filter_BW_GHz: float = 50.0,
        filter_order: int = 2,
        has_frequency_tracking: bool = False,
        tracking_residual_GHz: float = 0.0,
    ) -> float:
        """
        Compute power penalty from Doppler-induced filter misalignment.

        For a super-Gaussian filter of order n:
        H(f) = exp(-(f/B_3dB)^(2n))

        Reference: Yang, Tan, Ma, JLT 2010.
        """
        if has_frequency_tracking:
            effective_shift = tracking_residual_GHz
        else:
            effective_shift = doppler_shift_GHz

        if abs(effective_shift) < 1e-9:
            return 0.0

        b_eff = filter_BW_GHz / (math.log(2.0) ** (1.0 / (2.0 * filter_order)))

        transmission = math.exp(-((abs(effective_shift) / b_eff) ** (2 * filter_order)))

        if transmission < 1e-12:
            return 100.0

        return -10.0 * math.log10(transmission)


class CelestialBackground:
    """
    Celestial background light noise.

    P_bg = L_sky * Omega_FOV * A_rx * Delta_lambda
         + f_suppress(theta_sun) * L_sun * Omega_FOV * A_rx * Delta_lambda

    Reference:
      Leeb, W.R., Applied Optics, 28(16), 3443-3449, 1989.
      Chen, S.-P., Opt. Quant. Electron., 54:562, 2022.
    """

    def __init__(
        self,
        aperture_diameter_m: float = 0.08,
        fov_solid_angle_sr: float = 1.0e-10,
        filter_spectral_BW_um: float = 0.4e-3,
        sun_spectral_radiance: float = 1.5e6,
        sky_spectral_radiance: float = 3.0e-4,
        sun_avoidance_angle_deg: float = 3.0,
        sun_coupling_scale_deg: float = 5.0,
        sun_coupling_order: float = 4.0,
        direct_sun_noise_ref_W: float = 42.4e-3,
        direct_sun_ref_aperture_m: float = 0.01,
        direct_sun_ref_bandwidth_um: float = 1.0e-3,
    ):
        self.aperture_diameter_m = aperture_diameter_m
        self.aperture_area_m2 = math.pi * (aperture_diameter_m / 2.0) ** 2
        self.fov_sr = fov_solid_angle_sr
        self.filter_BW_um = filter_spectral_BW_um
        self.L_sun = sun_spectral_radiance
        self.L_sky = sky_spectral_radiance
        self.sun_avoidance_rad = math.radians(sun_avoidance_angle_deg)
        self.sun_avoidance_angle_deg = sun_avoidance_angle_deg
        self.sun_coupling_scale_deg = sun_coupling_scale_deg
        self.sun_coupling_order = sun_coupling_order
        self.direct_sun_noise_ref_W = direct_sun_noise_ref_W
        self.direct_sun_ref_aperture_m = direct_sun_ref_aperture_m
        self.direct_sun_ref_bandwidth_um = direct_sun_ref_bandwidth_um

        self.A_Omega_DeltaLambda = (
            self.aperture_area_m2 * self.fov_sr * self.filter_BW_um
        )

    def compute_sun_angle_deg(
        self,
        sat_lat_deg: float,
        sat_lon_deg: float,
        link_unit_vector: np.ndarray,
        sun_dec_deg: float = 0.0,
        sun_ra_deg: float = 0.0,
        time_hours: float = 12.0,
    ) -> float:
        """
        Compute angle between LISL receiving direction and the Sun.

        The receiving direction is -link_unit_vector (pointing toward sat_i).
        """
        lat_rad = math.radians(sat_lat_deg)
        lon_rad = math.radians(sat_lon_deg)
        sun_dec_rad = math.radians(sun_dec_deg)
        sun_ra_rad = math.radians(sun_ra_deg)

        lst_rad = math.radians(15.0 * time_hours) + lon_rad
        hour_angle_rad = lst_rad - sun_ra_rad

        sun_x = math.cos(sun_dec_rad) * math.cos(hour_angle_rad)
        sun_y = math.cos(sun_dec_rad) * math.sin(hour_angle_rad)
        sun_z = math.sin(sun_dec_rad)

        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)

        sun_ecef = np.array([
            -sun_x * sin_lon - sun_y * sin_lat * cos_lon + sun_z * cos_lat * cos_lon,
             sun_x * cos_lon - sun_y * sin_lat * sin_lon + sun_z * cos_lat * sin_lon,
             sun_y * cos_lat + sun_z * sin_lat,
        ])
        sun_ecef = sun_ecef / np.linalg.norm(sun_ecef)

        rx_direction = -link_unit_vector
        if np.linalg.norm(rx_direction) < 1e-9:
            return 180.0

        cos_angle = np.clip(np.dot(rx_direction, sun_ecef), -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def compute_sun_vector_ecef(
        self,
        sun_dec_deg: float = 0.0,
        sun_ra_deg: float = 0.0,
        time_hours: float = 12.0,
    ) -> np.ndarray:
        """Compute a global unit Sun direction vector in the ECEF frame."""
        sun_dec_rad = math.radians(sun_dec_deg)
        sun_ra_rad = math.radians(sun_ra_deg)
        greenwich_hour_angle = math.radians(15.0 * time_hours) - sun_ra_rad

        sun_vec = np.array([
            math.cos(sun_dec_rad) * math.cos(greenwich_hour_angle),
            math.cos(sun_dec_rad) * math.sin(greenwich_hour_angle),
            math.sin(sun_dec_rad),
        ])
        norm = np.linalg.norm(sun_vec)
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0])
        return sun_vec / norm

    def compute_link_sun_angle_deg(
        self,
        receiver_view_vector_ecef: np.ndarray,
        sun_vector_ecef: np.ndarray,
    ) -> float:
        """Angle between a receiver boresight/view vector and the Sun."""
        view_norm = np.linalg.norm(receiver_view_vector_ecef)
        sun_norm = np.linalg.norm(sun_vector_ecef)
        if view_norm < 1e-12 or sun_norm < 1e-12:
            return 180.0

        view = receiver_view_vector_ecef / view_norm
        sun = sun_vector_ecef / sun_norm
        cos_angle = np.clip(np.dot(view, sun), -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    def compute_bidirectional_link_sun_angle_deg(
        self,
        link_unit_vector_ecef: np.ndarray,
        sun_vector_ecef: np.ndarray,
    ) -> float:
        """
        Return the smaller sun angle seen by either endpoint of a bidirectional ISL.

        The link vector points from satellite i to satellite j.  Receiver views
        are +u at satellite i and -u at satellite j.
        """
        angle_i = self.compute_link_sun_angle_deg(
            link_unit_vector_ecef,
            sun_vector_ecef,
        )
        angle_j = self.compute_link_sun_angle_deg(
            -link_unit_vector_ecef,
            sun_vector_ecef,
        )
        return min(angle_i, angle_j)

    def compute_noise_power_W(
        self,
        sun_angle_deg: float,
        optical_efficiency: float = 0.6,
    ) -> Tuple[float, bool]:
        """
        Compute celestial background noise power.

        Reference: Leeb (1989) for sun angle dependency.
        """
        sun_angle_rad = math.radians(sun_angle_deg)
        is_blocked = sun_angle_rad <= self.sun_avoidance_rad

        if is_blocked:
            solar_coupling = 1.0
        else:
            scale = max(self.sun_coupling_scale_deg, 1e-9)
            excess_angle_deg = max(0.0, sun_angle_deg - self.sun_avoidance_angle_deg)
            solar_coupling = math.exp(
                -((excess_angle_deg / scale) ** self.sun_coupling_order)
            )

        sky_noise = self.L_sky * self.A_Omega_DeltaLambda * optical_efficiency
        direct_sun_noise = self.direct_sun_noise_ref_W
        direct_sun_noise *= (
            max(self.aperture_diameter_m, 1e-12)
            / max(self.direct_sun_ref_aperture_m, 1e-12)
        ) ** 2
        direct_sun_noise *= (
            max(self.filter_BW_um, 1e-12)
            / max(self.direct_sun_ref_bandwidth_um, 1e-12)
        )
        direct_sun_noise *= optical_efficiency

        p_noise = sky_noise + solar_coupling * direct_sun_noise

        return p_noise, is_blocked


class WDMCrosstalk:
    """
    WDM crosstalk noise model.

    Two types:
    1. Inter-wavelength (adjacent channel leakage through filter)
    2. Intra-wavelength (same-wavelength from other OXC ports)

    Reference:
      Yang, Q., Tan, L., Ma, J., JLT, 28(6), 931-938, 2010.
      Goldstein et al., IEEE PTL, 6(5), 657-660, 1994.
    """

    def __init__(
        self,
        channel_spacing_GHz: float = 50.0,
        filter_BW_GHz: float = 50.0,
        demux_isolation_dB: float = 30.0,
        oxc_isolation_dB: float = 35.0,
        num_channels: int = 80,
    ):
        self.channel_spacing_GHz = channel_spacing_GHz
        self.filter_BW_GHz = filter_BW_GHz
        self.demux_isolation_linear = 10.0 ** (-demux_isolation_dB / 10.0)
        self.oxc_isolation_linear = 10.0 ** (-oxc_isolation_dB / 10.0)
        self.num_channels = num_channels

    def inter_channel_crosstalk_coeff(
        self,
        target_channel: int,
        interfering_channel: int,
        doppler_shift_interferer_GHz: float = 0.0,
    ) -> float:
        """
        Compute crosstalk coefficient from channel m to target channel k.

        Reference: Yang, Tan, Ma, JLT 2010, Eq.(15).
        """
        if target_channel == interfering_channel:
            return 1.0

        delta_ch = interfering_channel - target_channel
        center_freq_offset = delta_ch * self.channel_spacing_GHz
        effective_offset = center_freq_offset + doppler_shift_interferer_GHz

        b_eff = self.filter_BW_GHz / (math.log(2.0) ** 0.25)
        rejection = math.exp(-((effective_offset / b_eff) ** 4))

        return rejection * self.demux_isolation_linear

    def compute_inter_wavelength_noise_var(
        self,
        target_channel: int,
        occupied_channels: List[int],
        received_power_per_channel_W: float,
        responsivity_A_W: float = 1.0,
        doppler_shifts: Optional[Dict[int, float]] = None,
    ) -> float:
        """
        Compute inter-wavelength crosstalk noise variance.

        Reference: Yang, Tan, Ma, JLT 2010, Eq.(18-19).
        """
        if doppler_shifts is None:
            doppler_shifts = {}

        total_variance = 0.0
        for ch in occupied_channels:
            if ch == target_channel:
                continue
            doppler = doppler_shifts.get(ch, 0.0)
            eps = self.inter_channel_crosstalk_coeff(target_channel, ch, doppler)
            variance = (eps * responsivity_A_W * received_power_per_channel_W) ** 2
            total_variance += variance

        return total_variance

    def compute_intra_wavelength_noise_var(
        self,
        num_interfering_ports: int,
        received_power_W: float,
        responsivity_A_W: float = 1.0,
    ) -> float:
        """
        Compute intra-wavelength crosstalk noise variance.

        Reference: Goldstein et al., IEEE PTL, 1994.
        """
        eps = self.oxc_isolation_linear
        variance = (
            num_interfering_ports
            * (eps * responsivity_A_W * received_power_W) ** 2
        )
        return variance


class SAARadiation:
    """
    South Atlantic Anomaly (SAA) radiation model and EDFA degradation.

    Reference:
      Facchini et al., Applied Sciences, 13(20), 11589, 2023.
      Ladaci et al., J. Applied Physics, 121(16), 163104, 2017.
      Giles, C.R., Desurvire, E., JLT, 9(2), 271-283, 1991.
    """

    def __init__(
        self,
        saa_center_lat_deg: float = -25.0,
        saa_center_lon_deg: float = -45.0,
        lat_window_deg: float = 15.0,
        lon_window_deg: float = 30.0,
        background_dose_rate_krad_yr: float = 0.1,
        saa_enhancement_factor: float = 10.0,
        amplitude_calibration: float = 1.0,
        nominal_gain_dB: float = 20.0,
        nominal_nf_dB: float = 5.0,
        gain_degradation_slope: float = 0.10,
        nf_degradation_slope: float = 0.20,
        wavelength_nm: float = 1550.0,
        osnr_ref_BW_GHz: float = 12.5,
        polarization_modes: int = 2,
    ):
        self.saa_lat0 = saa_center_lat_deg
        self.saa_lon0 = saa_center_lon_deg
        self.sigma_lat = lat_window_deg
        self.sigma_lon = lon_window_deg
        self.background_rate = background_dose_rate_krad_yr
        self.enhancement = saa_enhancement_factor
        self.amplitude = amplitude_calibration
        self.nominal_gain_dB = nominal_gain_dB
        self.nominal_nf_dB = nominal_nf_dB
        self.k_G = gain_degradation_slope
        self.k_NF = nf_degradation_slope
        self.wavelength_m = wavelength_nm * 1e-9
        self.freq_Hz = C_LIGHT_MS / self.wavelength_m
        self.B_ref_Hz = osnr_ref_BW_GHz * 1e9
        self.pol_modes = polarization_modes

    def compute_saa_risk_factor(
        self, lat_deg: float, lon_deg: float
    ) -> float:
        """
        Compute normalized SAA radiation risk factor using 2D Gaussian field.
        """
        dlat = lat_deg - self.saa_lat0
        dlon = lon_deg - self.saa_lon0

        if dlon > 180.0:
            dlon -= 360.0
        elif dlon < -180.0:
            dlon += 360.0

        risk = self.amplitude * math.exp(
            -0.5 * (dlat / self.sigma_lat) ** 2
            - 0.5 * (dlon / self.sigma_lon) ** 2
        )
        return risk

    def compute_dose_rate_krad_yr(
        self, lat_deg: float, lon_deg: float, altitude_km: float = 550.0
    ) -> float:
        """Compute instantaneous dose rate at a position."""
        risk = self.compute_saa_risk_factor(lat_deg, lon_deg)
        alt_factor = math.exp((altitude_km - 550.0) / 200.0)

        effective_rate = (
            self.background_rate
            * (1.0 + (self.enhancement - 1.0) * risk)
            * alt_factor
        )
        return effective_rate

    def update_cumulative_dose_krad(
        self,
        current_dose_krad: float,
        lat_deg: float,
        lon_deg: float,
        altitude_km: float,
        time_step_years: float,
    ) -> float:
        """Update cumulative radiation dose."""
        dose_rate = self.compute_dose_rate_krad_yr(lat_deg, lon_deg, altitude_km)
        return current_dose_krad + dose_rate * time_step_years

    def compute_gain_degradation_dB(self, cumulative_dose_krad: float) -> float:
        """Low-dose linear proxy: Delta_G_dB(D) = k_G * D."""
        return self.k_G * cumulative_dose_krad

    def compute_effective_gain_dB(self, cumulative_dose_krad: float) -> float:
        degradation = self.compute_gain_degradation_dB(cumulative_dose_krad)
        return max(0.0, self.nominal_gain_dB - degradation)

    def compute_nf_increase_dB(self, cumulative_dose_krad: float) -> float:
        """Low-dose linear proxy: Delta_NF_dB(D) = k_NF * D."""
        return self.k_NF * cumulative_dose_krad

    def compute_effective_nf_dB(self, cumulative_dose_krad: float) -> float:
        increase = self.compute_nf_increase_dB(cumulative_dose_krad)
        return self.nominal_nf_dB + increase

    def compute_ase_noise_power_W(
        self,
        gain_linear: float,
        nf_linear: float,
    ) -> float:
        """
        Compute ASE noise power in the reference bandwidth.

        P_ASE = n_sp * (G - 1) * h * nu * B_ref * N_pol
        """
        if gain_linear <= 1.0:
            gain_linear = 10.0 ** (self.nominal_gain_dB / 10.0)

        n_sp = nf_linear / 2.0
        photon_energy = H_PLANCK * self.freq_Hz

        p_ase = (
            self.pol_modes
            * n_sp
            * photon_energy
            * (gain_linear - 1.0)
            * self.B_ref_Hz
        )
        return p_ase


def compute_thermal_noise_power_W(
    noise_figure_dB: float = 4.0,
    bandwidth_Hz: float = 7.5e9,
    temperature_K: float = 290.0,
) -> float:
    """Compute receiver thermal noise power."""
    nf_linear = 10.0 ** (noise_figure_dB / 10.0)
    return BOLTZMANN * temperature_K * bandwidth_Hz * nf_linear
