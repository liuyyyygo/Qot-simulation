"""
OSNR accumulation and BER mapping for satellite optical networks.

Models:
  1. Single-link OSNR computation from all impairment contributions
  2. Multi-hop OSNR accumulation through transparent OXC paths
  3. BER estimation from OSNR for OOK modulation (Gaussian approximation)

References:
  Agrawal, G.P., Fiber-Optic Communication Systems, 5th ed., Wiley, 2021.
  Marcuse, D., JLT, 8(12), 1816-1823, 1990.
  Humblet, P.A., Azizoglu, M., JLT, 9(11), 1576-1582, 1991.
  Bergano, N.S., et al., IEEE PTL, 5(3), 304-306, 1993.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import math


@dataclass
class LinkOSNRResult:
    """OSNR result for a single LISL."""
    link_id: int
    sat_i: int
    sat_j: int
    wavelength_channel: int
    signal_power_dBm: float
    signal_power_W: float
    ase_noise_W: float
    celestial_noise_W: float
    inter_xt_noise_var: float
    intra_xt_noise_var: float
    thermal_noise_W: float
    total_noise_W: float
    osnr_linear: float
    osnr_dB: float


@dataclass
class PathOSNRResult:
    """OSNR result for a multi-hop lightpath."""
    path_links: List[int]
    wavelength_channel: int
    per_link_osnr_dB: List[float]
    per_link_osnr_linear: List[float]
    total_osnr_linear: float
    total_osnr_dB: float
    num_hops: int


@dataclass
class BERResult:
    """BER result for a lightpath."""
    osnr_dB: float
    osnr_linear: float
    q_factor: float
    ber: float
    modulation: str
    data_rate_Gbps: float
    ber_acceptable: bool
    fec_threshold: float


class OSNRCalculator:
    """
    OSNR calculator for single link and multi-hop transparent paths.

    Single-link OSNR:
      OSNR = P_sig / P_noise_total
      P_noise_total = P_ASE + P_cel + P_thermal + sigma²_inter/R² + sigma²_intra/R²

    Multi-hop OSNR accumulation (no regeneration):
      1/OSNR_total = sum(1/OSNR_i) for i in 1..H
    """

    def __init__(
        self,
        osnr_ref_BW_GHz: float = 12.5,
        rx_electrical_BW_GHz: float = 7.5,
        thermal_noise_dBm: float = -40.0,
        responsivity_A_W: float = 1.0,
    ):
        self.B_ref_Hz = osnr_ref_BW_GHz * 1e9
        self.B_e_Hz = rx_electrical_BW_GHz * 1e9
        self.thermal_noise_W = 10.0 ** ((thermal_noise_dBm - 30.0) / 10.0)
        self.responsivity = responsivity_A_W

    def compute_single_link_osnr(
        self,
        signal_power_W: float,
        ase_noise_W: float,
        celestial_noise_W: float,
        inter_xt_noise_var: float,
        intra_xt_noise_var: float,
    ) -> Tuple[float, float, float]:
        """Compute single-link OSNR. Returns (osnr_linear, osnr_dB, total_noise_W)."""
        xt_equivalent_W = math.sqrt(
            max(0.0, inter_xt_noise_var + intra_xt_noise_var)
        ) / max(self.responsivity, 1e-12)

        total_noise = (
            ase_noise_W
            + celestial_noise_W
            + self.thermal_noise_W
            + xt_equivalent_W
        )

        if total_noise <= 0 or signal_power_W <= 0:
            return 1e-6, -60.0, total_noise

        osnr_linear = signal_power_W / total_noise
        osnr_dB = 10.0 * math.log10(osnr_linear)

        return osnr_linear, osnr_dB, total_noise

    def compute_path_osnr(
        self,
        per_link_osnr_linear: List[float],
    ) -> Tuple[float, float]:
        """Compute end-to-end OSNR for a transparent multi-hop path."""
        if not per_link_osnr_linear:
            return 0.0, -float("inf")

        inv_sum = sum(1.0 / max(osnr, 1e-12) for osnr in per_link_osnr_linear)

        if inv_sum <= 0:
            return 0.0, -float("inf")

        osnr_total = 1.0 / inv_sum
        osnr_dB = 10.0 * math.log10(osnr_total)

        return osnr_total, osnr_dB


class BERCalculator:
    """
    BER calculator for OOK modulated optical links.

    For ASE-noise-dominated systems:
      Q ≈ sqrt(OSNR * B_ref / (2 * B_e))
      BER = 0.5 * erfc(Q / sqrt(2))
    """

    def __init__(
        self,
        modulation: str = "OOK",
        osnr_ref_BW_GHz: float = 12.5,
        rx_electrical_BW_GHz: float = 7.5,
        ber_threshold_no_fec: float = 1.0e-12,
        ber_threshold_with_fec: float = 2.0e-3,
        use_fec: bool = True,
    ):
        self.modulation = modulation
        self.B_ref_Hz = osnr_ref_BW_GHz * 1e9
        self.B_e_Hz = rx_electrical_BW_GHz * 1e9
        self.ber_threshold_no_fec = ber_threshold_no_fec
        self.ber_threshold_with_fec = ber_threshold_with_fec
        self.use_fec = use_fec
        self.B_ratio = self.B_ref_Hz / self.B_e_Hz

    @property
    def ber_threshold(self) -> float:
        return (
            self.ber_threshold_with_fec
            if self.use_fec
            else self.ber_threshold_no_fec
        )

    def compute_q_factor_osnr(self, osnr_linear: float) -> float:
        """Compute Q factor from OSNR. Q ≈ sqrt(OSNR · B_o / (2 · B_e))"""
        if osnr_linear <= 0:
            return 0.0
        return math.sqrt(max(osnr_linear, 0.0) * self.B_ratio / 2.0)

    def compute_q_factor_exact(
        self,
        osnr_linear: float,
        extinction_ratio_dB: float = 13.0,
    ) -> float:
        """More accurate Q factor including extinction ratio."""
        if osnr_linear <= 0:
            return 0.0

        r_ex = 10.0 ** (-extinction_ratio_dB / 10.0)
        q_0 = math.sqrt(max(0.0, 2.0 * osnr_linear * self.B_ratio))
        er_correction = (1.0 - r_ex) / (1.0 + math.sqrt(r_ex))
        return q_0 * er_correction

    def compute_ber_from_q(self, q_factor: float) -> float:
        """BER = 0.5 * erfc(Q / sqrt(2))"""
        if q_factor <= 0:
            return 0.5
        q_norm = q_factor / math.sqrt(2.0)
        ber = 0.5 * math.erfc(q_norm)
        return max(ber, 1e-300)

    def compute_ber_from_osnr(
        self,
        osnr_dB: float,
        use_exact: bool = False,
    ) -> "BERResult":
        """Compute BER from OSNR (dB)."""
        osnr_linear = 10.0 ** (osnr_dB / 10.0)

        if use_exact:
            q = self.compute_q_factor_exact(osnr_linear)
        else:
            q = self.compute_q_factor_osnr(osnr_linear)

        ber = self.compute_ber_from_q(q)
        acceptable = ber < self.ber_threshold

        return BERResult(
            osnr_dB=osnr_dB,
            osnr_linear=osnr_linear,
            q_factor=q,
            ber=ber,
            modulation=self.modulation,
            data_rate_Gbps=self.B_e_Hz / 0.75 / 1e9,
            ber_acceptable=acceptable,
            fec_threshold=(
                self.ber_threshold_with_fec
                if self.use_fec
                else self.ber_threshold_no_fec
            ),
        )

    def compute_required_osnr_dB(self, target_ber: float) -> float:
        """Compute required OSNR (dB) for target BER."""
        if target_ber >= 0.5:
            return 0.0

        q_required = math.sqrt(-2.0 * math.log(target_ber * math.sqrt(2.0 * math.pi)))

        for _ in range(3):
            ber_current = 0.5 * math.erfc(q_required / math.sqrt(2.0))
            if ber_current < 1e-300:
                break
            dber_dq = -math.exp(-0.5 * q_required**2) / math.sqrt(2.0 * math.pi)
            if abs(dber_dq) < 1e-300:
                break
            q_required -= (ber_current - target_ber) / dber_dq

        required_osnr_linear = 2.0 * q_required**2 / self.B_ratio
        return 10.0 * math.log10(required_osnr_linear)


def format_ber_scientific(ber: float) -> str:
    """Format BER in scientific notation."""
    if ber <= 0:
        return "0.0"
    exponent = int(math.floor(math.log10(ber)))
    mantissa = ber / (10.0 ** exponent)
    return f"{mantissa:.2f}×10^{exponent}"
