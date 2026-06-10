"""
Walker-Delta LEO satellite constellation model.

Models satellite positions and velocities using circular orbit approximation
(two-body problem), avoiding external SGP4 dependencies.

Reference:
  J.G. Walker, "Satellite Constellations," Journal of the British
  Interplanetary Society, vol.37, pp.559-571, 1984.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class SatelliteState:
    """Instantaneous state of a satellite."""
    sat_id: int
    plane_id: int
    index_in_plane: int
    lat_deg: float
    lon_deg: float
    altitude_km: float
    vx_ms: float
    vy_ms: float
    vz_ms: float
    cumulative_dose_krad: float = 0.0
    in_saa: bool = False


@dataclass
class LISL:
    """Inter-satellite link definition."""
    link_id: int
    sat_i: int
    sat_j: int
    link_type: str  # 'intra_orbit' or 'inter_orbit'
    distance_km: float = 0.0
    unit_vector: np.ndarray = None
    sun_angle_deg: float = 180.0
    doppler_shift_GHz: float = 0.0


class WalkerDeltaConstellation:
    """
    Configurable Walker-Delta constellation.

    Uses circular orbit two-body approximation for position/velocity computations.
    """

    GM = 3.986004418e14          # Earth gravitational constant (m³/s²)
    EARTH_RADIUS_M = 6371000.0   # Earth radius (m)
    EARTH_ANGULAR_VEL = 7.2921159e-5  # Earth rotation rate (rad/s)

    def __init__(
        self,
        N_planes: int = 72,
        sats_per_plane: int = 22,
        F: int = 11,
        altitude_km: float = 550.0,
        inclination_deg: float = 53.0,
        max_lisl_per_sat: int = 4,
    ):
        self.N_planes = N_planes
        self.sats_per_plane = sats_per_plane
        self.F = F
        self.altitude_m = altitude_km * 1000.0
        self.altitude_km = altitude_km
        self.inclination_rad = math.radians(inclination_deg)
        self.inclination_deg = inclination_deg
        self.max_lisl_per_sat = max_lisl_per_sat

        self.total_sats = N_planes * sats_per_plane
        self.semi_major_axis_m = self.EARTH_RADIUS_M + self.altitude_m

        # Orbital period (Kepler's third law)
        self.orbital_period_s = (
            2.0 * math.pi * math.sqrt(self.semi_major_axis_m**3 / self.GM)
        )
        self.mean_motion_rad_s = 2.0 * math.pi / self.orbital_period_s

        # Satellite storage
        self.satellites: Dict[int, SatelliteState] = {}
        self.topology: List[LISL] = []

        # Initialize satellite positions
        self._init_satellites()
        self._build_topology()

    def _init_satellites(self):
        """Initialize all satellite positions using Walker-Delta parameters."""
        for p in range(self.N_planes):
            raan_rad = 2.0 * math.pi * p / self.N_planes

            for s in range(self.sats_per_plane):
                sat_id = p * self.sats_per_plane + s

                phase_offset = 2.0 * math.pi * self.F * p / self.total_sats
                mean_anomaly_rad = (
                    2.0 * math.pi * s / self.sats_per_plane + phase_offset
                ) % (2.0 * math.pi)

                lat, lon = self._orbital_elements_to_latlon(
                    mean_anomaly_rad, raan_rad
                )

                self.satellites[sat_id] = SatelliteState(
                    sat_id=sat_id,
                    plane_id=p,
                    index_in_plane=s,
                    lat_deg=lat,
                    lon_deg=lon,
                    altitude_km=self.altitude_km,
                    vx_ms=0.0,
                    vy_ms=0.0,
                    vz_ms=0.0,
                )

    def _orbital_elements_to_latlon(
        self,
        mean_anomaly_rad: float,
        raan_rad: float,
    ) -> Tuple[float, float]:
        """
        Convert orbital elements to geodetic lat/lon.
        Uses spherical Earth approximation and circular orbit.
        """
        arg_lat_rad = mean_anomaly_rad

        lat_rad = math.asin(
            math.sin(self.inclination_rad) * math.sin(arg_lat_rad)
        )

        lon_rel_rad = math.atan2(
            math.cos(self.inclination_rad) * math.sin(arg_lat_rad),
            math.cos(arg_lat_rad),
        )

        lon_rad = (raan_rad + lon_rel_rad) % (2.0 * math.pi)
        if lon_rad > math.pi:
            lon_rad -= 2.0 * math.pi

        return math.degrees(lat_rad), math.degrees(lon_rad)

    def propagate(self, t_seconds: float) -> Dict[int, SatelliteState]:
        """
        Propagate all satellites to time t (seconds from reference epoch).

        Uses: mean anomaly = M0 + n*t, RAAN constant (no J2 perturbation).
        Velocity derived analytically from circular orbit.
        """
        earth_rotation_rad = self.EARTH_ANGULAR_VEL * t_seconds

        for sat_id, sat in self.satellites.items():
            p = sat.plane_id
            s = sat.index_in_plane

            raan_rad = 2.0 * math.pi * p / self.N_planes
            phase_offset = 2.0 * math.pi * self.F * p / self.total_sats

            mean_anomaly_rad = (
                2.0 * math.pi * s / self.sats_per_plane
                + phase_offset
                + self.mean_motion_rad_s * t_seconds
            ) % (2.0 * math.pi)

            arg_lat_rad = mean_anomaly_rad

            lat_rad = math.asin(
                math.sin(self.inclination_rad) * math.sin(arg_lat_rad)
            )

            lon_rel_rad = math.atan2(
                math.cos(self.inclination_rad) * math.sin(arg_lat_rad),
                math.cos(arg_lat_rad),
            )

            lon_rad = (raan_rad + lon_rel_rad - earth_rotation_rad) % (2.0 * math.pi)
            if lon_rad > math.pi:
                lon_rad -= 2.0 * math.pi

            v_orbital = math.sqrt(self.GM / self.semi_major_axis_m)

            vx_orbital = -v_orbital * math.sin(arg_lat_rad)
            vy_orbital = v_orbital * math.cos(arg_lat_rad)

            cos_raan = math.cos(raan_rad)
            sin_raan = math.sin(raan_rad)
            cos_inc = math.cos(self.inclination_rad)
            sin_inc = math.sin(self.inclination_rad)

            vx = cos_raan * vx_orbital - sin_raan * cos_inc * vy_orbital
            vy = sin_raan * vx_orbital + cos_raan * cos_inc * vy_orbital
            vz = sin_inc * vy_orbital

            self.satellites[sat_id] = SatelliteState(
                sat_id=sat_id,
                plane_id=p,
                index_in_plane=s,
                lat_deg=math.degrees(lat_rad),
                lon_deg=math.degrees(lon_rad),
                altitude_km=self.altitude_km,
                vx_ms=vx,
                vy_ms=vy,
                vz_ms=vz,
                cumulative_dose_krad=sat.cumulative_dose_krad,
                in_saa=sat.in_saa,
            )

        return self.satellites

    def _build_topology(self, max_distance_km: float = 5400.0):
        """
        Build LISL topology using Manhattan Street Network (MSN) pattern.

        Each satellite connects to:
          - 2 intra-orbit neighbors (forward/backward in same plane)
          - 2 inter-orbit neighbors (nearest satellites in adjacent planes)

        Links exceeding max_distance_km (Earth occlusion limit) are excluded.

        Reference: Suh & Ko, J-KICS 2025; Bhattacharjee et al., IEEE WCNC 2023.
        """
        self.topology = []
        link_id = 0

        self.propagate(0.0)

        for p in range(self.N_planes):
            for s in range(self.sats_per_plane):
                sat_i = p * self.sats_per_plane + s

                # Intra-orbit: forward neighbor (circular)
                sat_fwd = p * self.sats_per_plane + (s + 1) % self.sats_per_plane
                if sat_fwd > sat_i:
                    dist = self.compute_link_distance_km(sat_i, sat_fwd)
                    if dist <= max_distance_km:
                        self.topology.append(
                            LISL(
                                link_id=link_id,
                                sat_i=sat_i,
                                sat_j=sat_fwd,
                                link_type="intra_orbit",
                                distance_km=dist,
                            )
                        )
                        link_id += 1

                # Inter-orbit: find nearest satellite in next plane
                p_next = (p + 1) % self.N_planes
                best_sat = None
                best_dist = float("inf")

                search_window = min(5, self.sats_per_plane // 2)
                for ds in range(-search_window, search_window + 1):
                    cand_s = (s + ds) % self.sats_per_plane
                    cand_id = p_next * self.sats_per_plane + cand_s
                    dist = self.compute_link_distance_km(sat_i, cand_id)
                    if dist < best_dist and dist <= max_distance_km:
                        best_dist = dist
                        best_sat = cand_id

                if best_sat is not None:
                    self.topology.append(
                        LISL(
                            link_id=link_id,
                            sat_i=sat_i,
                            sat_j=best_sat,
                            link_type="inter_orbit",
                            distance_km=best_dist,
                        )
                    )
                    link_id += 1

    def get_topology(self) -> List[LISL]:
        """Return the LISL topology."""
        return self.topology

    def get_adjacency_list(self) -> Dict[int, List[int]]:
        """Build adjacency list from topology."""
        adj = {i: [] for i in range(self.total_sats)}
        for lisl in self.topology:
            adj[lisl.sat_i].append(lisl.sat_j)
            adj[lisl.sat_j].append(lisl.sat_i)
        return adj

    def get_position_ecef(self, sat_id: int) -> np.ndarray:
        """
        Get satellite position in ECEF coordinates (meters).

        Reference: Montenbruck & Gill, "Satellite Orbits," Springer, 2000.
        """
        sat = self.satellites[sat_id]
        lat_rad = math.radians(sat.lat_deg)
        lon_rad = math.radians(sat.lon_deg)
        r = self.EARTH_RADIUS_M + sat.altitude_km * 1000.0

        x = r * math.cos(lat_rad) * math.cos(lon_rad)
        y = r * math.cos(lat_rad) * math.sin(lon_rad)
        z = r * math.sin(lat_rad)

        return np.array([x, y, z])

    def get_velocity_ecef(self, sat_id: int) -> np.ndarray:
        """Get satellite velocity in ECEF coordinates (m/s)."""
        sat = self.satellites[sat_id]
        return np.array([sat.vx_ms, sat.vy_ms, sat.vz_ms])

    def compute_link_distance_km(self, sat_i: int, sat_j: int) -> float:
        """Compute Euclidean distance between two satellites (km)."""
        pos_i = self.get_position_ecef(sat_i)
        pos_j = self.get_position_ecef(sat_j)
        return float(np.linalg.norm(pos_j - pos_i)) / 1000.0

    def compute_link_unit_vector(
        self, sat_i: int, sat_j: int
    ) -> np.ndarray:
        """Compute unit vector from sat_i to sat_j."""
        pos_i = self.get_position_ecef(sat_i)
        pos_j = self.get_position_ecef(sat_j)
        vec = pos_j - pos_i
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 0.0])
        return vec / norm

    def compute_relative_radial_velocity_ms(
        self, sat_i: int, sat_j: int
    ) -> float:
        """
        Compute relative radial velocity between two satellites.

        v_radial = (v_j - v_i) · u_ij
        """
        v_i = self.get_velocity_ecef(sat_i)
        v_j = self.get_velocity_ecef(sat_j)
        u_ij = self.compute_link_unit_vector(sat_i, sat_j)
        return float(np.dot(v_j - v_i, u_ij))

    def is_eclipsed(self, sat_id: int) -> bool:
        """Check if satellite is in Earth's shadow (simplified)."""
        return False

    def get_satellite(self, sat_id: int) -> SatelliteState:
        """Get satellite state by ID."""
        return self.satellites[sat_id]
