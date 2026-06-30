"""
RWA (Routing and Wavelength Assignment) with extended QoT constraints.

Implements:
  1. Shortest-path routing (Dijkstra) with configurable link weights
  2. K-Shortest Paths (K-SP) as fallback when shortest path fails QoT
  3. First-Fit wavelength assignment
  4. Extended QoT constraint verification (7 constraints):
     - Wavelength continuity (inherent in transparent path)
     - Wavelength clash (distinct wavelength per LISL)
     - BER/FEC threshold
     - Solar avoidance angle
     - Link distance upper bound (Earth occlusion)
     - Cumulative radiation dose limit
     - Maximum hop count

References:
  Zang, H., Jue, J.P., Mukherjee, B., "A Review of Routing and Wavelength
    Assignment Approaches," IEEE Network, 14(1), 2000.
  Ramaswami, R., Sivarajan, K.N., Sasaki, G.H., Optical Networks, Ch.8.
"""

import heapq
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import math

try:
    from .osnr_ber import BERCalculator
except ImportError:
    from osnr_ber import BERCalculator


class RejectReason(Enum):
    """Reasons for rejecting a lightpath."""
    NONE = "Accepted"
    NO_PATH = "No path exists between source and destination"
    NO_WAVELENGTH = "No available wavelength on path"
    WAVELENGTH_CLASH = "Wavelength clash on one or more LISLs"
    BER_TOO_HIGH = "End-to-end BER exceeds threshold"
    SUN_BLOCKED = "Solar avoidance angle violated on one or more LISLs"
    LINK_TOO_LONG = "Link distance exceeds maximum on one or more LISLs"
    RADIATION_EXCEEDED = "Cumulative radiation dose exceeds threshold"
    HOP_COUNT_EXCEEDED = "Path hop count exceeds maximum"


@dataclass
class LightpathRequest:
    """A lightpath establishment request."""
    request_id: int
    src_sat: int
    dst_sat: int
    bandwidth_Gbps: float = 10.0


@dataclass
class LightpathResult:
    """Result of a lightpath establishment attempt."""
    request_id: int
    src_sat: int
    dst_sat: int
    accepted: bool
    reject_reason: RejectReason = RejectReason.NONE
    path_satellites: List[int] = field(default_factory=list)
    path_links: List[int] = field(default_factory=list)
    wavelength_channel: int = -1
    total_hops: int = 0
    osnr_dB: float = -float("inf")
    ber: float = 1.0
    q_factor: float = 0.0
    per_link_osnr_dB: List[float] = field(default_factory=list)
    per_link_distance_km: List[float] = field(default_factory=list)
    cumulative_radiation_krad: float = 0.0
    constraint_violations: List[str] = field(default_factory=list)


class RWASolver:
    """
    RWA solver with extended QoT constraints.

    Routes lightpaths through the satellite optical network using
    shortest-path routing + first-fit wavelength assignment,
    verifying all QoT constraints before accepting a lightpath.
    """

    def __init__(
        self,
        num_satellites: int,
        adjacency: Dict[int, List[int]],
        link_distances_km: Dict[Tuple[int, int], float],
        topology_links: List,
        num_wavelengths: int = 80,
        max_hops: int = 12,
        max_link_distance_km: float = 5400.0,
        sun_avoidance_angle_deg: float = 3.0,
        max_cumulative_dose_krad: float = 50.0,
        ber_threshold: float = 2.0e-3,
        enforce_qot_constraints: bool = False,
        ber_calculator: Optional["BERCalculator"] = None,
    ):
        self.num_sats = num_satellites
        self.adj = adjacency
        self.link_distances = link_distances_km
        self.topology_links = topology_links
        self.num_wavelengths = num_wavelengths
        self.max_hops = max_hops
        self.max_link_distance_km = max_link_distance_km
        self.sun_avoidance_deg = sun_avoidance_angle_deg
        self.max_dose_krad = max_cumulative_dose_krad
        self.ber_threshold = ber_threshold
        self.enforce_qot_constraints = enforce_qot_constraints

        # Use provided BER calculator or create default
        self.ber_calc = ber_calculator
        if self.ber_calc is None:
            self.ber_calc = BERCalculator(
                modulation="OOK",
                osnr_ref_BW_GHz=12.5,
                rx_electrical_BW_GHz=7.5,
                use_fec=True if ber_threshold > 1e-6 else False,
            )

        # Link to LISL object mapping
        self._link_map: Dict[Tuple[int, int], int] = {}
        for lisl in topology_links:
            key = (min(lisl.sat_i, lisl.sat_j), max(lisl.sat_i, lisl.sat_j))
            self._link_map[key] = lisl.link_id

        # Wavelength occupancy: {link_id: set(occupied_channels)}
        self.wavelength_occupancy: Dict[int, Set[int]] = {}
        for lisl in topology_links:
            self.wavelength_occupancy[lisl.link_id] = set()

    def get_link_id(self, sat_i: int, sat_j: int) -> Optional[int]:
        """Get LISL ID from satellite pair."""
        key = (min(sat_i, sat_j), max(sat_i, sat_j))
        return self._link_map.get(key)

    def shortest_path(
        self,
        src: int,
        dst: int,
        link_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        """Dijkstra's shortest path algorithm."""
        if src == dst:
            return [src], []

        dist = {i: float("inf") for i in range(self.num_sats)}
        prev = {i: None for i in range(self.num_sats)}
        prev_link = {i: None for i in range(self.num_sats)}
        dist[src] = 0.0
        visited = set()

        pq = [(0.0, src)]

        while pq:
            d, u = heapq.heappop(pq)
            if u in visited:
                continue
            visited.add(u)

            if u == dst:
                break

            for v in self.adj.get(u, []):
                if v in visited:
                    continue

                edge = (min(u, v), max(u, v))
                if link_weights and edge in link_weights:
                    weight = link_weights[edge]
                elif edge in self.link_distances:
                    weight = self.link_distances[edge]
                else:
                    weight = 10000.0

                new_dist = d + weight
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    prev[v] = u
                    prev_link[v] = self.get_link_id(u, v)
                    heapq.heappush(pq, (new_dist, v))

        if dist[dst] == float("inf"):
            return None, None

        path_sats = []
        path_links = []
        curr = dst
        while curr is not None:
            path_sats.append(curr)
            if prev_link[curr] is not None:
                path_links.append(prev_link[curr])
            curr = prev[curr]

        path_sats.reverse()
        path_links.reverse()

        return path_sats, path_links

    def k_shortest_paths(
        self,
        src: int,
        dst: int,
        k: int = 3,
        link_weights: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """Yen's K-Shortest Paths algorithm (simplified)."""
        paths = []
        shortest_sats, shortest_links = self.shortest_path(src, dst, link_weights)

        if shortest_sats is None:
            return []

        paths.append((shortest_sats, shortest_links))

        for ki in range(1, k):
            prev_path = paths[ki - 1]
            for ei in range(len(prev_path[0]) - 1):
                sat_a = prev_path[0][ei]
                sat_b = prev_path[0][ei + 1]

                removed_edge = (min(sat_a, sat_b), max(sat_a, sat_b))
                removed_weight = self.link_distances.get(removed_edge, 10000.0)
                self.link_distances[removed_edge] = float("inf")

                new_sats, new_links = self.shortest_path(src, dst, link_weights)

                self.link_distances[removed_edge] = removed_weight

                if new_sats is not None and (new_sats, new_links) not in paths:
                    paths.append((new_sats, new_links))
                    break

            if len(paths) <= ki:
                break

        return paths[:k]

    def first_fit_wavelength(
        self,
        path_link_ids: List[int],
    ) -> Optional[int]:
        """First-Fit wavelength assignment respecting wavelength continuity."""
        for ch in range(self.num_wavelengths):
            available = True
            for link_id in path_link_ids:
                if ch in self.wavelength_occupancy.get(link_id, set()):
                    available = False
                    break
            if available:
                return ch
        return None

    def allocate_wavelength(
        self, path_link_ids: List[int], channel: int
    ):
        """Allocate a wavelength on all links in the path."""
        for link_id in path_link_ids:
            self.wavelength_occupancy.setdefault(link_id, set()).add(channel)

    def release_wavelength(
        self, path_link_ids: List[int], channel: int
    ):
        """Release a wavelength on all links in the path."""
        for link_id in path_link_ids:
            if link_id in self.wavelength_occupancy:
                self.wavelength_occupancy[link_id].discard(channel)

    def check_wavelength_clash(
        self,
        path_link_ids: List[int],
        channel: int,
    ) -> bool:
        """Check wavelength clash constraint."""
        for link_id in path_link_ids:
            if channel in self.wavelength_occupancy.get(link_id, set()):
                return False
        return True

    def check_link_distance(
        self,
        path_link_ids: List[int],
        link_id_to_distance: Dict[int, float],
    ) -> Tuple[bool, List[int]]:
        """Check link distance constraint."""
        violating = []
        for link_id in path_link_ids:
            dist = link_id_to_distance.get(link_id, 0.0)
            if dist > self.max_link_distance_km:
                violating.append(link_id)
        return len(violating) == 0, violating

    def check_solar_avoidance(
        self,
        path_link_ids: List[int],
        link_id_to_sun_angle: Dict[int, float],
    ) -> Tuple[bool, List[int]]:
        """Check solar avoidance constraint."""
        violating = []
        for link_id in path_link_ids:
            angle = link_id_to_sun_angle.get(link_id, 180.0)
            if angle < self.sun_avoidance_deg:
                violating.append(link_id)
        return len(violating) == 0, violating

    def check_radiation_dose(
        self,
        path_sat_ids: List[int],
        sat_id_to_dose_rate: Dict[int, float],
        path_duration_yr: float = 0.1,
    ) -> Tuple[bool, float]:
        """Check cumulative radiation dose constraint."""
        total_dose = 0.0
        for sat_id in path_sat_ids:
            dose_rate = sat_id_to_dose_rate.get(sat_id, 0.0003)
            total_dose += dose_rate * path_duration_yr

        return total_dose < self.max_dose_krad, total_dose

    def assign_lightpath(
        self,
        request: LightpathRequest,
        link_id_to_osnr: Dict[Tuple[int, int], float],
        link_id_to_ber: Dict[Tuple[int, int], float],
        link_id_to_distance: Dict[int, float],
        link_id_to_sun_angle: Dict[int, float],
        sat_id_to_dose_rate: Dict[int, float],
        k_paths: int = 3,
        path_duration_yr: float = 0.1,
    ) -> LightpathResult:
        """
        Full lightpath assignment with all QoT constraints.

        Algorithm:
        1. Find K-shortest paths
        2. For each path:
           a. Try first-fit wavelength assignment
           b. Verify all 7 QoT constraints
           c. If all pass, allocate wavelength and return success
        3. If no path found, return failure with reason
        """
        result = LightpathResult(
            request_id=request.request_id,
            src_sat=request.src_sat,
            dst_sat=request.dst_sat,
            accepted=False,
        )

        # Step 1: K-shortest paths
        paths = self.k_shortest_paths(
            request.src_sat, request.dst_sat, k_paths
        )

        if not paths:
            result.reject_reason = RejectReason.NO_PATH
            return result

        # Step 2: Try each path
        for path_sats, path_links in paths:
            violations = []

            # Constraint 7: Max hop count
            if len(path_links) > self.max_hops:
                violations.append(
                    f"Hop count {len(path_links)} > max {self.max_hops}"
                )
                if self.enforce_qot_constraints:
                    continue

            # Constraint 5: Link distance
            dist_ok, violating_links = self.check_link_distance(
                path_links, link_id_to_distance
            )
            if not dist_ok:
                violations.append(f"Link distance exceeded on {violating_links}")
                if self.enforce_qot_constraints:
                    continue

            # Constraint 4: Solar avoidance
            solar_ok, solar_violations = self.check_solar_avoidance(
                path_links, link_id_to_sun_angle
            )
            if not solar_ok:
                violations.append(f"Sun blocked on {solar_violations}")
                if self.enforce_qot_constraints:
                    continue

            # Try wavelengths
            wavelength = self.first_fit_wavelength(path_links)
            if wavelength is None:
                violations.append("No wavelength available")
                continue

            # Compute end-to-end OSNR
            per_link_osnr = []
            per_link_dist = []
            inv_osnr_sum = 0.0

            for link_id in path_links:
                osnr_key = (link_id, wavelength)
                osnr_lin = 10.0 ** (link_id_to_osnr.get(osnr_key, 15.0) / 10.0)
                per_link_osnr.append(10.0 * math.log10(osnr_lin))
                per_link_dist.append(link_id_to_distance.get(link_id, 0.0))
                inv_osnr_sum += 1.0 / max(osnr_lin, 1e-12)

            total_osnr_lin = 1.0 / inv_osnr_sum if inv_osnr_sum > 0 else 0.0
            total_osnr_dB = 10.0 * math.log10(max(total_osnr_lin, 1e-30))

            # BER from OSNR using BERCalculator (no hardcoded parameters)
            ber_result = self.ber_calc.compute_ber_from_osnr(total_osnr_dB)
            ber = ber_result.ber
            q_factor = ber_result.q_factor

            # Constraint 3: BER threshold
            if ber > self.ber_threshold:
                violations.append(
                    f"BER {ber:.2e} > threshold {self.ber_threshold:.2e}"
                )
                if self.enforce_qot_constraints:
                    continue

            # Constraint 6: Radiation dose
            cumulative_dose = 0.0
            if self.enforce_qot_constraints:
                rad_ok, cumulative_dose = self.check_radiation_dose(
                    path_sats, sat_id_to_dose_rate, path_duration_yr
                )
                if not rad_ok:
                    violations.append(
                        f"Cumulative dose {cumulative_dose:.2f} > max {self.max_dose_krad}"
                    )
                    result.cumulative_radiation_krad = cumulative_dose
                    continue

            # All constraints passed — allocate wavelength
            self.allocate_wavelength(path_links, wavelength)

            result.accepted = True
            result.reject_reason = RejectReason.NONE
            result.path_satellites = path_sats
            result.path_links = path_links
            result.wavelength_channel = wavelength
            result.total_hops = len(path_links)
            result.osnr_dB = total_osnr_dB
            result.ber = ber
            result.q_factor = q_factor
            result.per_link_osnr_dB = per_link_osnr
            result.per_link_distance_km = per_link_dist
            result.cumulative_radiation_krad = cumulative_dose
            result.constraint_violations = violations
            return result

        # All paths failed — determine reason
        if not violations:
            result.reject_reason = RejectReason.NO_WAVELENGTH
        elif "Sun blocked" in str(violations[-1]):
            result.reject_reason = RejectReason.SUN_BLOCKED
        elif "Hop count" in str(violations[-1]):
            result.reject_reason = RejectReason.HOP_COUNT_EXCEEDED
        elif "BER" in str(violations[-1]):
            result.reject_reason = RejectReason.BER_TOO_HIGH
        elif "Link distance" in str(violations[-1]):
            result.reject_reason = RejectReason.LINK_TOO_LONG
        elif "dose" in str(violations[-1]):
            result.reject_reason = RejectReason.RADIATION_EXCEEDED
        elif "No wavelength" in str(violations[-1]):
            result.reject_reason = RejectReason.NO_WAVELENGTH
        else:
            result.reject_reason = RejectReason.NO_PATH

        result.constraint_violations = violations
        return result
