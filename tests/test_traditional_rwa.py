import math
import unittest
from types import SimpleNamespace

from qot_simulation.rwa import LightpathRequest, RejectReason, RWASolver


class TraditionalRWATest(unittest.TestCase):
    def make_solver(self, **kwargs):
        link = SimpleNamespace(link_id=0, sat_i=0, sat_j=1)
        return RWASolver(
            num_satellites=2,
            adjacency={0: [1], 1: [0]},
            link_distances_km={(0, 1): 1000.0},
            topology_links=[link],
            num_wavelengths=1,
            ber_threshold=1e-12,
            **kwargs,
        )

    def test_traditional_rwa_accepts_low_qot_path_and_reports_performance(self):
        solver = self.make_solver(enforce_qot_constraints=False)

        result = solver.assign_lightpath(
            request=LightpathRequest(request_id=1, src_sat=0, dst_sat=1),
            link_id_to_osnr={(0, 0): -20.0},
            link_id_to_ber={},
            link_id_to_distance={0: 1000.0},
            link_id_to_sun_angle={0: 180.0},
            sat_id_to_dose_rate={0: 1e6, 1: 1e6},
            path_duration_yr=10.0,
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.reject_reason, RejectReason.NONE)
        self.assertEqual(result.wavelength_channel, 0)
        self.assertGreater(result.ber, solver.ber_threshold)
        self.assertTrue(math.isfinite(result.osnr_dB))
        self.assertFalse(
            any("Cumulative dose" in v for v in result.constraint_violations)
        )

    def test_traditional_rwa_still_rejects_when_no_wavelength_is_available(self):
        solver = self.make_solver(enforce_qot_constraints=False)
        solver.allocate_wavelength([0], 0)

        result = solver.assign_lightpath(
            request=LightpathRequest(request_id=1, src_sat=0, dst_sat=1),
            link_id_to_osnr={(0, 0): 30.0},
            link_id_to_ber={},
            link_id_to_distance={0: 1000.0},
            link_id_to_sun_angle={0: 180.0},
            sat_id_to_dose_rate={0: 0.0, 1: 0.0},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reject_reason, RejectReason.NO_WAVELENGTH)


if __name__ == "__main__":
    unittest.main()
