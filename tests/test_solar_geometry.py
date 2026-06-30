import unittest

import numpy as np

from qot_simulation.impairments import CelestialBackground
from qot_simulation.simulation import QoTSimulation, SimulationConfig


class SolarGeometryTest(unittest.TestCase):
    def test_global_sun_angle_uses_ecef_link_direction(self):
        celestial = CelestialBackground()
        sun_vec = celestial.compute_sun_vector_ecef(
            sun_dec_deg=0.0,
            sun_ra_deg=0.0,
            time_hours=12.0,
        )

        self.assertAlmostEqual(
            celestial.compute_link_sun_angle_deg(sun_vec, sun_vec),
            0.0,
            places=6,
        )
        self.assertAlmostEqual(
            celestial.compute_link_sun_angle_deg(-sun_vec, sun_vec),
            180.0,
            places=6,
        )

    def test_bidirectional_link_sun_angle_takes_nearest_receiver_view(self):
        celestial = CelestialBackground()
        sun_vec = np.array([1.0, 0.0, 0.0])
        link_vec = np.array([-1.0, 0.0, 0.0])

        self.assertAlmostEqual(
            celestial.compute_bidirectional_link_sun_angle_deg(link_vec, sun_vec),
            0.0,
            places=6,
        )

    def test_solar_noise_has_affected_core_and_continuous_tail(self):
        celestial = CelestialBackground(
            sun_avoidance_angle_deg=3.0,
            sun_coupling_scale_deg=5.0,
        )

        noise_inside, blocked_inside = celestial.compute_noise_power_W(1.0)
        noise_tail, blocked_tail = celestial.compute_noise_power_W(6.0)
        noise_far, blocked_far = celestial.compute_noise_power_W(30.0)

        self.assertTrue(blocked_inside)
        self.assertFalse(blocked_tail)
        self.assertFalse(blocked_far)
        self.assertGreater(noise_inside, noise_tail)
        self.assertGreater(noise_tail, noise_far)

    def test_celestial_filter_bandwidth_follows_optical_filter_config(self):
        sim = QoTSimulation(
            SimulationConfig(
                optical_filter_BW_GHz=12.5,
                rx_aperture_diameter_mm=80.0,
            )
        )

        self.assertAlmostEqual(
            sim.celestial.filter_BW_um,
            0.10017e-3,
            delta=0.001e-3,
        )


if __name__ == "__main__":
    unittest.main()
