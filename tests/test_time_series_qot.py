import unittest

from qot_simulation.simulation import (
    LightpathRequest,
    QoTSimulation,
    SimulationConfig,
)


class TimeSeriesQoTTest(unittest.TestCase):
    def test_evaluates_established_lightpaths_over_time_without_rerouting(self):
        config = SimulationConfig(
            N_planes=6,
            sats_per_plane=11,
            F=1,
            num_channels=4,
            sim_time_seconds=0.0,
        )
        sim = QoTSimulation(config)
        results = sim.run_requests(
            [LightpathRequest(request_id=1, src_sat=14, dst_sat=3)],
            verbose=False,
        )
        accepted = [r for r in results if r.accepted]
        self.assertEqual(len(accepted), 1)

        samples = sim.evaluate_established_lightpaths_over_time(
            accepted,
            duration_s=120.0,
            step_s=60.0,
        )

        self.assertEqual([s.time_seconds for s in samples], [0.0, 60.0, 120.0])
        self.assertTrue(all(s.request_id == 1 for s in samples))
        self.assertTrue(all(s.path_links == accepted[0].path_links for s in samples))
        self.assertTrue(all(s.wavelength_channel == accepted[0].wavelength_channel for s in samples))
        self.assertTrue(all(s.osnr_dB == s.osnr_dB for s in samples))


if __name__ == "__main__":
    unittest.main()
