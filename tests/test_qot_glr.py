import unittest

from qot_simulation.simulation import (
    LightpathRequest,
    QoTSimulation,
    SimulationConfig,
)


class QoTGLRTest(unittest.TestCase):
    def test_qot_glr_establishes_lightpath_with_risk_score(self):
        config = SimulationConfig(
            N_planes=12,
            sats_per_plane=11,
            num_channels=8,
            routing_strategy="qot_glr",
            k_shortest_paths=3,
            qot_glr_risk_duration_s=600.0,
            qot_glr_risk_step_s=300.0,
            qot_glr_pruning_threshold=1.0,
        )
        sim = QoTSimulation(config)

        result = sim.run_requests(
            [LightpathRequest(request_id=1, src_sat=0, dst_sat=1)],
            verbose=False,
        )[0]

        self.assertTrue(result.accepted)
        self.assertGreaterEqual(result.wavelength_channel, 0)
        self.assertTrue(result.constraint_violations)
        self.assertIn("QoT-GLR score=", result.constraint_violations[0])


if __name__ == "__main__":
    unittest.main()
