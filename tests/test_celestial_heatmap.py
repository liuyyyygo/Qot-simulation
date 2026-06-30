import tempfile
import unittest
from pathlib import Path

from plotting import (
    fig18_celestial_osnr_penalty_heatmap,
    fig19_solar_background_osnr_time_series,
)
from qot_simulation.simulation import QoTSimulation, SimulationConfig


class CelestialHeatmapFigureTest(unittest.TestCase):
    def test_generates_inter_orbit_celestial_osnr_heatmap(self):
        sim = QoTSimulation(
            SimulationConfig(
                N_planes=6,
                sats_per_plane=11,
                F=1,
                num_channels=4,
            )
        )
        sim.compute_all_impairments()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig18_celestial_osnr_penalty_heatmap(sim, tmpdir)
            output = Path(tmpdir) / "fig18_celestial_osnr_penalty_heatmap.svg"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)

    def test_generates_solar_background_osnr_timeseries(self):
        sim = QoTSimulation(
            SimulationConfig(
                N_planes=6,
                sats_per_plane=11,
                F=1,
                num_channels=4,
            )
        )
        sim.compute_all_impairments()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig19_solar_background_osnr_time_series(
                sim,
                tmpdir,
                duration_s=600.0,
                step_s=60.0,
                n_links=2,
            )
            output = Path(tmpdir) / "fig19_solar_background_osnr_timeseries.svg"
            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
