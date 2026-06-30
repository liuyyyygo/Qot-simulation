import unittest

from qot_simulation.simulation import QoTSimulation, SimulationConfig


class MethodologyAlignmentTest(unittest.TestCase):
    def test_default_config_uses_current_dense_wdm_parameters(self):
        config = SimulationConfig()

        self.assertEqual(config.channel_spacing_GHz, 25.0)
        self.assertEqual(config.num_channels, 24)
        self.assertEqual(config.optical_filter_BW_GHz, 12.5)
        self.assertEqual(config.demux_isolation_dB, 25.0)
        self.assertEqual(config.oxc_isolation_dB, 30.0)

    def test_same_wavelength_oxc_interference_counts_adjacent_node_ports(self):
        config = SimulationConfig(
            N_planes=12,
            sats_per_plane=11,
            F=1,
            num_channels=4,
        )
        sim = QoTSimulation(config)
        sim.compute_all_impairments()

        target_link = sim.constellation.topology[0]
        incident_link = next(
            link
            for link in sim.constellation.topology
            if link.link_id != target_link.link_id
            and (
                link.sat_i in (target_link.sat_i, target_link.sat_j)
                or link.sat_j in (target_link.sat_i, target_link.sat_j)
            )
        )

        sim.rwa.wavelength_occupancy[incident_link.link_id].add(0)

        self.assertEqual(
            sim._count_same_wavelength_oxc_interferers(target_link, 0),
            1,
        )


if __name__ == "__main__":
    unittest.main()
