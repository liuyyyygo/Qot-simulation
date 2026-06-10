# QoT Simulation for LEO Satellite Optical Networks

Quality of Transmission (QoT) simulation for LEO satellite optical networks under
WDM-based optical circuit switching (OXC) architecture.

Simulates establishment of transparent optical lightpaths through a Walker-Delta
constellation, computing end-to-end OSNR and BER under 5 physical impairments
and 7 QoT constraints.

## Quick Start

```bash
# Run small demo (10 requests on 12x11=132 satellite constellation)
python run.py

# Run demo with 20 requests
python run.py --demo --num-requests 20

# Generate all paper-quality figures (low/medium/high traffic)
python run.py --figures --outdir ./figures
```

## Architecture

```
Qot-simulation-6-10/
├── qot_simulation/          # Core simulation package
│   ├── constellation.py     # Walker-Delta constellation, LISL topology
│   ├── impairments.py       # 5 physical layer impairment models
│   ├── osnr_ber.py          # OSNR accumulation + BER estimation
│   ├── rwa.py               # RWA solver (K-SP + First-Fit, 7 QoT checks)
│   └── simulation.py        # Top-level simulation engine + config
├── plotting.py              # 16 paper-quality figures (Nature journal style)
├── run_figures.py           # Multi-load simulation runner
├── run.py                   # CLI entry point
├── config.yaml              # Example YAML configuration
└── README.md
```

## Impairment Models

| Impairment | Model |
|---|---|
| Free Space Loss | Friis transmission equation |
| Doppler Shift & Filter Penalty | Δλ computation + Gaussian filter response |
| Celestial Background | Solar/lunar/sky radiance → shot noise |
| WDM Crosstalk | Inter- and intra-wavelength crosstalk (OXC + demux) |
| SAA Radiation | Position-dependent EDFA gain/NF degradation |

## QoT Constraints

1. Wavelength continuity (transparent path)
2. Wavelength clash (distinct λ per LISL)
3. BER/FEC threshold (HD-FEC: BER < 2×10⁻³)
4. Solar avoidance angle
5. Link distance upper bound (Earth occlusion)
6. Cumulative radiation dose (total ionizing dose)
7. Maximum hop count

## CLI Reference

```
python run.py [options]

Options:
  --demo                Run demonstration with sample requests
  --figures             Run multi-load simulation and generate paper figures
  --leo                 Use representative LEO constellation (12×11)
  --starlink            Use Starlink-scale constellation (72×22)
  --num-requests N      Number of traffic requests (default: 10)
  --config PATH         YAML configuration file path
  --outdir PATH         Output directory for figures (default: ./figures)
  --low N               Low-load request count (default: 10)
  --medium N            Medium-load request count (default: 50)
  --high N              High-load request count (default: 200)
  --seed N              Random seed (default: 42)
  --detail ID           Print detailed info for a specific request ID
```

## Dependencies

- Python 3.8+
- numpy
- matplotlib

## References

See `qot_simulation/references.md` for a comprehensive academic reference list
covering all impairment models, constellation theory, and RWA algorithms.
