"""
Top-level entry point for QoT simulation.

Usage:
  python run.py                          # Run small demo
  python run.py --leo                    # Run with 12x11 constellation
  python run.py --starlink               # Run with Starlink-scale (72x22)
  python run.py --config config.yaml     # Run with YAML config file
  python run.py --demo --num-requests 20 # Run demo with 20 requests
  python run.py --figures                # Generate all paper figures
"""

import sys

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="QoT performance simulation for traditional RWA in LEO SONs"
    )
    parser.add_argument("--figures", action="store_true",
                        help="Run multi-load simulation and generate paper figures")
    parser.add_argument("--demo", action="store_true",
                        help="Run demonstration with sample requests")
    parser.add_argument("--leo", action="store_true",
                        help="Use representative LEO constellation (12x11)")
    parser.add_argument("--starlink", action="store_true",
                        help="Use Starlink-scale constellation (72x22)")
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--detail", type=int, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--outdir", type=str, default="./figures")
    parser.add_argument("--low", type=int, default=10)
    parser.add_argument("--medium", type=int, default=50)
    parser.add_argument("--high", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--observe-duration", type=float, default=None)
    parser.add_argument("--observe-step", type=float, default=None)

    args, unknown = parser.parse_known_args()

    if args.figures:
        from run_figures import main as figures_main
        # Pass through relevant args
        sys.argv = [sys.argv[0]]
        if args.starlink:
            sys.argv.append("--starlink")
        sys.argv.extend(["--outdir", args.outdir])
        sys.argv.extend(["--low", str(args.low)])
        sys.argv.extend(["--medium", str(args.medium)])
        sys.argv.extend(["--high", str(args.high)])
        sys.argv.extend(["--seed", str(args.seed)])
        figures_main()
    else:
        from qot_simulation.simulation import main as sim_main
        sim_main()
