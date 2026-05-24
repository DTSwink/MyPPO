"""Entry point for the 2D kinematic walk visualizer."""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 2D kinematic visualizer.")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable auto-restart on code changes.",
    )
    args = parser.parse_args()

    if args.no_reload:
        from kinematic2d.visualizer import main

        main()
    else:
        from kinematic2d.dev_reload import run_with_reload

        run_with_reload()
