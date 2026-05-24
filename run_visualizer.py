"""Entry point for the 2D kinematic walk visualizer."""

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the 2D kinematic visualizer.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Auto-restart the visualizer when Python source files change.",
    )
    args = parser.parse_args()

    if args.reload:
        from kinematic2d.dev_reload import run_with_reload

        run_with_reload()
    else:
        from kinematic2d.visualizer import main

        main()
