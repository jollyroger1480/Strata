"""Strata's small, dependency-free terminal entry point.

Run ``python strata.py --help`` from a source checkout. Optional terminal UI
dependencies are imported only by ``monitor``.
"""

from strata_cli.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
