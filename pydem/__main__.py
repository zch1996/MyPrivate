#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyDEM Main Entry Point
----------------------
This is the main entry point for the PyDEM package.
When run as `pydem` or `python -m pydem`, it starts the UI controller
and allows running simulation scripts with visualization.
"""

import os
import sys
import importlib.util
import argparse
import logging
import threading
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PyDEM.Main")

# Import PyDEM core
from pydem import Omega, O


def main():
    """Main entry point for PyDEM."""
    parser = argparse.ArgumentParser(description="PyDEM - Python DEM Simulation")
    parser.add_argument(
        "--headless", action="store_true", help="Run without UI or visualization"
    )


if __name__ == "__main__":
    sys.exit(main())
