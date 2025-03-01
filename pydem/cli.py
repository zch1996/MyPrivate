#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse
import threading
import logging
from IPython import embed
import importlib.util

# Import PyDEM core modules
import pydem
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.FunctorFactory import FunctorFactory


"""
- setup_environment
- setup_logging
- run_script
- start_interactive
- main

"""


def setup_logging(log_level):
    """Set up logging system"""
    # Create a root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))

    # Clear handlers
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Create a console handler and set its level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        "[%(asctime)s] - [%(name)s] - [%(levelname)s] - [%(message)s]"
    )

    root_logger.addHandler(console_handler)

    # Set 3rd-party logging level to root
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numpy").setLevel(logging.WARNING)

    return root_logger


def setup_environment():
    """Set up PyDEM environment."""
    # Set maximum thread count
    max_threads = os.cpu_count() or 4
    threading.current_thread().name = "PyDEM-Main"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("PyDEM")
    logger.info(f"Initializing PyDEM environment, max threads: {max_threads}")

    # Get factory instance (functors should already be registered from __init__.py)
    factory = FunctorFactory.instance()

    # Create environment dictionary
    env = {
        "max_threads": max_threads,
        "logger": logger,
        "factory": factory,
        "O": pydem.O,
        "Vector3r": pydem.Vector3r,
        "Matrix3r": pydem.Matrix3r,
        "Quaternionr": pydem.Quaternionr,
        "Scene": pydem.Scene,
        "DEMField": pydem.DEMField,
    }

    return env


def run_script(script_path, env):
    """Run the specified Python script."""
    logger = env["logger"]
    logger.info(f"Executing script: {script_path}")

    # Load and execute script
    try:
        spec = importlib.util.spec_from_file_location("pydem_script", script_path)
        module = importlib.util.module_from_spec(spec)

        # Inject PyDEM environment into module
        module.PYDEM_ENV = env

        spec.loader.exec_module(module)
        logger.info(f"Script execution completed: {script_path}")
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        raise


def start_interactive(env):
    """Start interactive IPython environment."""
    logger = env["logger"]
    logger.info("Starting interactive PyDEM environment")

    # Prepare IPython environment
    banner = """
    =====================================================
    PyDEM Interactive Environment
    -----------------------------------------------------
    PyDEM environment has been initialized.
    All PyDEM functionality is available.
    
    Environment variables:
    - PYDEM_ENV: Contains PyDEM environment configuration
    =====================================================
    """

    # Create IPython namespace
    namespace = {
        "PYDEM_ENV": env,
        # Import common modules into namespace
        "Vector3r": pydem.Vector3r,
        "Matrix3r": pydem.Matrix3r,
        "Quaternionr": pydem.Quaternionr,
        "Scene": pydem.Scene,
        "DEMField": pydem.DEMField,
        "O": pydem.O,
        "Sphere": pydem.Sphere,
        "Wall": pydem.Wall,
        "Facet": pydem.Facet,
        "Capsule": pydem.Capsule,
        "Cone": pydem.Cone,
        "Ellipsoid": pydem.Ellipsoid,
        "InfCylinder": pydem.InfCylinder,
        "Material": pydem.Material,
        "FrictMat": pydem.FrictMat,
    }

    # Start IPython
    embed(banner1=banner, user_ns=namespace)


def main():
    """Main function, handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="PyDEM - Discrete Element Method Framework"
    )
    parser.add_argument("script", nargs="?", help="Python script to execute")
    parser.add_argument("--threads", type=int, help="Set maximum thread count")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set log level",
    )

    args = parser.parse_args()

    # Set up environment
    env = setup_environment()

    # Apply command line arguments
    if args.threads:
        env["max_threads"] = args.threads
        from pydem.src.OpenMPSimulator import omp_set_num_threads

        omp_set_num_threads(args.threads)

    if args.log_level:
        logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Execute script or start interactive environment
    if args.script:
        run_script(args.script, env)
    else:
        start_interactive(env)


if __name__ == "__main__":
    main()
