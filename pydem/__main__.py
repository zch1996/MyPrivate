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
import multiprocessing
import threading
import time
import signal
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PyDEM.Main")

# Import PyDEM core
from pydem import Omega, O


def load_script(script_path):
    """Load and execute a Python script as a module."""
    try:
        logger.info(f"Loading script: {script_path}")

        # Get absolute path
        script_path = os.path.abspath(script_path)

        # Extract directory and filename
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)
        module_name = os.path.splitext(script_name)[0]

        # Add script directory to path if not already there
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Load module
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        if spec is None:
            logger.error(f"Could not find script: {script_path}")
            return False

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        logger.info(f"Script loaded successfully: {script_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading script: {str(e)}")
        return False


def simulation_process(script_path, cmd_queue, status_queue, headless=False):
    """Process that runs the simulation core."""
    try:
        logger.info("Simulation process started")

        # Initialize the simulation
        omega = Omega.instance()
        scene = omega.getScene()

        # Load the script if provided
        if script_path:
            success = load_script(script_path)
            if not success:
                status_queue.put(("ERROR", "Failed to load script"))
                return

        # Main loop for the simulation process
        running = True
        while running:
            try:
                # Check for commands
                try:
                    cmd, args = cmd_queue.get(block=False)

                    if cmd == "EXIT":
                        running = False
                    elif cmd == "START":
                        scene.run(wait=False)
                    elif cmd == "STOP":
                        scene.stop()
                    elif cmd == "STEP":
                        scene.oneStep()
                    elif cmd == "SET_PARAM":
                        # Set a parameter in the simulation
                        param_name, param_value = args
                        # Implementation depends on what parameters are supported
                        pass
                    elif cmd == "GET_STATS":
                        # Collect statistics and send back
                        stats = {
                            "step": scene.step,
                            "time": scene.time,
                            "particles": len(scene.field.getParticles()),
                            "contacts": scene.field.getContacts().size(),
                            "running": scene.isRunning(),
                        }
                        status_queue.put(("STATS", stats))
                    else:
                        logger.warning(f"Unknown command: {cmd}")

                except queue.Empty:
                    pass

                # Send periodic status updates
                if scene.isRunning():
                    stats = {
                        "step": scene.step,
                        "time": scene.time,
                        "particles": len(scene.field.getParticles()),
                        "contacts": scene.field.getContacts().size(),
                    }
                    status_queue.put(("STATUS", stats))

                # Sleep to avoid using too much CPU
                time.sleep(0.01)

            except Exception as e:
                logger.error(f"Error in simulation process: {str(e)}")
                status_queue.put(("ERROR", str(e)))

        logger.info("Simulation process shutting down")

    except Exception as e:
        logger.error(f"Fatal error in simulation process: {str(e)}")
        status_queue.put(("FATAL", str(e)))


def visualization_process(cmd_queue, data_queue, headless=False):
    """Process that handles the OpenGL visualization."""
    if headless:
        logger.info("Visualization disabled in headless mode")
        return

    try:
        logger.info("Visualization process started")

        # Import visualization modules here to keep them isolated
        from pydem.visualization import VisualizationEngine

        # Create visualization engine
        vis_engine = VisualizationEngine()

        # Main loop for visualization
        running = True
        while running:
            try:
                # Check for commands
                try:
                    cmd, args = cmd_queue.get(block=False)

                    if cmd == "EXIT":
                        running = False
                    elif cmd == "CAMERA":
                        # Update camera position/orientation
                        vis_engine.set_camera(*args)
                    elif cmd == "RENDER_MODE":
                        # Change rendering mode
                        vis_engine.set_render_mode(args)
                    else:
                        logger.warning(f"Unknown visualization command: {cmd}")

                except queue.Empty:
                    pass

                # Check for new scene data
                try:
                    data_type, data = data_queue.get(block=False)

                    if data_type == "SCENE_DATA":
                        vis_engine.update_scene(data)
                    elif data_type == "PARTICLES":
                        vis_engine.update_particles(data)
                    elif data_type == "CONTACTS":
                        vis_engine.update_contacts(data)

                except queue.Empty:
                    pass

                # Render frame
                vis_engine.render()

                # Sleep to maintain frame rate
                time.sleep(1 / 60)  # Target 60 FPS

            except Exception as e:
                logger.error(f"Error in visualization process: {str(e)}")

        # Clean up visualization resources
        vis_engine.cleanup()
        logger.info("Visualization process shutting down")

    except Exception as e:
        logger.error(f"Fatal error in visualization process: {str(e)}")


def ui_process(
    sim_cmd_queue, sim_status_queue, vis_cmd_queue, vis_data_queue, headless=False
):
    """Process that handles the user interface."""
    if headless:
        logger.info("UI disabled in headless mode")
        return

    try:
        logger.info("UI process started")

        # Import UI modules here to keep them isolated
        from pydem.ui import UIController

        # Create UI controller
        ui_controller = UIController(
            sim_cmd_queue=sim_cmd_queue,
            sim_status_queue=sim_status_queue,
            vis_cmd_queue=vis_cmd_queue,
            vis_data_queue=vis_data_queue,
        )

        # Start the UI main loop (this will block until UI is closed)
        ui_controller.run()

        # Signal other processes to exit
        sim_cmd_queue.put(("EXIT", None))
        vis_cmd_queue.put(("EXIT", None))

        logger.info("UI process shutting down")

    except Exception as e:
        logger.error(f"Fatal error in UI process: {str(e)}")
        # Signal other processes to exit
        sim_cmd_queue.put(("EXIT", None))
        vis_cmd_queue.put(("EXIT", None))


def start_ipython_shell(script_path=None):
    """Start an IPython shell with PyDEM pre-loaded."""
    try:
        from IPython import embed

        # Initialize PyDEM
        omega = Omega.instance()
        scene = omega.getScene()

        # Load script if provided
        if script_path:
            load_script(script_path)

        # Create a banner for the IPython shell
        banner = """
===========================================================
PyDEM Interactive Shell
===========================================================
Available objects:
  - omega: The Omega singleton instance
  - scene: The current simulation scene
  - O: Shorthand for Omega.instance()

Example commands:
  - scene.run(100)     # Run simulation for 100 steps
  - scene.stop()       # Stop the simulation
  - scene.oneStep()    # Execute a single step
===========================================================
"""

        # Start IPython shell
        embed(banner1=banner)

    except ImportError:
        logger.error("IPython is not installed. Cannot start interactive shell.")
        print("Please install IPython: pip install ipython")


def main():
    """Main entry point for PyDEM."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="PyDEM - Python DEM Simulation")
    parser.add_argument("script", nargs="?", help="Python script to execute")
    parser.add_argument(
        "--headless", action="store_true", help="Run without UI or visualization"
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive IPython shell"
    )
    parser.add_argument(
        "--cores", type=int, default=None, help="Number of CPU cores to use"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level",
    )
    args = parser.parse_args()

    # Configure logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set the number of CPU cores to use
    if args.cores:
        multiprocessing.set_start_method("spawn")
        logger.info(f"Using {args.cores} CPU cores")
        os.environ["OMP_NUM_THREADS"] = str(args.cores)

    # Handle interactive mode
    if args.interactive:
        start_ipython_shell(args.script)
        return 0

    # Handle headless mode with script execution
    if args.headless:
        if args.script:
            # Just run the simulation in the current process
            omega = Omega.instance()
            scene = omega.getScene()

            if load_script(args.script):
                # Run the simulation
                scene.run(wait=True)
                return 0
            else:
                return 1
        else:
            logger.info("No script provided in headless mode")
            start_ipython_shell()
            return 1

    # Setup multiprocessing queues for inter-process communication
    sim_cmd_queue = multiprocessing.Queue()  # Commands to simulation
    sim_status_queue = multiprocessing.Queue()  # Status messages from simulation
    vis_cmd_queue = multiprocessing.Queue()  # Commands to visualization
    vis_data_queue = multiprocessing.Queue()  # Data for visualization

    # Create processes
    sim_process = multiprocessing.Process(
        target=simulation_process,
        args=(args.script, sim_cmd_queue, sim_status_queue, args.headless),
    )

    vis_process = None
    ui_process_obj = None

    if not args.headless:
        vis_process = multiprocessing.Process(
            target=visualization_process,
            args=(vis_cmd_queue, vis_data_queue, args.headless),
        )

        ui_process_obj = multiprocessing.Process(
            target=ui_process,
            args=(
                sim_cmd_queue,
                sim_status_queue,
                vis_cmd_queue,
                vis_data_queue,
                args.headless,
            ),
        )

    # Start processes
    sim_process.start()
    if vis_process:
        vis_process.start()
    if ui_process_obj:
        ui_process_obj.start()

    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        sim_cmd_queue.put(("EXIT", None))
        if vis_process:
            vis_cmd_queue.put(("EXIT", None))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # If headless without UI, we need to forward status messages
    if args.headless and args.script:
        try:
            while sim_process.is_alive():
                try:
                    msg_type, msg = sim_status_queue.get(timeout=1)
                    if msg_type == "STATUS":
                        logger.info(
                            f"Simulation status: Step={msg['step']}, Time={msg['time']}"
                        )
                    elif msg_type == "ERROR" or msg_type == "FATAL":
                        logger.error(f"Simulation error: {msg}")
                except queue.Empty:
                    pass
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sim_cmd_queue.put(("EXIT", None))

    # Wait for processes to finish
    sim_process.join()
    if vis_process:
        vis_process.join()
    if ui_process_obj:
        ui_process_obj.join()

    logger.info("PyDEM successfully shut down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
