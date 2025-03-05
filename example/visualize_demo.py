#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PyDEM Multiprocess Demo
----------------------
This script demonstrates the PyDEM visualization system with multiprocessing,
allowing separate processes for UI, visualization, and interactive console.
"""

import sys
import os
import time
import math
import random
import numpy as np
import logging
import threading
import multiprocessing
from multiprocessing import Process, Queue, Value

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PyDEM.Demo")

# Suppress pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Queue objects for communication between processes
cmd_queue = Queue()  # Commands from main process to workers
ui_queue = Queue()  # Status updates from UI process
viz_queue = Queue()  # Status updates from visualization process

# Flag to track if processes are running
ui_running = Value("i", 0)
viz_running = Value("i", 0)


def create_demo_scene():
    """Create a demo scene with various particles."""
    # Import PyDEM core modules
    from pydem import O, Scene, DEMField
    from pydem.src import utils
    from pydem.src.Material import FrictMat

    logger.info("Creating demo scene...")

    # Create a new scene if one doesn't exist
    if O.scene is None:
        O.scene = Scene()

    # Create a DEM field if one doesn't exist
    if not hasattr(O.scene, "field") or O.scene.field is None:
        field = DEMField()
        O.scene.setField(field)

    dem = O.scene.field

    # Set gravity
    dem.setGravity([0, -9.81, 0])

    # Set up simulation parameters
    O.scene.dt = 1e-4
    O.scene.dtSafety = 0.9

    # Set up default engines
    engines = utils.defaultEngines(damping=0.3, verletDist=0.01)
    for engine in engines:
        O.scene.addEngine(engine)

    # Create default material
    defaultMaterial = FrictMat()
    defaultMaterial.density = 2600  # kg/m³
    defaultMaterial.young = 5e7  # Pa
    defaultMaterial.poisson = 0.3  # -
    defaultMaterial.frictionAngle = math.radians(30)  # 30 degrees in radians

    # Add a ground plane using a large sphere
    ground = utils.sphere(
        center=[0, -50, 0],
        radius=50,
        mat=defaultMaterial,
        fixed=True,
        color=0.3,
    )
    O.scene.addParticle(ground)

    # Add some spheres
    add_demo_spheres(O.scene, 20, defaultMaterial)

    # Enable energy tracking
    O.scene.trackEnergy = True

    logger.info(f"Demo scene created")
    return O.scene


def add_demo_spheres(scene, count, defaultMaterial):
    """Add some spheres to the scene."""
    from pydem.src import utils

    for i in range(count):
        # Random radius between 0.1 and 0.3
        radius = random.uniform(0.1, 0.3)

        # Random position above the ground
        x = random.uniform(-4, 4)
        y = random.uniform(5, 10)
        z = random.uniform(-4, 4)

        # Random color (0.0 to 1.0)
        color = random.random()

        sp = utils.sphere(
            center=[x, y, z],
            radius=radius,
            mat=defaultMaterial,
            vel=[0, 0, 0],
            color=color,
        )

        # Add to scene
        scene.addParticle(sp)


def ui_process_target():
    """Function to run in the UI process."""
    try:
        # Import UI components
        from pydem.visualization import start_control_ui
        from PyQt5 import QtWidgets

        # Create Omega instance for this process
        from pydem import O

        # Create Qt application
        app = QtWidgets.QApplication(sys.argv)

        # Add debug output
        ui_queue.put(("debug", "Creating UI window..."))

        # Create UI window
        try:
            window = start_control_ui(O)
            if not window:
                ui_queue.put(("error", "Failed to start UI controller"))
                return
        except Exception as e:
            ui_queue.put(("error", f"Exception creating UI window: {str(e)}"))
            import traceback

            traceback.print_exc()
            return

        # Signal that UI is now running
        ui_running.value = 1
        ui_queue.put(("started", "UI started successfully"))

        # Setup command checker timer
        def check_commands():
            try:
                if not cmd_queue.empty():
                    cmd, args = cmd_queue.get(block=False)

                    if cmd == "quit":
                        app.quit()
                    elif cmd == "update":
                        # Force UI stats update
                        window.update_simulation_stats()
                    elif cmd == "add_spheres":
                        # Add more spheres to the scene
                        count = args if isinstance(args, int) else 5
                        add_random_spheres(count)
                        window.update_simulation_stats()
                        cmd_queue.put(
                            ("update_viz", None)
                        )  # Notify visualization to update
            except Exception as e:
                logger.error(f"Error processing command in UI: {e}")

            # Schedule next check
            if window and hasattr(window, "update_timer"):
                window.update_timer.singleShot(100, check_commands)

        # Start command checking
        if window and hasattr(window, "update_timer"):
            window.update_timer.singleShot(100, check_commands)

        # Start Qt event loop
        app_result = app.exec_()

        # Signal that UI has stopped
        ui_running.value = 0
        ui_queue.put(("stopped", "UI stopped"))

    except Exception as e:
        logger.error(f"Error in UI process: {e}")
        import traceback

        traceback.print_exc()
        ui_queue.put(("error", f"UI process error: {str(e)}"))
        ui_running.value = 0


def viz_process_target():
    """Function to run in the visualization process."""
    try:
        # Suppress pygame welcome message
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

        # Import visualization components
        from pydem.visualization import start_visualization

        # Create Omega instance for this process
        from pydem import O

        # Start visualization
        viz = start_visualization(O)
        if not viz:
            viz_queue.put(("error", "Failed to start visualization"))
            return

        # Signal that visualization is now running
        viz_running.value = 1
        viz_queue.put(("started", "Visualization started successfully"))

        # Main loop - keep visualization running and check for commands
        while viz.running:
            try:
                if not cmd_queue.empty():
                    cmd, args = cmd_queue.get(block=False)

                    if cmd == "quit":
                        viz.stop()
                        break
                    elif cmd == "update_viz":
                        # Force scene data update
                        viz.command_queue.put(("update", None))
                # Short sleep to prevent high CPU usage
                time.sleep(0.05)
            except Exception as e:
                logger.error(f"Error processing command in visualization: {e}")

        # Signal that visualization has stopped
        viz_running.value = 0
        viz_queue.put(("stopped", "Visualization stopped"))

    except Exception as e:
        logger.error(f"Error in visualization process: {e}")
        import traceback

        traceback.print_exc()
        viz_queue.put(("error", f"Visualization process error: {str(e)}"))
        viz_running.value = 0


def add_random_spheres(count=5):
    """Add random spheres to the current scene."""
    from pydem import O
    from pydem.src import utils
    from pydem.src.Material import FrictMat

    # Get the current scene
    scene = O.scene
    if not scene:
        logger.error("No scene available")
        return False

    # Create material
    defaultMaterial = FrictMat()
    defaultMaterial.density = 2600  # kg/m³
    defaultMaterial.young = 5e7  # Pa
    defaultMaterial.poisson = 0.3  # -
    defaultMaterial.frictionAngle = math.radians(30)  # 30 degrees in radians

    # Add spheres
    add_demo_spheres(scene, count, defaultMaterial)

    logger.info(f"Added {count} new spheres to the scene")
    return True


def run_simulation(steps=None):
    """Run the simulation for a specified number of steps."""
    from pydem import O

    scene = O.scene
    if not scene:
        logger.error("No scene available")
        return False

    if steps is None:
        # Run indefinitely
        scene.run(wait=False)
    else:
        # Run for specified steps
        scene.run(steps=steps, wait=False)

    logger.info(f"Started simulation for {steps if steps else 'indefinite'} steps")
    return True


def stop_simulation():
    """Stop the currently running simulation."""
    from pydem import O

    scene = O.scene
    if not scene:
        logger.error("No scene available")
        return False

    scene.stop()
    logger.info("Stopped simulation")
    return True


def print_help():
    """Print help for interactive console commands."""
    print("\nPyDEM Multiprocess Demo - Available Commands:")
    print("---------------------------------------------")
    print("add(n)      - Add n spheres to the scene (default: 5)")
    print("run(n)      - Run simulation for n steps (default: indefinitely)")
    print("stop()      - Stop the simulation")
    print("update()    - Force update of UI and visualization")
    print("quit()      - Exit the demo")
    print("help()      - Show this help message")
    print()


def main():
    """Main function to start the demo."""
    # Make sure multiprocessing works correctly on all platforms
    multiprocessing.set_start_method("spawn", force=True)

    # Create the demo scene
    create_demo_scene()

    # Start UI process
    ui_process = Process(target=ui_process_target)
    ui_process.daemon = True
    ui_process.start()

    # Start visualization process
    viz_process = Process(target=viz_process_target)
    viz_process.daemon = True
    viz_process.start()

    # Don't wait for UI to start - just continue with demo
    print("\nPyDEM Multiprocess Demo")
    print("======================")
    print("UI and visualization are starting in separate processes.")
    print("You can interact with the simulation using the functions below.")
    print_help()

    # Define functions for interactive console
    def add(count=5):
        """Add spheres to the scene."""
        add_random_spheres(count)
        cmd_queue.put(("update", None))  # Update UI
        cmd_queue.put(("update_viz", None))  # Update visualization

    def run(steps=None):
        """Run simulation."""
        run_simulation(steps)

    def stop():
        """Stop simulation."""
        stop_simulation()

    def update():
        """Force UI and visualization update."""
        cmd_queue.put(("update", None))
        cmd_queue.put(("update_viz", None))

    def quit():
        """Exit the demo."""
        cmd_queue.put(("quit", None))
        time.sleep(0.5)  # Give processes time to receive the quit command
        sys.exit(0)

    def help():
        """Show help."""
        print_help()

    # Start interactive console
    try:
        # Try to use IPython if available
        import IPython

        IPython.embed(
            header="",
            user_ns={
                "add": add,
                "run": run,
                "stop": stop,
                "update": update,
                "quit": quit,
                "help": help,
            },
        )
    except ImportError:
        # Fall back to simple interactive console
        import code

        code.interact(
            banner="",
            local={
                "add": add,
                "run": run,
                "stop": stop,
                "update": update,
                "quit": quit,
                "help": help,
            },
        )

    # Clean up before exiting
    cmd_queue.put(("quit", None))

    # Wait briefly for processes to clean up
    time.sleep(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
