# pydem/visualization/engine.py
"""
Visualization Engine
-------------------
Core component for 3D visualization of DEM simulations.
"""

import logging
import threading
import time
import numpy as np
import queue

from .renderer import Renderer
from .camera import Camera
from .scene_renderer import SceneRenderer

logger = logging.getLogger("PyDEM.Visualization")


class VisualizationEngine:
    """Main engine for visualization of DEM simulations."""

    def __init__(self, omega=None):
        """
        Initialize visualization engine.

        Args:
            omega: Omega instance, if None, will use global instance
        """
        # Get Omega instance if not provided
        if omega is None:
            from pydem import Omega

            self.omega = Omega.instance()
        else:
            self.omega = omega

        # Initialize components
        self.renderer = None
        self.camera = None
        self.scene_renderer = None

        # Threading and synchronization
        self.render_thread = None
        self.command_queue = queue.Queue()
        self.running = False

        # Scene data
        self.scene = self.omega.getScene()

        # Initialize OpenGL context and objects
        self._initialize()

    def _initialize(self):
        """Initialize OpenGL context and objects."""
        try:
            # Create core renderer (handles OpenGL context)
            self.renderer = Renderer()

            # Create camera controller
            self.camera = Camera()

            # Create scene renderer
            self.scene_renderer = SceneRenderer(self.renderer, self.scene)

            logger.info("Visualization engine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize visualization engine: {str(e)}")
            raise

    def start(self):
        """Start the visualization rendering loop in a separate thread."""
        if self.running:
            logger.warning("Visualization engine is already running")
            return

        self.running = True
        self.render_thread = threading.Thread(target=self._render_loop)
        self.render_thread.daemon = True
        self.render_thread.start()
        logger.info("Visualization rendering thread started")

    def stop(self):
        """Stop the visualization rendering loop."""
        self.running = False
        if self.render_thread and self.render_thread.is_alive():
            self.render_thread.join(timeout=1.0)
        logger.info("Visualization rendering thread stopped")

    def _render_loop(self):
        """Main rendering loop, runs in a separate thread."""
        try:
            # Make the OpenGL context current in this thread
            self.renderer.make_context_current()

            # Main rendering loop
            while self.running:
                # Process any pending commands
                self._process_commands()

                # Update scene data
                self._update_scene_data()

                # Render the scene
                self.renderer.begin_frame()
                self.scene_renderer.render(self.camera)
                self.renderer.end_frame()

                # Cap frame rate
                time.sleep(1 / 60)  # Target 60 FPS

        except Exception as e:
            logger.error(f"Error in rendering loop: {str(e)}")
            self.running = False

    def _process_commands(self):
        """Process commands from the command queue."""
        try:
            while True:
                cmd, args = self.command_queue.get(block=False)

                if cmd == "EXIT":
                    self.running = False
                elif cmd == "CAMERA_POSITION":
                    self.camera.set_position(*args)
                elif cmd == "CAMERA_TARGET":
                    self.camera.set_target(*args)
                elif cmd == "RENDER_MODE":
                    self.scene_renderer.set_render_mode(args)

                self.command_queue.task_done()

        except queue.Empty:
            pass

    def _update_scene_data(self):
        """Update scene data for rendering."""
        # Just pass the current scene to the scene renderer
        self.scene_renderer.update_scene(self.scene)

    def set_camera(self, position=None, target=None):
        """Set camera position and target."""
        self.command_queue.put(("CAMERA_POSITION", position))
        if target is not None:
            self.command_queue.put(("CAMERA_TARGET", target))

    def set_render_mode(self, mode):
        """Set rendering mode."""
        self.command_queue.put(("RENDER_MODE", mode))

    def update_scene(self, scene_data):
        """Update with new scene data."""
        # This method would be used when receiving data from another process
        # Currently we're directly accessing the scene, so this is a placeholder
        pass

    def update_particles(self, particle_data):
        """Update particle data."""
        # This method would be used when receiving data from another process
        pass

    def update_contacts(self, contact_data):
        """Update contact data."""
        # This method would be used when receiving data from another process
        pass

    def render(self):
        """Render a single frame (for synchronous rendering)."""
        # Make sure this is called from the main thread with GL context
        self.renderer.begin_frame()
        self.scene_renderer.render(self.camera)
        self.renderer.end_frame()

    def cleanup(self):
        """Clean up resources."""
        self.stop()
        if self.renderer:
            self.renderer.cleanup()
