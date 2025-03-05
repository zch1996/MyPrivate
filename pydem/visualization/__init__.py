# pydem/visualization/__init__.py
"""
PyDEM Visualization Module
--------------------------
This module provides real-time 3D visualization of DEM simulations.
"""

from .engine import VisualizationEngine
from .renderer import Renderer
from .camera import Camera


# Function to initialize visualization system
def initialize_visualization(omega=None):
    """Initialize the visualization system."""
    from .engine import VisualizationEngine

    engine = VisualizationEngine(omega)
    return engine


# Function to start the control UI
def start_control_ui(omega=None):
    """Start the visualization with control UI."""
    from pydem.ui.controller import UIController

    controller = UIController(omega)
    return controller
