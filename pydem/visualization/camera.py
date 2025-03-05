# pydem/visualization/camera.py
"""
Camera Module
------------
Handles camera positioning and controls for 3D visualization.
"""

import numpy as np
import math


class Camera:
    """Camera for 3D visualization."""

    def __init__(self):
        """Initialize camera with default settings."""
        # Camera position (spherical coordinates)
        self.distance = 10.0
        self.azimuth = 0.0  # Horizontal angle in radians
        self.elevation = 0.0  # Vertical angle in radians

        # Target position (center of view)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Camera orientation
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # View frustum parameters
        self.fov = 45.0  # Field of view in degrees
        self.near = 0.1
        self.far = 1000.0

        # Update matrices
        self._update_matrices()

    def _update_matrices(self):
        """Update view and projection matrices."""
        # Calculate camera position in Cartesian coordinates
        self.position = np.array(
            [
                self.target[0]
                + self.distance * math.cos(self.elevation) * math.cos(self.azimuth),
                self.target[1]
                + self.distance * math.cos(self.elevation) * math.sin(self.azimuth),
                self.target[2] + self.distance * math.sin(self.elevation),
            ],
            dtype=np.float32,
        )

        # Calculate view matrix
        self.view_matrix = self._look_at(self.position, self.target, self.up)

    def _look_at(self, eye, target, up):
        """
        Compute the view matrix using look-at method.

        Args:
            eye: Camera position
            target: Target position (center of view)
            up: Up vector

        Returns:
            4x4 view matrix
        """
        # Create an orthogonal basis
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)

        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        # Create view matrix
        view = np.identity(4, dtype=np.float32)
        view[0, 0:3] = x_axis
        view[1, 0:3] = y_axis
        view[2, 0:3] = z_axis

        # Translation part
        view[0, 3] = -np.dot(x_axis, eye)
        view[1, 3] = -np.dot(y_axis, eye)
        view[2, 3] = -np.dot(z_axis, eye)

        return view

    def perspective_matrix(self, aspect_ratio):
        """
        Compute the perspective projection matrix.

        Args:
            aspect_ratio: Width/height aspect ratio

        Returns:
            4x4 perspective projection matrix
        """
        # Convert FOV from degrees to radians
        fov_rad = math.radians(self.fov)

        # Calculate projection matrix
        f = 1.0 / math.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0

        return proj

    def set_position(self, distance=None, azimuth=None, elevation=None):
        """Set camera position using spherical coordinates."""
        if distance is not None:
            self.distance = max(0.1, distance)
        if azimuth is not None:
            self.azimuth = azimuth
        if elevation is not None:
            # Limit elevation to avoid gimbal lock
            self.elevation = max(
                -math.pi / 2 + 0.01, min(math.pi / 2 - 0.01, elevation)
            )

        self._update_matrices()

    def set_target(self, x=None, y=None, z=None):
        """Set camera target position."""
        if x is not None:
            self.target[0] = x
        if y is not None:
            self.target[1] = y
        if z is not None:
            self.target[2] = z

        self._update_matrices()

    def move(self, delta_distance, delta_azimuth, delta_elevation):
        """Move camera relative to current position."""
        self.set_position(
            self.distance + delta_distance,
            self.azimuth + delta_azimuth,
            self.elevation + delta_elevation,
        )

    def pan(self, delta_x, delta_y):
        """Pan camera (move target position)."""
        # Calculate right and up vectors in camera space
        right = np.array(
            [
                math.cos(self.azimuth - math.pi / 2),
                math.sin(self.azimuth - math.pi / 2),
                0.0,
            ],
            dtype=np.float32,
        )

        up = np.array(
            [
                -math.sin(self.elevation) * math.cos(self.azimuth),
                -math.sin(self.elevation) * math.sin(self.azimuth),
                math.cos(self.elevation),
            ],
            dtype=np.float32,
        )

        # Scale movement based on distance
        scale = self.distance * 0.01

        # Update target
        self.target += right * delta_x * scale
        self.target += up * delta_y * scale

        self._update_matrices()

    def zoom(self, delta):
        """Zoom camera in or out."""
        self.distance = max(0.1, self.distance - delta)
        self._update_matrices()
