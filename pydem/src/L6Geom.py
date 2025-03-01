#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np

from pydem.src.demmath import Vector3r, Vector2r, Matrix3r, Real, NAN, Math
from pydem.src.ContactGeom import ContactGeom


class L6Geom(ContactGeom):
    """
    Contact geometry with 6 degrees of freedom.

    This class represents contact geometry with relative velocity and angular velocity
    in local coordinates, along with normal displacement and reference lengths.
    """

    def __init__(self):
        """Initialize L6Geom with default values."""
        super().__init__()
        self.vel = Vector3r(0, 0, 0)  # Relative velocity in local coordinates
        self.angVel = Vector3r(
            0, 0, 0
        )  # Relative angular velocity in local coordinates
        self.uN = NAN  # Normal displacement
        self.lens = Vector2r(0, 0)  # Reference lengths
        self.contA = NAN  # Contact area
        self.trsf = np.identity(3)  # Transform from global to local coordinates

    def getMinRefLen(self) -> Real:
        """
        Get minimum reference length.

        Returns:
            Minimum reference length
        """
        if self.lens[0] <= 0:
            return self.lens[1]
        elif self.lens[1] <= 0:
            return self.lens[0]
        else:
            return min(self.lens[0], self.lens[1])

    def setInitialLocalCoords(self, locX: Vector3r):
        """
        Set initial local coordinates.

        Args:
            locX: Local x-axis direction
        """
        # Normalize the x-axis
        x = Math.safeNormalize(locX)

        # Find arbitrary y-axis perpendicular to x
        if abs(x[0]) > abs(x[1]):
            y = Math.safeNormalize(np.cross(Vector3r(0, 1, 0), x))
        else:
            y = Math.safeNormalize(np.cross(Vector3r(1, 0, 0), x))

        # Complete the right-handed coordinate system
        z = np.cross(x, y)

        # Set the transformation matrix
        self.trsf = np.zeros((3, 3), dtype=Real)
        self.trsf[:, 0] = x
        self.trsf[:, 1] = y
        self.trsf[:, 2] = z
