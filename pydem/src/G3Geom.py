#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
import numpy as np

from pydem.src.demmath import Vector3r, Real, NAN
from pydem.src.ContactGeom import ContactGeom


class G3Geom(ContactGeom):
    """
    Contact geometry with 3 degrees of freedom.

    This class represents contact geometry with normal displacement,
    shear displacement, and rotation axes.
    """

    def __init__(self):
        """Initialize G3Geom with default values."""
        super().__init__()
        self.uN = NAN  # Normal displacement
        self.dShear = Vector3r(0, 0, 0)  # Shear displacement delta
        self.twistAxis = Vector3r(NAN, NAN, NAN)  # Twist rotation axis
        self.orthonormalAxis = Vector3r(NAN, NAN, NAN)  # Axis normal to twist axis
        self.normal = Vector3r(NAN, NAN, NAN)  # Contact normal in global coordinates

    def rotateVectorWithContact(self, v: Vector3r):
        """
        Rotate vector with contact.

        Args:
            v: Vector to rotate (modified in-place)
        """
        # Create rotation matrix from normal, twistAxis, and orthonormalAxis
        rotMat = np.zeros((3, 3))
        rotMat[:, 0] = self.normal
        rotMat[:, 1] = self.twistAxis
        rotMat[:, 2] = self.orthonormalAxis

        # Rotate the vector
        v = rotMat @ v

    @staticmethod
    def getIncidentVel(
        dd1, dd2, dt, shift2, shiftVel2, avoidGranularRatcheting, useAlpha
    ):
        """
        Get incident velocity.

        Args:
            dd1: DEM data for first particle
            dd2: DEM data for second particle
            dt: Time step
            shift2: Shift vector for second particle
            shiftVel2: Shift velocity for second particle
            avoidGranularRatcheting: Whether to avoid granular ratcheting
            useAlpha: Whether to use alpha parameter

        Returns:
            Incident velocity vector
        """
        # Calculate positions at t+dt/2
        if useAlpha:
            pos1 = dd1.pos + dt * dd1.vel / 2
            pos2 = dd2.pos + dt * dd2.vel / 2 + shift2
        else:
            pos1 = dd1.pos
            pos2 = dd2.pos + shift2

        # Calculate branch vectors
        if avoidGranularRatcheting:
            # Use current position only
            c1x = Vector3r(0, 0, 0)  # Branch vector for particle 1
            c2x = Vector3r(0, 0, 0)  # Branch vector for particle 2
        else:
            # Use real branch vectors
            c1x = dd1.pos - pos1  # Branch vector for particle 1
            c2x = dd2.pos - pos2  # Branch vector for particle 2

        # Calculate incident velocity
        incidentV = (dd2.vel + shiftVel2 + dd2.angVel.cross(c2x)) - (
            dd1.vel + dd1.angVel.cross(c1x)
        )

        return incidentV
