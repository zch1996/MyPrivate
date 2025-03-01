#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Object import Object
from pydem.src.demmath import Vector3r, NAN


class ContactPhys(Object):
    """Base class for contact physics information."""

    def __init__(self):
        """Initialize ContactPhys with default values."""
        super().__init__()
        self.force = Vector3r(0.0, 0.0, 0.0)
        self.torque = Vector3r(0.0, 0.0, 0.0)

    def getForce(self):
        """Get contact force."""
        return self.force

    def setForce(self, f):
        """Set contact force."""
        self.force = f

    def getTorque(self):
        """Get contact torque."""
        return self.torque

    def setTorque(self, t):
        """Set contact torque."""
        self.torque = t

    def getContactPhysType(self):
        """Get the contact physics type."""
        return self.getClassName()

    def toString(self) -> str:
        """Return string representation."""
        return f"{self.getClassName()}"


class FrictPhys(ContactPhys):
    """Friction physics information."""

    def __init__(self):
        super().__init__()
        self.tanPhi = NAN
        self.kn = NAN  # Normal stiffness
        self.kt = NAN  # Tangential stiffness
