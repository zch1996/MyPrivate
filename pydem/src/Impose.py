#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import weakref
from enum import IntFlag
from typing import List, Optional, Union, Set, Dict, Any
from pydem.src.demmath import Vector3r, NAN

from pydem.src.Object import Object


class Impose(Object):
    """Base class for imposing constraints on nodes."""

    # Constraint types as IntFlag for bit operations
    class Type(IntFlag):
        NONE = 0
        VELOCITY = 1 << 0
        FORCE = 1 << 1
        INIT_VELOCITY = 1 << 2
        READ_FORCE = 1 << 3

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.what = self.Type.NONE
        self.stepLast = -1
        self.timeLast = NAN
        self.mutex = threading.RLock()

    def velocity(self, scene, node):
        """Apply velocity constraint."""
        raise RuntimeError(f"Abstract Impose::velocity called")

    def force(self, scene, node):
        """Apply force constraint."""
        raise RuntimeError(f"Abstract Impose::force called")

    def readForce(self, scene, node):
        """Read force from constraint."""
        raise RuntimeError(f"Abstract Impose::readForce called")

    def selfTest(self, node, demData, prefix):
        """Verify constraint is valid."""
        # Default implementation does nothing
        pass

    def isFirstStepRun(self, scene, lastTime=None):
        """Check if this is the first time running in this step."""
        if scene is None:
            raise ValueError("Scene cannot be null")

        # Double check with lock
        with self.mutex:
            if self.stepLast == scene.step:
                return False

            if lastTime is not None:
                lastTime[0] = self.timeLast

            self.timeLast = scene.time
            self.stepLast = scene.step
            return True

    def getType(self):
        """Get constraint type."""
        return self.what

    def setType(self, type):
        """Set constraint type."""
        self.what = type

    def hasType(self, type):
        """Check if constraint has specific type."""
        return bool(self.what & type)

    def getLastStep(self):
        """Get last step when constraint was applied."""
        return self.stepLast

    def getLastTime(self):
        """Get last time when constraint was applied."""
        return self.timeLast

    def combine(self, other):
        """Combine with another impose."""
        if other is None:
            return self

        combined = CombinedImpose()
        combined.addImpose(self)
        combined.addImpose(other)
        return combined


class CombinedImpose(Impose):
    """Combined impose for multiple constraints."""

    def __init__(self):
        """Initialize with empty impose list."""
        super().__init__()
        self.imposes = []

    def velocity(self, scene, node):
        """Apply all velocity constraints."""
        for impose in self.imposes:
            if impose.hasType(Impose.Type.VELOCITY):
                impose.velocity(scene, node)

    def force(self, scene, node):
        """Apply all force constraints."""
        for impose in self.imposes:
            if impose.hasType(Impose.Type.FORCE):
                impose.force(scene, node)

    def readForce(self, scene, node):
        """Read forces from all constraints."""
        for impose in self.imposes:
            if impose.hasType(Impose.Type.READ_FORCE):
                impose.readForce(scene, node)

    def selfTest(self, node, demData, prefix):
        """Verify all constraints are valid."""
        for impose in self.imposes:
            impose.selfTest(node, demData, f"{prefix}[combined]")

    def addImpose(self, impose):
        """Add an impose to the collection."""
        if impose is None:
            raise ValueError("Cannot add null impose")

        # Check if impose is already in the list
        if impose in self.imposes:
            return

        # Handle nested CombinedImposes
        if isinstance(impose, CombinedImpose):
            for subImpose in impose.getImposes():
                self.addImpose(subImpose)
            return

        self.imposes.append(impose)
        self.what |= impose.what

    def getImposes(self):
        """Get list of imposes."""
        return self.imposes


# Additional concrete Impose implementations


class ImposeVelocity(Impose):
    """Impose specific velocity on a node."""

    def __init__(self, vel=None):
        """Initialize with velocity vector."""
        super().__init__()
        self.vel = vel if vel is not None else Vector3r(0.0, 0.0, 0.0)
        self.setType(Impose.Type.VELOCITY)

    def velocity(self, scene, node):
        """Apply velocity constraint."""
        demData = node.getDataTyped(DEMData)
        demData.vel = self.vel.copy()

    def getVel(self):
        """Get imposed velocity."""
        return self.vel

    def setVel(self, vel):
        """Set imposed velocity."""
        self.vel = vel


class ImposeForce(Impose):
    """Impose specific force on a node."""

    def __init__(self, force=None):
        """Initialize with force vector."""
        super().__init__()
        self.force = force if force is not None else Vector3r(0.0, 0.0, 0.0)
        self.setType(Impose.Type.FORCE)

    def force(self, scene, node):
        """Apply force constraint."""
        demData = node.getDataTyped(DEMData)
        demData.addForce(self.force)

    def getForce(self):
        """Get imposed force."""
        return self.force

    def setForce(self, force):
        """Set imposed force."""
        self.force = force


class ImposeAngularVelocity(Impose):
    """Impose specific angular velocity on a node."""

    def __init__(self, angVel=None):
        """Initialize with angular velocity vector."""
        super().__init__()
        self.angVel = angVel if angVel is not None else Vector3r(0.0, 0.0, 0.0)
        self.setType(Impose.Type.VELOCITY)

    def velocity(self, scene, node):
        """Apply angular velocity constraint."""
        demData = node.getDataTyped(DEMData)
        demData.setAngVel(self.angVel.copy())

    def getAngVel(self):
        """Get imposed angular velocity."""
        return self.angVel

    def setAngVel(self, angVel):
        """Set imposed angular velocity."""
        self.angVel = angVel


class ImposeTorque(Impose):
    """Impose specific torque on a node."""

    def __init__(self, torque=None):
        """Initialize with torque vector."""
        super().__init__()
        self.torque = torque if torque is not None else Vector3r(0.0, 0.0, 0.0)
        self.setType(Impose.Type.FORCE)

    def force(self, scene, node):
        """Apply torque constraint."""
        demData = node.getDataTyped(DEMData)
        demData.addForceTorque(Vector3r(0.0, 0.0, 0.0), self.torque)

    def getTorque(self):
        """Get imposed torque."""
        return self.torque

    def setTorque(self, torque):
        """Set imposed torque."""
        self.torque = torque


# For completeness, import the DEMData class
from pydem.src.DEMData import DEMData
