#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from enum import IntEnum, auto
import threading
import weakref
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, Any
from abc import ABC, abstractmethod

# Import math utilities from demmath
from pydem.src.demmath import Vector3r, Quaternionr, Matrix3r, NAN, INF, Real, EPSILON

from pydem.src.Object import Object
from pydem.src.Node import NodeData, Node


class DEMData(NodeData):
    """DEM-specific node data class."""

    # DOF flags
    class DOF(IntEnum):
        DOF_NONE = 0
        DOF_X = 1 << 0
        DOF_Y = 1 << 1
        DOF_Z = 1 << 2
        DOF_RX = 1 << 3
        DOF_RY = 1 << 4
        DOF_RZ = 1 << 5
        CLUMP_CLUMPED = 1 << 6
        CLUMP_CLUMP = 1 << 7
        ENERGY_SKIP = 1 << 8
        GRAVITY_SKIP = 1 << 9
        TRACER_SKIP = 1 << 10
        DAMPING_SKIP = 1 << 11

    # Common DOF combinations
    DOF_ALL = DOF.DOF_X | DOF.DOF_Y | DOF.DOF_Z | DOF.DOF_RX | DOF.DOF_RY | DOF.DOF_RZ
    DOF_XYZ = DOF.DOF_X | DOF.DOF_Y | DOF.DOF_Z
    DOF_RXRYRZ = DOF.DOF_RX | DOF.DOF_RY | DOF.DOF_RZ

    def __init__(self):
        super().__init__()
        self.vel = Vector3r(0.0, 0.0, 0.0)
        self.angVel = Vector3r(0.0, 0.0, 0.0)
        self.angMom = Vector3r(NAN, NAN, NAN)
        self.force = Vector3r(0.0, 0.0, 0.0)
        self.torque = Vector3r(0.0, 0.0, 0.0)
        self.inertia = Vector3r(0.0, 0.0, 0.0)
        self.mass = 0.0
        self.flags = 0
        self.linIx = -1
        self.parRef = []  # List of Particle objects
        self.impose = None
        self.master = None  # Will be a weakref for reference to master node
        self.mutex = threading.RLock()  # Mutex for thread-safe operations

    def getterName(self) -> str:
        """Return the getter name for this data type."""
        return "dem"

    def setDataOnNode(self, node: Node) -> None:
        """Set this data on the given node."""
        node.setDataTyped(self)

    # DOF operations
    @staticmethod
    def axisDOF(axis: int, rotational: bool = False) -> int:
        """Get DOF flag for given axis."""
        if axis < 0 or axis > 2:
            raise ValueError("Axis must be 0, 1, or 2")
        return 1 << (axis + (3 if rotational else 0))

    def getBlockedDOFs(self) -> str:
        """Get string representation of blocked DOFs."""
        result = ""
        if self.flags & DEMData.DOF.DOF_X:
            result += "x"
        if self.flags & DEMData.DOF.DOF_Y:
            result += "y"
        if self.flags & DEMData.DOF.DOF_Z:
            result += "z"
        if self.flags & DEMData.DOF.DOF_RX:
            result += "X"
        if self.flags & DEMData.DOF.DOF_RY:
            result += "Y"
        if self.flags & DEMData.DOF.DOF_RZ:
            result += "Z"
        return result

    def setBlockedDOFs(self, dofs: str) -> None:
        """Set blocked DOFs from string representation."""
        self.flags &= ~DEMData.DOF_ALL  # Reset all DOF flags

        for c in dofs:
            if c == "x":
                self.flags |= DEMData.DOF.DOF_X
            elif c == "y":
                self.flags |= DEMData.DOF.DOF_Y
            elif c == "z":
                self.flags |= DEMData.DOF.DOF_Z
            elif c == "X":
                self.flags |= DEMData.DOF.DOF_RX
            elif c == "Y":
                self.flags |= DEMData.DOF.DOF_RY
            elif c == "Z":
                self.flags |= DEMData.DOF.DOF_RZ
            else:
                raise ValueError(
                    f"Invalid DOF specification '{c}', must be one of: x,y,z,X,Y,Z"
                )

    def isBlockedNone(self) -> bool:
        """Check if no DOFs are blocked."""
        return (self.flags & DEMData.DOF_ALL) == DEMData.DOF.DOF_NONE

    def setBlockedNone(self) -> None:
        """Unblock all DOFs."""
        self.flags &= ~DEMData.DOF_ALL

    def isBlockedAll(self) -> bool:
        """Check if all DOFs are blocked."""
        return (self.flags & DEMData.DOF_ALL) == DEMData.DOF_ALL

    def isBlockedAllTrans(self) -> bool:
        """Check if all translational DOFs are blocked."""
        return (self.flags & DEMData.DOF_XYZ) == DEMData.DOF_XYZ

    def isBlockedAllRot(self) -> bool:
        """Check if all rotational DOFs are blocked."""
        return (self.flags & DEMData.DOF_RXRYRZ) == DEMData.DOF_RXRYRZ

    def setBlockedAll(self) -> None:
        """Block all DOFs."""
        self.flags |= DEMData.DOF_ALL

    def isBlockedAxisDOF(self, axis: int, rot: bool) -> bool:
        """Check if specific axis DOF is blocked."""
        return bool(self.flags & DEMData.axisDOF(axis, rot))

    # Clump operations
    def isClumped(self) -> bool:
        """Check if node is clumped."""
        return bool(self.flags & DEMData.DOF.CLUMP_CLUMPED)

    def isClump(self) -> bool:
        """Check if node is a clump."""
        return bool(self.flags & DEMData.DOF.CLUMP_CLUMP)

    def isNoClump(self) -> bool:
        """Check if node is not part of a clump."""
        return not self.isClumped() and not self.isClump()

    def setClumped(self, masterNode: Node) -> None:
        """Set node as clumped with master node."""
        self.master = weakref.ref(masterNode)
        self.flags |= DEMData.DOF.CLUMP_CLUMPED
        self.flags &= ~DEMData.DOF.CLUMP_CLUMP

    def setClump(self) -> None:
        """Set node as a clump."""
        self.flags |= DEMData.DOF.CLUMP_CLUMP
        self.flags &= ~DEMData.DOF.CLUMP_CLUMPED

    def setNoClump(self) -> None:
        """Set node as not part of a clump."""
        self.flags &= ~(DEMData.DOF.CLUMP_CLUMP | DEMData.DOF.CLUMP_CLUMPED)

    # Skip flags
    def isEnergySkip(self) -> bool:
        """Check if energy calculation is skipped."""
        return bool(self.flags & DEMData.DOF.ENERGY_SKIP)

    def setEnergySkip(self, skip: bool) -> None:
        """Set energy calculation skip flag."""
        if skip:
            self.flags |= DEMData.DOF.ENERGY_SKIP
        else:
            self.flags &= ~DEMData.DOF.ENERGY_SKIP

    def isGravitySkip(self) -> bool:
        """Check if gravity calculation is skipped."""
        return bool(self.flags & DEMData.DOF.GRAVITY_SKIP)

    def setGravitySkip(self, skip: bool) -> None:
        """Set gravity calculation skip flag."""
        if skip:
            self.flags |= DEMData.DOF.GRAVITY_SKIP
        else:
            self.flags &= ~DEMData.DOF.GRAVITY_SKIP

    def isTracerSkip(self) -> bool:
        """Check if tracer calculation is skipped."""
        return bool(self.flags & DEMData.DOF.TRACER_SKIP)

    def setTracerSkip(self, skip: bool) -> None:
        """Set tracer calculation skip flag."""
        if skip:
            self.flags |= DEMData.DOF.TRACER_SKIP
        else:
            self.flags &= ~DEMData.DOF.TRACER_SKIP

    def isDampingSkip(self) -> bool:
        """Check if damping calculation is skipped."""
        return bool(self.flags & DEMData.DOF.DAMPING_SKIP)

    def setDampingSkip(self, skip: bool) -> None:
        """Set damping calculation skip flag."""
        if skip:
            self.flags |= DEMData.DOF.DAMPING_SKIP
        else:
            self.flags &= ~DEMData.DOF.DAMPING_SKIP

    # Force & Torque
    def addForceTorque(self, force: Vector3r, torque: Vector3r = None) -> None:
        """Add force and torque to node."""
        if torque is None:
            torque = Vector3r(0.0, 0.0, 0.0)

        with self.mutex:
            self.force += force
            self.torque += torque

    def addForce(self, force: Vector3r) -> None:
        """Add force to node."""
        with self.mutex:
            self.force += force

    # Kinetic energy
    @staticmethod
    def getEkAny(node: Node, trans: bool, rot: bool, scene: "Scene" = None) -> Real:
        """Get kinetic energy of node."""
        if node is None or not node.hasDataTyped(DEMData):
            raise ValueError("Invalid node or missing DEMData")

        data = node.getDataTyped(DEMData)
        energy = 0.0

        if trans:
            fluctVel = data.vel.copy()
            if scene is not None and scene.isPeriodic:
                fluctVel = scene.cell.pprevFluctVel(node.pos, data.vel, scene.dt)
            energy += 0.5 * data.mass * np.dot(fluctVel, fluctVel)

        if rot:
            T = node.ori.toRotationMatrix()
            I = np.diag(data.inertia)
            fluctAngVel = data.angVel.copy()

            if scene is not None and scene.isPeriodic:
                fluctAngVel = scene.cell.pprevFluctAngVel(data.angVel)

            energy += 0.5 * fluctAngVel.dot(T.transpose() @ I @ T @ fluctAngVel)

        return energy

    # Particle references
    def addParticleRef(self, particle: "Particle") -> None:
        """Add particle reference."""
        if particle is None:
            raise ValueError("Cannot add null particle reference")

        if particle not in self.parRef:
            self.parRef.append(particle)

    def getParticleRefs(self) -> List["Particle"]:
        """Get particle references."""
        return self.parRef

    # Node operations
    def getMaster(self) -> Optional[Node]:
        """Get master node for clumped nodes."""
        return self.master() if self.master is not None else None

    @staticmethod
    def setOriMassInertia(node: Node) -> None:
        """Set orientation, mass, and inertia based on particles."""
        if not node.hasDataTyped(DEMData):
            raise RuntimeError("Node has no DEMData")

        data = node.getDataTyped(DEMData)
        inertia = np.zeros((3, 3), dtype=Real)
        mass = 0.0
        canRotate = True

        for p in data.parRef:
            if p.getShape() is None or p.getMaterial() is None:
                continue

            rot = True
            partialMass = 0.0
            partialInertia = np.zeros((3, 3), dtype=Real)

            p.getShape().lumpMassInertia(
                node, p.getMaterial().getDensity(), partialMass, partialInertia, rot
            )

            if not rot:
                canRotate = False

            mass += partialMass
            inertia += partialInertia

        # Check if inertia is diagonal
        isDiagonal = np.allclose(
            inertia - np.diag(np.diagonal(inertia)), 0, atol=EPSILON
        )

        if isDiagonal:
            data.mass = mass
            data.inertia = Vector3r(inertia[0, 0], inertia[1, 1], inertia[2, 2])
            return

        if not canRotate:
            raise RuntimeError("Non-diagonal inertia tensor but rotation not allowed")

        # Rotate node to align with principal axes
        eigenvalues, eigenvectors = np.linalg.eigh(inertia)
        rotation = Quaternionr.fromRotationMatrix(eigenvectors)
        node.ori = rotation * node.ori
        data.inertia = Vector3r(eigenvalues[0], eigenvalues[1], eigenvalues[2])
        data.mass = mass

    # Angular velocity
    def setAngVel(self, angVel: Vector3r) -> None:
        """Set angular velocity."""
        self.angVel = angVel
        self.angMom = Vector3r(NAN, NAN, NAN)

    def isAspherical(self) -> bool:
        """Check if inertia tensor is not spherical."""
        return not (
            abs(self.inertia[0] - self.inertia[1]) < EPSILON
            and abs(self.inertia[1] - self.inertia[2]) < EPSILON
        )

    def useAsphericalLeapfrog(self) -> bool:
        """Check if aspherical leapfrog integration should be used."""
        return self.isAspherical() and not self.isBlockedAllRot()

    # Guess if node will move
    def guessMoving(self) -> bool:
        """Guess if node will move."""
        return (
            (self.mass != 0 and not self.isBlockedAll())
            or not np.allclose(self.vel, Vector3r(0.0, 0.0, 0.0), atol=EPSILON)
            or not np.allclose(self.angVel, Vector3r(0.0, 0.0, 0.0), atol=EPSILON)
            or self.impose is not None
        )

    # Self test
    def selfTest(self, node: Node, prefix: str) -> None:
        """Perform self-test to validate data."""
        if not node.hasDataTyped(DEMData):
            raise RuntimeError(f"{prefix}: node has no DEMData")

        if node.getDataPtr(DEMData) is not self:
            raise RuntimeError(f"{prefix}: node's DEMData doesn't match this instance")

        if not self.isBlockedAllTrans() and not self.isClumped() and self.mass <= 0:
            raise RuntimeError(
                f"{prefix}: mass={self.mass} is non-positive but translations not blocked"
            )

        for axis in range(3):
            if (
                not self.isBlockedAxisDOF(axis, True)
                and not self.isClumped()
                and self.inertia[axis] <= 0
            ):
                raise RuntimeError(
                    f"{prefix}: inertia[{axis}]={self.inertia[axis]} is non-positive but rotation not blocked"
                )

        if self.impose is not None:
            self.impose.selfTest(node, self, f"{prefix}.impose")


# Register DEMData with NodeData.DataIndex
NodeData.DataIndex.DEMDATA = 0  # Use the same value as Node.DataType.DEM
