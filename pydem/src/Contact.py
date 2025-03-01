#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import weakref
from typing import Tuple, Optional, Any
import numpy as np

from pydem.src.Object import Object
from pydem.src.demmath import Vector3r, NAN
from pydem.src.ContactGeom import ContactGeom
from pydem.src.ContactPhys import ContactPhys
from pydem.src.ContactData import ContactData
from pydem.src.DEMLogging import DEM_LOGGER


@DEM_LOGGER
class Contact(Object):
    """Class representing a contact between two particles."""

    def __init__(self):
        """Initialize contact with default values."""
        super().__init__()
        self.geom = None  # ContactGeom
        self.phys = None  # ContactPhys
        self.data = None  # ContactData
        self.pA = None  # weakref to Particle A
        self.pB = None  # weakref to Particle B
        self.cellDist = np.array(
            [0, 0, 0], dtype=int
        )  # Cell distance for periodic boundary
        self.stepCreated = -1  # Step when contact was created
        self.stepLastSeen = -1  # Step when contact was last seen
        self.minDist00Sq = -1  # Minimum distance squared
        self.linIx = 0  # Linear index in container

    # Status checks
    def isReal(self) -> bool:
        """Check if contact is real (has geometry and physics)."""
        return self.geom is not None and self.phys is not None

    def isColliding(self) -> bool:
        """Check if particles are colliding."""
        return self.stepCreated >= 0

    def setNotColliding(self) -> None:
        """Set contact as not colliding."""
        self.stepCreated = -1

    def isFresh(self, scene) -> bool:
        """Check if contact was created in the current step."""
        return scene.step == self.stepCreated

    # Particle operations
    def getParticleA(self):
        """Get particle A."""
        return self.pA() if self.pA is not None else None

    def getParticleB(self):
        """Get particle B."""
        return self.pB() if self.pB is not None else None

    def pIndex(self, p) -> int:
        """Get index (0 for A, 1 for B) of particle."""
        if isinstance(p, object) and hasattr(p, "get"):
            p = p.get()
        return 0 if p == self.leakPA() else 1

    # Force operations
    def getForceSign(self, p) -> int:
        """
        Get sign of force for a particle.

        Args:
            p: Particle

        Returns:
            1 for particle A, -1 for particle B
        """
        ptrA = self.leakPA()
        ptrB = self.leakPB()

        if p != ptrA and p != ptrB:
            raise RuntimeError(
                f"Particle {p.toString()} does not participate in contact"
            )
        return 1 if p == ptrA else -1

    def getForceSignById(self, id) -> int:
        """
        Get sign of force for a particle by ID.

        Args:
            id: Particle ID

        Returns:
            1 for particle A, -1 for particle B
        """
        idA = self.getParticleA().getId() if self.getParticleA() else -1
        idB = self.getParticleB().getId() if self.getParticleB() else -1

        if id != idA and id != idB:
            raise RuntimeError(f"Particle #{id} does not participate in contact")
        return 1 if id == idA else -1

    # Position operations
    def getPositionDifference(self, scene) -> Vector3r:
        """
        Get position difference between particles.

        Args:
            scene: Scene object

        Returns:
            Vector3r: Position difference
        """
        if not self.leakPA() or not self.leakPB():
            return Vector3r(NAN, NAN, NAN)

        pA = self.leakPA()
        pB = self.leakPB()

        pA.checkNodes(False, True)
        pB.checkNodes(False, True)

        diff = pB.getShape().getNodes()[0].pos - pA.getShape().getNodes()[0].pos

        if not scene.isPeriodic or np.all(self.cellDist == 0):
            return diff

        return diff + scene.getCell().intrShiftPos(self.cellDist)

    def getDistance(self) -> float:
        """Get distance between particles."""
        from Omega import Omega

        return np.linalg.norm(self.getPositionDifference(Omega.instance().getScene()))

    # Contact management
    def swapOrder(self) -> None:
        """Swap order of particles."""
        if self.geom or self.phys:
            raise RuntimeError(
                "Cannot swap particles in contact with existing geom or phys"
            )
        self.pA, self.pB = self.pB, self.pA
        self.cellDist *= -1

    def reset(self) -> None:
        """Reset contact geometry and physics."""
        self.geom = None
        self.phys = None

    # Force and torque computation
    def getForceTorqueBranch(
        self, particle, nodeIndex, scene
    ) -> Tuple[Vector3r, Vector3r, Vector3r]:
        """
        Get force, torque, and branch vector for a particle.

        Args:
            particle: Particle
            nodeIndex: Node index
            scene: Scene object

        Returns:
            Tuple of (force, torque, branch vector)
        """
        if self.leakPA() != particle and self.leakPB() != particle:
            raise ValueError("Particle not in contact")

        if not self.geom or not self.phys:
            raise RuntimeError("Contact has no geometry or physics")

        if (
            nodeIndex < 0
            or not particle.getShape()
            or nodeIndex >= len(particle.getShape().getNodes())
        ):
            raise IndexError("Invalid node index")

        isParticleA = self.leakPA() == particle
        sign = 1 if isParticleA else -1

        # Calculate force in global coordinates
        force = self.geom.getNode().ori * self.phys.getForce() * sign

        # Calculate torque
        if np.allclose(self.phys.getTorque(), Vector3r(0.0, 0.0, 0.0)):
            torque = Vector3r(0.0, 0.0, 0.0)
        else:
            torque = self.geom.getNode().ori * self.phys.getTorque() * sign

        # Calculate branch vector
        shift = Vector3r(0.0, 0.0, 0.0)
        if not isParticleA and scene.isPeriodic:
            shift = scene.getCell().intrShiftPos(self.cellDist)

        branchVector = self.geom.getNode().pos - (
            particle.getShape().getNodes()[nodeIndex].pos + shift
        )

        return (force, torque, branchVector)

    # Helper methods
    def leakPA(self):
        """Get raw pointer to particle A."""
        if self.pA is None:
            return None
        return self.pA() if callable(self.pA) else self.pA

    def leakPB(self):
        """Get raw pointer to particle B."""
        if self.pB is None:
            return None
        return self.pB() if callable(self.pB) else self.pB

    def leakOther(self, p):
        """Get raw pointer to the other particle."""
        if p != self.leakPA() and p != self.leakPB():
            return None
        return self.leakPA() if p != self.leakPA() else self.leakPB()

    # Getters
    def getGeom(self):
        """Get contact geometry."""
        return self.geom

    def getPhys(self):
        """Get contact physics."""
        return self.phys

    def getData(self):
        """Get contact data."""
        return self.data

    def getCellDist(self):
        """Get cell distance vector for periodic boundaries."""
        return self.cellDist

    def getMinDist00Sq(self):
        """Get minimum distance squared."""
        return self.minDist00Sq

    def getStepCreated(self):
        """Get step when contact was created."""
        return self.stepCreated

    def getStepLastSeen(self):
        """Get step when contact was last seen."""
        return self.stepLastSeen

    def getLinearIndex(self):
        """Get linear index in contact container."""
        return self.linIx

    def toString(self) -> str:
        """Return string representation of the contact."""
        id_a = -1 if self.pA is None or self.leakPA() is None else self.leakPA().getId()
        id_b = -1 if self.pB is None or self.leakPB() is None else self.leakPB().getId()
        return f"Contact ##{id_a}+{id_b} @ {id(self)}"
