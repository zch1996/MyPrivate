#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Engine import Engine
from pydem.src.Dispatcher import BoundDispatcher
from pydem.src.Aabb import Aabb
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import Vector3r, Real, INF, NAN
import numpy as np
import math


@DEM_LOGGER
class Collider(Engine):
    """Abstract base class for collision detection."""

    def __init__(self):
        """Initialize collider with default values."""
        super().__init__()
        self.nFullRuns = 0  # Number of full collision detection runs

    def probeAabb(self, min_corner, max_corner):
        """
        Probe axis-aligned bounding box for particle presence.

        Args:
            min_corner: Minimum corner of AABB
            max_corner: Maximum corner of AABB

        Returns:
            List of particle IDs within the AABB
        """
        raise RuntimeError("Calling abstract Collider::probeAabb")

    @staticmethod
    def mayCollide(dem, pA, pB):
        """
        Check if two particles may collide.

        Args:
            dem: DEM field
            pA: First particle
            pB: Second particle

        Returns:
            bool: True if particles may collide
        """
        # Basic validity checks
        if not pA or not pB or not pA.getShape() or not pB.getShape() or pA is pB:
            return False

        # Check collision masks
        if not (pA.getMask() & pB.getMask()):
            return False

        # Check lone masks
        if (pA.getMask() & pB.getMask() & dem.getLoneMask()) != 0:
            return False

        # Check node sharing
        nodesA = pA.getShape().getNodes()
        nodesB = pB.getShape().getNodes()

        for nA in nodesA:
            for nB in nodesB:
                # Check direct node sharing
                if nA is nB:
                    return False

                # Check clump membership
                if not nA.hasDataTyped(DEMData) or not nB.hasDataTyped(DEMData):
                    continue

                dynA = nA.getDataTyped(DEMData)
                dynB = nB.getDataTyped(DEMData)

                if dynA.isClumped() and dynB.isClumped():
                    masterA = dynA.getMaster()
                    masterB = dynB.getMaster()
                    if masterA and masterB and masterA is masterB:
                        return False

        return True

    def invalidatePersistentData(self):
        """Invalidate persistent data."""
        pass


@DEM_LOGGER
class AabbCollider(Collider):
    """Collider using axis-aligned bounding boxes."""

    def __init__(self):
        """Initialize AABB collider with default values."""
        super().__init__()
        self.boundDispatcher = BoundDispatcher()
        self.verletDist = -0.05  # Negative means auto-compute
        self.noBoundOk = False  # Allow particles without bounds

    def aabbIsDirty(self, p):
        """
        Check if AABB needs update.

        Args:
            p: Particle

        Returns:
            bool: True if AABB needs update
        """
        if not p.getShape().bound:
            self.trace(f"Recomputing bounds for #{p.getId()} without bound")
            return True

        aabb = p.getShape().bound.cast(Aabb)
        nNodes = len(p.getShape().getNodes())

        # Check for invalid bounds
        if (
            len(aabb.nodeLastPos) != nNodes
            or np.isnan(np.max(aabb.min))
            or np.isnan(np.max(aabb.max))
        ):
            return True

        # Check rotation
        moveDueToRot2 = 0.0
        if aabb.maxRot >= 0:
            maxRot = 0.0
            for i in range(nNodes):
                # Calculate rotation angle between last and current orientation
                q1 = aabb.nodeLastOri[i].conjugate()
                q2 = p.getShape().getNodes()[i].ori
                q = q1 * q2
                angle = 2 * math.acos(min(1.0, abs(q.w)))
                maxRot = max(maxRot, abs(angle))

            if maxRot > aabb.maxRot:
                self.trace(f"Recomputing bounds for #{p.getId()} due to rotation")
                return True

            moveDueToRot2 = (0.5 * np.max(aabb.max - aabb.min) * maxRot) ** 2

        # Check translation
        maxMove2 = 0.0
        for i in range(nNodes):
            dist2 = np.sum((aabb.nodeLastPos[i] - p.getShape().getNodes()[i].pos) ** 2)
            maxMove2 = max(maxMove2, dist2)

        if maxMove2 + moveDueToRot2 > aabb.maxD2:
            self.trace(f"Recomputing bounds for #{p.getId()} due to translation")
            return True

        print("Aabb is not dirty")
        return False

    def updateScenePtr(self, scene, field):
        """
        Update scene and field pointers.

        Args:
            scene: Scene object
            field: Field object
        """
        super().updateScenePtr(scene, field)

        # Update boundDispatcher's scene and field
        if not self.boundDispatcher:
            raise RuntimeError(
                f"AabbCollider.boundDispatcher is None ({self.toString()})."
            )

        # Initialize functors in dispatchers
        self.boundDispatcher.initializeFunctors()
        self.boundDispatcher.updateScenePtr(scene, field)

    def updateAabb(self, p):
        """
        Update AABB for particle.

        Args:
            p: Particle
        """
        if not p or not p.getShape():
            return

        # Update bounds using dispatcher
        self.boundDispatcher(p.getShape())

        # Check bound creation
        if not p.getShape().bound:
            if self.noBoundOk:
                return
            raise RuntimeError(f"No bound created for particle #{p.getId()}")

        # Update AABB data
        aabb = p.getShape().bound.cast(Aabb)
        nodes = p.getShape().getNodes()
        nNodes = len(nodes)

        # Store current positions and orientations
        aabb.nodeLastPos = [nodes[i].pos.copy() for i in range(nNodes)]
        aabb.nodeLastOri = [nodes[i].ori.copy() for i in range(nNodes)]
        aabb.maxD2 = self.verletDist**2

        if np.isnan(aabb.maxRot):
            raise RuntimeError(
                f"Bound functor did not set maxRot for particle #{p.getId()}"
            )

        # Expand bounds by Verlet distance
        if self.verletDist > 0:
            if aabb.maxRot >= 0:
                maxArm = 0.5 * np.max(aabb.max - aabb.min)
                aabb.maxRot = math.atan(self.verletDist / maxArm) if maxArm > 0 else 0.0

            expansion = self.verletDist * np.ones(3)
            print(f"AABB Before Update: {aabb.min} {aabb.max}")
            aabb.max = aabb.max + expansion
            aabb.min = aabb.min - expansion
            print(f"AABB After Update: {aabb.min} {aabb.max}")

    def setVerletDist(self, scene, dem):
        """
        Set Verlet distance.

        Args:
            scene: Scene object
            dem: DEM field
        """
        if self.verletDist >= 0:
            return

        # Find minimum radius among particles and inlets
        minR = INF

        # Check particles
        for p in dem.getParticles():
            if not p or not p.getShape():
                continue
            r = p.getShape().getEquivalentRadius()
            if not np.isnan(r):
                minR = min(r, minR)

        # Check inlets
        for e in scene.engines:
            if hasattr(e, "minMaxDiam"):  # Check if it's an Inlet
                dMin = e.minMaxDiam()[0]
                if not np.isnan(dMin):
                    minR = min(0.5 * dMin, minR)

        # Set Verlet distance
        if not np.isinf(minR):
            self.verletDist = abs(self.verletDist) * minR
        else:
            self.warning("No valid radius found for auto Verlet distance, setting to 0")
            self.verletDist = 0.0


# Import at the end to avoid circular imports
from pydem.src.DEMData import DEMData
