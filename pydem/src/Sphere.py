#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from typing import ClassVar, Type


from pydem.src.Shape import Shape
from pydem.src.Bound import Bound
from pydem.src.Aabb import Aabb
from pydem.src.Functor import BoundFunctor, IntraFunctor
from pydem.src.ContactGeomFunctor import Cg2_Any_Any_L6Geom__Base
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.DEMData import DEMData

from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr, AlignedBox3r, Real, NAN


class Sphere(Shape):
    """Sphere shape for DEM simulations."""

    def __init__(self):
        """Initialize sphere with default values."""
        super().__init__()
        self.shapeType = "Sphere"
        self.radius = NAN  # Sphere radius

    def selfTest(self, p):
        """
        Test sphere validity.

        Args:
            p: Particle containing this sphere

        Raises:
            RuntimeError: If sphere is invalid
        """
        if self.radius <= 0:
            raise RuntimeError(
                f"Sphere #{p.getId()}: radius must be positive (not {self.radius})"
            )

        if not self.checkNumNodes():
            raise RuntimeError(
                f"Sphere #{p.getId()}: checkNumNodes() failed (has {len(self.nodes)} nodes)"
            )

        super().selfTest(p)

    def getNumNodes(self) -> int:
        """
        Get number of nodes for this shape.

        Returns:
            Number of nodes (1 for sphere)
        """
        return 1

    def setFromRaw(self, center, radius, nn, raw):
        """
        Set sphere from raw data.

        Args:
            center: Center position
            radius: Sphere radius
            nn: Node list to append to
            raw: Raw data

        Returns:
            Updated nn list
        """
        self.setFromRaw_helper_checkRaw_makeNodes(raw, 0)
        self.radius = radius
        self.nodes[0].pos = center
        nn.append(self.nodes[0])
        return nn  # Return modified list

    def asRaw(self, center, radius, nn, raw):
        """
        Convert sphere to raw data.

        Args:
            center: Output center position
            radius: Output radius
            nn: Output node list
            raw: Output raw data

        Returns:
            Tuple of (center, radius, nn, raw) with updated values
        """
        center = self.nodes[0].pos
        radius = self.radius
        nn.append(self.nodes[0])
        # No raw data needed for sphere
        return center, radius, nn, raw  # Return all modified values

    def isInside(self, pt) -> bool:
        """
        Check if point is inside sphere.

        Args:
            pt: Point to check

        Returns:
            True if point is inside sphere, False otherwise
        """
        return (pt - self.nodes[0].pos).norm() <= self.radius

    def lumpMassInertia(self, n, density, mass, I, rotateOk):
        """
        Compute mass and inertia for sphere.

        Args:
            n: Node to lump mass to
            density: Material density
            mass: Current mass value
            I: Current inertia tensor
            rotateOk: Current rotation flag

        Returns:
            Tuple of (mass, I, rotateOk) with updated values
        """
        if n != self.nodes[0]:
            return mass, I, rotateOk  # Return unchanged values

        self.checkNodesHaveDemData()
        rotateOk = True  # Modify the flag

        m = (4.0 / 3.0) * math.pi * self.radius**3 * density
        mass += m  # Modify the mass

        # Update the inertia tensor by adding to diagonal elements
        inertia_increment = (2.0 / 5.0) * m * self.radius**2
        I = I + np.diag([inertia_increment] * 3)

        return mass, I, rotateOk  # Return modified values

    def getEquivalentRadius(self) -> Real:
        """
        Get equivalent radius.

        Returns:
            Sphere radius
        """
        return self.radius

    def getVolume(self) -> Real:
        """
        Get sphere volume.

        Returns:
            Sphere volume
        """
        return (4.0 / 3.0) * math.pi * self.radius**3

    def getAlignedBox(self) -> AlignedBox3r:
        """
        Get axis-aligned bounding box.

        Returns:
            Axis-aligned bounding box
        """
        ret = AlignedBox3r()
        ret.extend(self.nodes[0].pos - self.radius * Vector3r.ones())
        ret.extend(self.nodes[0].pos + self.radius * Vector3r.ones())
        return ret

    def applyScale(self, scale):
        """
        Apply scaling to sphere.

        Args:
            scale: Scale factor
        """
        self.radius *= scale

    def getRadius(self) -> Real:
        """
        Get sphere radius.

        Returns:
            Sphere radius
        """
        return self.radius

    def setRadius(self, r):
        """
        Set sphere radius.

        Args:
            r: New radius

        Raises:
            ValueError: If radius is not positive
        """
        if r <= 0:
            raise ValueError("Radius must be positive")
        self.radius = r

    def toString(self) -> str:
        """
        Get string representation.

        Returns:
            String representation
        """
        return f"<Sphere r={self.radius} @ {id(self)}>"


@DEM_LOGGER
class Bo1_Sphere_Aabb(BoundFunctor):
    """Functor to create Aabb from Sphere."""

    # Class variable to store the shape type this functor works with
    FUNCTOR1D_TYPE: ClassVar[Type] = None  # Will be set after Sphere is imported

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, shape):
        """
        Compute Aabb for a Sphere.

        Args:
            shape: Sphere shape
        """
        # Implementation...
        # Create bound if it doesn't exist
        if not shape.bound:
            shape.bound = Aabb()
            shape.bound.maxRot = -1  # Ignore node rotation

        aabb = shape.bound
        rad = shape.radius
        pos = shape.nodes[0].pos

        # Set AABB dimensions
        aabb.min = pos - Vector3r(rad, rad, rad)
        aabb.max = pos + Vector3r(rad, rad, rad)

    # @classmethod
    # def registerClass(cls):
    #     """Register this functor class with the FunctorFactory."""
    #     # Import here to avoid circular imports
    #     from pydem.src.FunctorFactory import FunctorFactory

    #     # Set the shape type if not already set
    #     if cls.FUNCTOR1D_TYPE is None:
    #         from pydem.src.Sphere import Sphere

    #         cls.FUNCTOR1D_TYPE = Sphere

    #     # Register with factory
    #     factory = FunctorFactory.instance()
    #     factory.registerBoundFunctor(cls.FUNCTOR1D_TYPE, cls)

    #     cls.debug(
    #         f"Registered bound functor: {cls.__name__} for shape {cls.FUNCTOR1D_TYPE.__name__}"
    #     )
    #     return cls


@DEM_LOGGER
class In2_Sphere_ElastMat(IntraFunctor):
    """Functor for handling internal forces within spheres."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.alreadyWarned_ContactLoopWithApplyForces = False

    def go(self, sh, mat, particle):
        """
        Process internal forces for a sphere.

        Args:
            sh: Sphere shape
            mat: Material
            particle: Particle
        """
        # Check if contact loop applies forces
        if not self.alreadyWarned_ContactLoopWithApplyForces:
            for e in self.scene.engines:
                if hasattr(e, "applyForces") and e.applyForces:
                    self.warn(
                        "ContactLoop.applyForces=True and In2_Sphere_ElastMat in the same simulation; "
                        "this will apply contact forces twice and is probably not what you want."
                    )
                    self.alreadyWarned_ContactLoopWithApplyForces = True
                    break

        # Apply forces from contacts
        for c in self.field.getContactContainer().getContactsForParticle(particle.id):
            if not c.isReal():
                continue

            # Get contact data
            isPA = c.getParticleA().id == particle.id
            pA = c.getParticleA()
            pB = c.getParticleB()

            # Skip if not a sphere
            if not hasattr(pA.getShape(), "getRadius") or not hasattr(
                pB.getShape(), "getRadius"
            ):
                continue

            # Get contact force and torque
            force, torque, contactPoint = c.getForceTorqueBranch(
                particle, 0, self.scene
            )

            # Debug logging
            if self.isDebug():
                self.debug(
                    f"Step {self.scene.step}: apply #{pA.getId()}/{pB.getId()}: "
                    f"F={force.transpose().format(3)}, T={torque.transpose().format(3)}"
                )
                self.debug(
                    f"\t#{pA.getId() if isPA else pB.getId()} @ {contactPoint.transpose().format(3)}, "
                    f"F={force.transpose().format(3)}, "
                    f"T={(contactPoint.cross(force) + torque).transpose().format(3)}"
                )

            # Apply force and torque
            sh.nodes[0].getData(DEMData).addForceTorque(
                force, contactPoint.cross(force) + torque
            )


@DEM_LOGGER
class Cg2_Sphere_Sphere_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """Contact geometry functor for sphere-sphere contacts."""

    FUNCTOR2D_TYPES = (Sphere, Sphere)

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.distFactor = -1  # Deprecated, use DemField.distFactor

    def setMinDist00Sq(self, s1, s2, C):
        """
        Set minimum distance squared between shapes.

        Args:
            s1: First shape
            s2: Second shape
            C: Contact object
        """
        distFactor = self.field.getDistFactor()
        C.minDist00Sq = (abs(distFactor) * (s1.getRadius() + s2.getRadius())) ** 2

    def go(self, s1, s2, shift2, force, C):
        """
        Process contact geometry between two spheres.

        Args:
            s1: First sphere
            s2: Second sphere
            shift2: Shift vector for second sphere (for periodic boundaries)
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact geometry was created/updated, False otherwise
        """
        r1 = s1.getRadius()
        r2 = s2.getRadius()

        # Check nodes and data
        assert s1.checkNumNodes()
        assert s2.checkNumNodes()
        assert s1.nodes[0].hasDataTyped(DEMData)
        assert s2.nodes[0].hasDataTyped(DEMData)

        dyn1 = s1.nodes[0].getDataTyped(DEMData)
        dyn2 = s2.nodes[0].getDataTyped(DEMData)
        distFactor = self.field.getDistFactor()

        relPos = s2.nodes[0].pos + shift2 - s1.nodes[0].pos
        unDistSq = relPos.dot(relPos) - (abs(distFactor) * (r1 + r2)) ** 2

        if unDistSq > 0 and not C.isReal() and not force:
            return False

        dist = np.linalg.norm(relPos)
        uN = dist - (r1 + r2)
        normal = relPos / dist
        contPt = s1.nodes[0].pos + (r1 + 0.5 * uN) * normal

        self.handleSpheresLikeContact(
            C,
            s1.nodes[0].pos,
            dyn1.vel,
            dyn1.angVel,
            s2.nodes[0].pos + shift2,
            dyn2.vel,
            dyn2.angVel,
            normal,
            contPt,
            uN,
            r1,
            r2,
        )

        return True
