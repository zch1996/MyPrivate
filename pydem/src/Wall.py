#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from pydem.src.Shape import Shape
from pydem.src.Functor import BoundFunctor, IntraFunctor
from pydem.src.ContactGeomFunctor import Cg2_Any_Any_L6Geom__Base
from pydem.src.Aabb import Aabb
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import (
    Vector3r,
    Vector2r,
    Matrix3r,
    Quaternionr,
    AlignedBox2r,
    Real,
    INF,
    NAN,
)
from pydem.src.DEMData import DEMData
from pydem.src.Sphere import Sphere
from pydem.src.Facet import Facet
from pydem.src.Node import Node
from pydem.src.Material import ElastMat
from pydem.src.L6Geom import L6Geom


class Wall(Shape):
    """
    Object representing infinite plane aligned with the coordinate system.

    This shape represents an axis-aligned wall that extends infinitely
    along two coordinate axes.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.sense = (
            0  # Which side interacts: -1 for negative, 0 for both, +1 for positive
        )
        self.axis = 0  # Axis of the normal (0,1,2 for x,y,z)
        self.glAB = AlignedBox2r(
            Vector2r(NAN, NAN), Vector2r(NAN, NAN)
        )  # Points for rendering

    def numNodes(self):
        """Return number of nodes needed by this shape."""
        return 1

    def lumpMassInertia(self, node, density, mass, I, rotateOk):
        """
        Compute mass and inertia for this shape.

        Args:
            node: Node to compute mass for
            density: Material density
            mass: Mass value to update
            I: Inertia tensor to update
            rotateOk: Whether rotation is allowed

        Returns:
            Updated mass, inertia tensor, and rotateOk flag
        """
        # Wall doesn't add mass or inertia
        rotateOk = True
        return mass, I, rotateOk


@DEM_LOGGER
class Bo1_Wall_Aabb(BoundFunctor):
    """
    Creates/updates an Aabb of a Wall.

    This functor computes the axis-aligned bounding box for a wall.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for a Wall.

        Args:
            sh: Wall shape
        """
        wall = sh

        # Create bound if it doesn't exist
        if not wall.bound:
            wall.bound = Aabb()
            wall.bound.maxRot = -1  # Ignore node rotation

        assert wall.numNodesOk()
        aabb = wall.bound

        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "Walls not supported in skewed (Scene.cell.trsf is not diagonal) periodic boundary conditions."
            )

        # Set AABB dimensions - infinite in all directions except wall axis
        aabb.min = Vector3r(-INF, -INF, -INF)
        aabb.max = Vector3r(INF, INF, INF)

        # Wall position defines the plane
        aabb.min[wall.axis] = aabb.max[wall.axis] = sh.nodes[0].pos[wall.axis]

    # @classmethod
    # def registerClass(cls):
    #     """Register this functor class with the FunctorFactory."""
    #     # Import here to avoid circular imports
    #     from pydem.src.FunctorFactory import FunctorFactory

    #     # Set the shape type if not already set
    #     if cls.FUNCTOR1D_TYPE is None:
    #         from pydem.src.Wall import Wall

    #         cls.FUNCTOR1D_TYPE = Wall

    #     # Register with factory
    #     factory = FunctorFactory.instance()
    #     factory.registerBoundFunctor(cls.FUNCTOR1D_TYPE, cls)

    #     cls.debug(
    #         f"Registered bound functor: {cls.__name__} for shape {cls.FUNCTOR1D_TYPE.__name__}"
    #     )
    #     return cls


@DEM_LOGGER
class In2_Wall_ElastMat(IntraFunctor):
    """
    Apply contact forces on wall.

    Wall generates no internal forces as such. Torque from applied forces
    is discarded, as Wall does not rotate.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def __call__(self, shape, material, particle, update=True):
        """
        Apply forces to wall from contacts.

        Args:
            shape: Wall shape
            material: Material
            particle: Particle
            update: Whether to update forces
        """
        # Process all contacts
        for C in particle.contacts.values():
            if not C.isReal():
                continue

            # Get force, torque, and branch vector
            F, T, xc = C.getForceTorqueBranch(particle, 0, self.scene)

            # Add force to wall node, but discard torque
            shape.nodes[0].getDataTyped(DEMData).addForceTorque(F, Vector3r(0, 0, 0))


@DEM_LOGGER
class Cg2_Wall_Sphere_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between Wall and Sphere.

    This functor handles collision detection and geometry computation
    between a wall and a sphere.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Wall and Sphere.

        Args:
            sh1: Wall shape
            sh2: Sphere shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "Cg2_Wall_Sphere_L6Geom does not handle periodic boundary conditions with skew (Scene.cell.trsf is not diagonal)."
            )

        wall = sh1
        sphere = sh2

        assert wall.numNodesOk()
        assert sphere.numNodesOk()

        radius = sphere.radius
        ax = wall.axis
        sense = wall.sense
        wallPos = wall.nodes[0].pos
        spherePos = sphere.nodes[0].pos + shift2

        # Signed "distance" between centers
        dist = spherePos[ax] - wallPos[ax]

        # Check if wall and sphere are too far from each other
        if not C.isReal() and abs(dist) > radius and not force:
            return False

        # Contact point is sphere center projected onto the wall
        contPt = spherePos.copy()
        contPt[ax] = wallPos[ax]

        normal = Vector3r(0, 0, 0)

        # Determine normal direction based on wall sense
        assert sense == -1 or sense == 0 or sense == 1

        if sense == 0:
            # For new contacts, normal given by the sense of approaching the wall
            if not C.geom:
                normal[ax] = 1.0 if dist > 0 else -1.0
            # For existing contacts, use the previous normal
            else:
                normal[ax] = C.geom.trsf[0, ax]
        else:
            normal[ax] = 1.0 if sense == 1 else -1.0

        # Calculate penetration depth
        uN = normal[ax] * dist - radius

        # Get dynamics data
        wallDyn = wall.nodes[0].getDataTyped(DEMData)
        sphereDyn = sphere.nodes[0].getDataTyped(DEMData)

        # Handle contact like sphere-sphere contact
        self.handleSpheresLikeContact(
            C,
            wallPos,
            wallDyn.vel,
            wallDyn.angVel,
            spherePos,
            sphereDyn.vel,
            sphereDyn.angVel,
            normal,
            contPt,
            uN,
            -radius,  # r1 (negative radius for wall)
            radius,  # r2
        )

        return True

    def goReverse(self, sh1, sh2, shift2, force, C):
        """Handle reversed contact arguments."""
        raise RuntimeError(
            "ContactLoop should swap interaction arguments, should be Wall+Sphere, but is "
            + sh1.__class__.__name__
            + "+"
            + sh2.__class__.__name__
        )

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = -1


@DEM_LOGGER
class Cg2_Wall_Facet_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between Wall and Facet.

    This functor handles collision detection and geometry computation
    between a wall and a facet.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Wall and Facet.

        Args:
            sh1: Wall shape
            sh2: Facet shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "Cg2_Wall_Facet_L6Geom does not handle periodic boundary conditions with skew (Scene.cell.trsf is not diagonal)."
            )

        wall = sh1
        facet = sh2

        assert wall.numNodesOk()
        assert facet.numNodesOk()

        # Check for zero-thickness facet
        if not (facet.halfThick > 0.0):
            self.warning(
                "Cg2_Wall_Facet_L6Geom: Contact of Wall with zero-thickness facet is always false."
            )
            return False

        radius = facet.halfThick
        ax = wall.axis
        sense = wall.sense
        wallPos = wall.nodes[0].pos

        # Get facet vertex positions
        fPos = [
            facet.nodes[0].pos + shift2,
            facet.nodes[1].pos + shift2,
            facet.nodes[2].pos + shift2,
        ]

        # Calculate distances from wall to facet vertices
        dist = np.array(
            [
                fPos[0][ax] - wallPos[ax],
                fPos[1][ax] - wallPos[ax],
                fPos[2][ax] - wallPos[ax],
            ]
        )

        # Check if wall and facet are too far from each other
        if (
            not C.isReal()
            and abs(dist[0]) > radius
            and abs(dist[1]) > radius
            and abs(dist[2]) > radius
            and not force
        ):
            return False

        normal = Vector3r(0, 0, 0)

        # Determine normal direction
        if sense == 0:
            # For new contacts, normal is given by the sense the wall is being approached
            if not C.geom:
                normal[ax] = 1.0 if dist.sum() > 0 else -1.0
            # Use the previous normal for existing contacts
            else:
                normal[ax] = C.geom.trsf[0, ax]
        else:
            normal[ax] = 1.0 if sense == 1 else -1.0

        # Calculate contact point
        contPt = Vector3r(0, 0, 0)
        contPtWeightSum = 0.0
        uN = INF
        minIx = -1

        # Find minimum distance vertex and calculate weighted contact point
        for i in range(3):
            # Penetration depth for this vertex
            uNi = dist[i] * normal[ax] - radius

            # Track minimum distance vertex
            if uNi < uN:
                uN = uNi
                minIx = i

            # Skip vertices with no overlap
            if uNi >= 0:
                continue

            # Add weighted contribution to contact point
            weight = -uNi  # Use penetration depth as weight
            contPt += fPos[i] * weight
            contPtWeightSum += weight

        # Finalize contact point
        if contPtWeightSum != 0.0:
            # Use weighted average of overlapping vertices
            contPt /= contPtWeightSum
        else:
            # Use closest vertex if none are overlapping
            contPt = fPos[minIx]

        # Project contact point onto wall
        contPt[ax] = wallPos[ax]

        # Get facet center and velocities
        fCenter = facet.getCentroid()
        fLinVel, fAngVel = facet.interpolatePtLinAngVel(fCenter)

        # Get wall dynamics
        wallDyn = wall.nodes[0].getDataTyped(DEMData)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            wallPos,
            wallDyn.vel,
            wallDyn.angVel,
            fCenter,
            fLinVel,
            fAngVel,
            normal,
            contPt,
            uN,
            -radius,  # r1 (negative radius for wall)
            radius,  # r2
        )

        return True

    def goReverse(self, sh1, sh2, shift2, force, C):
        """Handle reversed contact arguments."""
        raise RuntimeError(
            "ContactLoop should swap interaction arguments, should be Wall+Facet, but is "
            + sh1.__class__.__name__
            + "+"
            + sh2.__class__.__name__
        )

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = -1


# # Register classes for dispatching
# Wall.registerClass()
# Bo1_Wall_Aabb.registerClass()
# In2_Wall_ElastMat.registerClass()
# Cg2_Wall_Sphere_L6Geom.registerClass()
# Cg2_Wall_Facet_L6Geom.registerClass()
