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
from pydem.src.demmath import Vector3r, Vector2r, Matrix3r, Quaternionr, Real, INF, NAN
from pydem.src.DEMData import DEMData
from pydem.src.Sphere import Sphere
from pydem.src.Node import Node


class InfCylinder(Shape):
    """
    Object representing infinite cylinder aligned with a coordinate axis.

    This shape represents an infinite cylinder that extends infinitely along
    one of the coordinate axes (x, y, or z).
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.radius = NAN  # Radius of the cylinder
        self.axis = 0  # Axis of the cylinder (0,1,2 for x,y,z)
        self.glAB = Vector2r(NAN, NAN)  # Endpoints for rendering

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
        # Infinite cylinder has infinite mass and inertia
        mass += INF
        I[0, 0] += INF
        I[1, 1] += INF
        I[2, 2] += INF
        rotateOk = False

        return mass, I, rotateOk


@DEM_LOGGER
class Bo1_InfCylinder_Aabb(BoundFunctor):
    """
    Creates/updates an Aabb of an InfCylinder.

    This functor computes the axis-aligned bounding box for an infinite cylinder.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for an InfCylinder.

        Args:
            sh: InfCylinder shape
        """
        cyl = sh
        assert cyl.axis >= 0 and cyl.axis < 3

        # Create bound if it doesn't exist
        if not cyl.bound:
            cyl.bound = Aabb()
            cyl.bound.maxRot = -1  # Ignore node rotation

        assert cyl.numNodesOk()
        aabb = cyl.bound

        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "InfCylinder not supported in periodic cell with skew (Scene.cell.trsf is not diagonal)."
            )

        pos = cyl.nodes[0].pos
        ax0 = cyl.axis
        ax1 = (cyl.axis + 1) % 3
        ax2 = (cyl.axis + 2) % 3

        # Set AABB dimensions
        aabb.min[ax0] = -INF
        aabb.max[ax0] = INF
        aabb.min[ax1] = pos[ax1] - cyl.radius
        aabb.max[ax1] = pos[ax1] + cyl.radius
        aabb.min[ax2] = pos[ax2] - cyl.radius
        aabb.max[ax2] = pos[ax2] + cyl.radius


@DEM_LOGGER
class Cg2_InfCylinder_Sphere_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between InfCylinder and Sphere.

    This functor handles collision detection and geometry computation
    between an infinite cylinder and a sphere.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between InfCylinder and Sphere.

        Args:
            sh1: InfCylinder shape
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
                "Cg2_InfCylinder_Sphere_L6Geom does not handle periodic boundary conditions with skew (Scene.cell.trsf is not diagonal)."
            )

        cyl = sh1
        sphere = sh2

        assert cyl.numNodesOk()
        assert sphere.numNodesOk()

        sphRad = sphere.radius
        cylRad = cyl.radius
        ax = cyl.axis
        cylPos = cyl.nodes[0].pos
        sphPos = sphere.nodes[0].pos + shift2

        # Calculate relative position, ignoring cylinder axis
        relPos = sphPos - cylPos
        relPos[ax] = 0.0

        # Check if objects are too far apart
        if not C.isReal() and relPos.dot(relPos) > (cylRad + sphRad) ** 2 and not force:
            return False

        # Calculate distance and normal
        dist = relPos.norm()

        # Handle edge case where sphere center is on cylinder axis
        if dist == 0.0:
            self.fatal(
                f"dist==0.0 between InfCylinder #{C.leakPA().id} @ {cyl.nodes[0].pos}, r={cylRad} "
                f"and Sphere #{C.leakPB().id} @ {sphere.nodes[0].pos}, r={sphere.radius}"
            )

        # Calculate penetration depth and normal
        uN = dist - (cylRad + sphRad)
        normal = relPos / dist

        # Calculate contact point
        cylPosAx = cylPos.copy()
        cylPosAx[ax] = sphPos[ax]  # Project onto sphere's axis position
        contPt = cylPosAx + (cylRad + 0.5 * uN) * normal

        # Get dynamics data
        cylDyn = cyl.nodes[0].getDataTyped(DEMData)
        sphDyn = sphere.nodes[0].getDataTyped(DEMData)

        # Check impossible rotations of the infinite cylinder
        assert cylDyn.angVel[(ax + 1) % 3] == 0.0 and cylDyn.angVel[(ax + 2) % 3] == 0.0

        # Handle contact like sphere-sphere contact
        self.handleSpheresLikeContact(
            C,
            cylPos,
            cylDyn.vel,
            cylDyn.angVel,
            sphPos,
            sphDyn.vel,
            sphDyn.angVel,
            normal,
            contPt,
            uN,
            cylRad,
            sphRad,
        )

        return True

    def goReverse(self, sh1, sh2, shift2, force, C):
        """Handle reversed contact arguments."""
        raise RuntimeError(
            "ContactLoop should swap interaction arguments, should be InfCylinder+Sphere, but is "
            + sh1.__class__.__name__
            + "+"
            + sh2.__class__.__name__
        )
