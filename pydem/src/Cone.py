#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from pydem.src.Shape import Shape
from pydem.src.Functor import BoundFunctor
from pydem.src.ContactGeomFunctor import Cg2_Any_Any_L6Geom__Base
from pydem.src.Aabb import Aabb
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import (
    Vector3r,
    Vector2r,
    Matrix3r,
    Quaternionr,
    AngleAxisr,
    Real,
    INF,
    NAN,
)
from pydem.src.DEMData import DEMData
from pydem.src.Sphere import Sphere
from pydem.src.Node import Node


class Cone(Shape):
    """
    Line element without internal forces, with circular cross-section and hemi-spherical caps at both ends.

    Geometrically the same as Capsule, but with 2 nodes.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.radii = Vector2r(
            NAN, NAN
        )  # Radii of the cone (can be different at each end)

    def numNodes(self):
        """Return number of nodes needed by this shape."""
        return 2


@DEM_LOGGER
class Bo1_Cone_Aabb(BoundFunctor):
    """
    Compute Aabb of a Cone particle.

    This functor computes the axis-aligned bounding box for a cone.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for a Cone.

        Args:
            sh: Cone shape
        """
        assert sh.numNodesOk()

        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "Bo1_Cone_Aabb::go: sheared periodic boundaries for Cone not (yet) implemented."
            )

        c = sh

        # Create bound if it doesn't exist
        if not c.bound:
            c.bound = Aabb()
            c.bound.maxRot = -1  # Ignore node rotation

        aabb = c.bound

        # Calculate axis unit vector
        axUnit = (c.nodes[1].pos - c.nodes[0].pos).normalized()

        # Initialize AABB with first point
        for ax in range(3):
            for end in range(2):
                r = c.radii[end]
                # Calculate maximum deviation from axis in this direction
                dAx = r * np.linalg.norm(np.cross(axUnit, Vector3r.unit(ax)))

                for dir in [-1, 1]:
                    P = c.nodes[end].pos[ax] + dir * dAx

                    if end == 0 and dir == -1:
                        # Initialize with first point
                        aabb.min[ax] = aabb.max[ax] = P
                    else:
                        # Expand AABB
                        aabb.min[ax] = min(P, aabb.min[ax])
                        aabb.max[ax] = max(P, aabb.max[ax])


@DEM_LOGGER
class Cg2_Cone_Sphere_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between Cone and Sphere.

    Uses attributes of Cg2_Sphere_Sphere_L6Geom.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Cone and Sphere.

        Args:
            sh1: Cone shape
            sh2: Sphere shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        c = sh1
        s = sh2

        assert sh1.numNodesOk()
        assert sh2.numNodesOk()
        assert sh1.nodes[0].hasDataTyped(DEMData)
        assert sh2.nodes[0].hasDataTyped(DEMData)

        dynCone = c.nodes[0].getDataTyped(DEMData)
        dynSphere = s.nodes[0].getDataTyped(DEMData)

        A = c.nodes[0].pos
        B = c.nodes[1].pos
        S = s.nodes[0].pos + shift2

        AB1 = (B - A).normalized()

        # Perpendicular to the axis-sphere plane (x-perp)
        perpX1 = ((S - A).cross(B - A)).normalized()

        # Perpendicular to the axis, in the axis-sphere plane
        perpA1 = AB1.cross(perpX1)  # Both normalized and perpendicular

        contPt = Vector3r(0, 0, 0)
        coneContRad = 0.0
        inside = False  # Sphere center is inside the cone

        # For degenerate case (exactly on the axis), the norm will be zero
        if perpX1.norm_squared() > 0:  # OK
            P = A + perpA1 * c.radii[0]
            Q = B + perpA1 * c.radii[1]

            # Find closest point on side line to sphere center
            normSidePos = 0.0
            sidePt = self.closestSegmentPt(S, P, Q, normSidePos)

            # Radius interpolated along the side
            coneContRad = (1 - normSidePos) * c.radii[0] + normSidePos * c.radii[1]

            # Touches rim or base
            if normSidePos == 0 or normSidePos == 1:
                E = A if normSidePos == 0 else B
                axDist = (S - E).dot(perpA1)

                # Touches the rim
                if axDist >= c.radii[0 if normSidePos == 0 else 1]:
                    contPt = sidePt
                # Touches the base
                else:
                    contPt = E + perpA1 * axDist

                dir = -1 if normSidePos == 0 else 1
                if (S - E).dot(AB1 * dir) < 0:
                    inside = True
            # Touches the side
            else:
                outerVec = (Q - P).cross(perpX1)
                if (S - sidePt).dot(outerVec) <= 0:
                    inside = True
                contPt = sidePt
        else:
            # Degenerate case
            self.error(
                "Degenerate cone-sphere intersection (exactly on the axis) not handled yet!"
            )
            raise RuntimeError("Degenerate cone-sphere intersection.")

        unDistSq = (S - contPt).norm_squared() - s.radius**2

        if not inside and unDistSq > 0 and not C.isReal() and not force:
            return False

        uN = (inside and -1 or 1) * (S - contPt).norm() - s.radius
        normal = (inside and -1 or 1) * (S - contPt).normalized()
        coneAxisPt = A + AB1 * AB1.dot(contPt - A)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            coneAxisPt,
            dynCone.vel,
            dynCone.angVel,
            S,
            dynSphere.vel,
            dynSphere.angVel,
            normal,
            contPt,
            uN,
            coneContRad,
            s.radius,
        )

        return True

    def closestSegmentPt(self, p, a, b, t=None):
        """
        Find closest point on segment to a point.

        Args:
            p: Point to find closest to
            a: Segment start
            b: Segment end
            t: Optional parameter to store relative position

        Returns:
            Closest point on segment
        """
        ab = b - a
        ap = p - a

        # Calculate projection
        ab_dot = ab.dot(ab)
        if ab_dot < 1e-10:
            if t is not None:
                t = 0.0
            return a

        t_val = ap.dot(ab) / ab_dot
        t_val = max(0.0, min(1.0, t_val))

        if t is not None:
            t = t_val

        return a + t_val * ab


# # Register classes for dispatching
# Cone.registerClass()
# Bo1_Cone_Aabb.registerClass()
# Cg2_Cone_Sphere_L6Geom.registerClass()
