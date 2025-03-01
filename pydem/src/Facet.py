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
from pydem.src.InfCylinder import InfCylinder
from pydem.src.Node import Node
from pydem.src.utils import CompUtils


class Facet(Shape):
    """
    Facet (triangle in 3d) particle.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.fakeVel = Vector3r(0, 0, 0)  # Fake velocity when computing contact
        self.halfThick = 0.0  # Geometric thickness (added in all directions)
        self.n21lim = Vector3r(NAN, NAN, NAN)  # Edge & vertex contact limit

    def numNodes(self):
        """Return number of nodes needed by this shape."""
        return 3

    def selfTest(self, p):
        """
        Perform self-test to verify shape validity.

        Args:
            p: Particle using this shape
        """
        if not self.numNodesOk():
            raise RuntimeError(
                f"Facet #{p.getId()}: numNodesOk() failed (has {len(self.nodes)} nodes)"
            )

        for i in range(3):
            if (self.nodes[i].pos - self.nodes[(i + 1) % 3].pos).norm_squared() == 0:
                raise RuntimeError(
                    f"Facet #{p.getId()}: nodes {i} and {(i + 1) % 3} are coincident."
                )

        # Check for contacts with other facets
        ffCon = 0
        for c in p.contacts.values():
            if c.isReal() and isinstance(c.leakOther(p).shape, Facet):
                ffCon += 1

        if ffCon > 0:
            self.warning(
                f"Facet.selfTest: Facet #{p.getId()} has {ffCon} contacts with other facets. "
                "This is not per se an error though very likely unintended -- there is no "
                "functor to handle such contact and it will be uselessly recomputed in "
                "every step. Set both particles masks to have some DemField.loneMask "
                "bits and contact will not be created at all."
            )

        super().selfTest(p)

    def getNormal(self):
        """Return normal vector of the facet."""
        assert self.numNodesOk()
        return (
            (self.nodes[1].pos - self.nodes[0].pos).cross(
                self.nodes[2].pos - self.nodes[0].pos
            )
        ).normalized()

    def getCentroid(self):
        """Return centroid of the facet."""
        return (1 / 3.0) * (self.nodes[0].pos + self.nodes[1].pos + self.nodes[2].pos)

    def getArea(self):
        """Return surface area of the facet."""
        assert self.numNodesOk()
        A = self.nodes[0].pos
        B = self.nodes[1].pos
        C = self.nodes[2].pos
        return 0.5 * ((B - A).cross(C - A)).norm()

    def getPerimeterSq(self):
        """Return squared perimeter of the facet."""
        assert self.numNodesOk()
        return (
            (self.nodes[1].pos - self.nodes[0].pos).norm_squared()
            + (self.nodes[2].pos - self.nodes[1].pos).norm_squared()
            + (self.nodes[0].pos - self.nodes[2].pos).norm_squared()
        )

    def getOuterVectors(self):
        """Return outer vectors of the facet."""
        assert self.numNodesOk()
        # Not normalized
        nn = (self.nodes[1].pos - self.nodes[0].pos).cross(
            self.nodes[2].pos - self.nodes[0].pos
        )
        return (
            (self.nodes[1].pos - self.nodes[0].pos).cross(nn),
            (self.nodes[2].pos - self.nodes[1].pos).cross(nn),
            (self.nodes[0].pos - self.nodes[2].pos).cross(nn),
        )

    def outerEdgeNormals(self):
        """Return outer edge normal vectors."""
        o = self.getOuterVectors()
        return [o[0].normalized(), o[1].normalized(), o[2].normalized()]

    def getNearestPt(self, pt):
        """
        Find closest point on the facet to a given point.

        Args:
            pt: Point to find closest to

        Returns:
            Closest point on facet
        """
        # Project point to facet's plane
        fNormal = self.getNormal()
        planeDist = (pt - self.nodes[0].pos).dot(fNormal)
        fC = pt - planeDist * fNormal

        # Get outer vectors
        outVec = self.getOuterVectors()

        # Check which region the point is in
        w = 0
        for i in range(3):
            if outVec[i].dot(fC - self.nodes[i].pos) > 0:
                w |= 1 << i

        # Return appropriate point based on region
        if w == 0:
            return fC  # Inside triangle
        elif w == 1:
            return CompUtils.closestSegmentPt(
                fC, self.nodes[0].pos, self.nodes[1].pos
            )  # Edge 0-1
        elif w == 2:
            return CompUtils.closestSegmentPt(
                fC, self.nodes[1].pos, self.nodes[2].pos
            )  # Edge 1-2
        elif w == 4:
            return CompUtils.closestSegmentPt(
                fC, self.nodes[2].pos, self.nodes[0].pos
            )  # Edge 2-0
        elif w == 3:
            return self.nodes[1].pos  # Vertex 1
        elif w == 5:
            return self.nodes[0].pos  # Vertex 0
        elif w == 6:
            return self.nodes[2].pos  # Vertex 2
        elif w == 7:
            raise RuntimeError(
                "Facet::getNearestPt: Impossible sphere-facet intersection (all points are outside the edges)."
            )
        else:
            raise RuntimeError("Facet::getNearestPt: Nonsense intersection value.")

    def interpolatePtLinAngVel(self, x):
        """
        Interpolate linear and angular velocity at a point.

        Args:
            x: Point to interpolate at

        Returns:
            Tuple of (linear velocity, angular velocity)
        """
        assert self.numNodesOk()

        # Special case for NaN in fakeVel.x
        if np.isnan(self.fakeVel[0]):
            return (Vector3r(0, 0, 0), Vector3r(0, 0, 0))

        # Calculate barycentric coordinates
        a = CompUtils.triangleBarycentrics(
            x, self.nodes[0].pos, self.nodes[1].pos, self.nodes[2].pos
        )

        # Get velocities of nodes
        vv = [
            self.nodes[0].getDataTyped(DEMData).vel,
            self.nodes[1].getDataTyped(DEMData).vel,
            self.nodes[2].getDataTyped(DEMData).vel,
        ]

        # Get angular velocities of nodes
        ww = [
            self.nodes[0].getDataTyped(DEMData).angVel,
            self.nodes[1].getDataTyped(DEMData).angVel,
            self.nodes[2].getDataTyped(DEMData).angVel,
        ]

        # Interpolate velocities
        ret = Vector3r(0, 0, 0)
        for i in range(3):
            ret += a[i] * vv[i]

        # Add fake velocity
        ret += self.fakeVel

        # Calculate angular velocity
        angVel = Vector3r(0, 0, 0)
        for i in range(3):
            angVel += a[i] * ww[i]

        return (ret, angVel)

    def computeNeighborAngles(self):
        """
        Compute n21Min using neighboring facets.

        This allows contact direction to be adjusted.
        """
        # Implementation would depend on how neighboring facets are stored
        # This is a placeholder for the actual implementation
        pass

    def lumpMassInertia(self, n, density, mass, I, rotateOk):
        """
        Compute mass and inertia for this shape.

        Args:
            n: Node to compute mass for
            density: Material density
            mass: Mass value to update
            I: Inertia tensor to update
            rotateOk: Whether rotation is allowed

        Returns:
            Updated mass, inertia tensor, and rotateOk flag
        """
        self.checkNodesHaveDemData()
        rotateOk = True

        if not (self.halfThick > 0):
            return mass, I, rotateOk

        # Check if node belongs to facet
        if n not in self.nodes:
            return mass, I, rotateOk

        ix = self.nodes.index(n)

        # Calculate midpoint local coords
        vv = [
            0.5 * n.glob2loc(self.nodes[(ix + 1) % 3].pos),
            0.5 * n.glob2loc(self.nodes[(ix + 2) % 3].pos),
        ]

        C = n.glob2loc(self.getCentroid())

        # Calculate inertia contribution
        # Note: This is a simplified version - the actual calculation would use
        # the triangleInertia and triangleArea functions from Volumetric
        I_contrib = (
            density
            * (2 * self.halfThick)
            * self.calculateTriangleInertia(Vector3r(0, 0, 0), vv[0], vv[1])
        )
        I += I_contrib

        I_contrib2 = (
            density
            * (2 * self.halfThick)
            * self.calculateTriangleInertia(vv[0], vv[1], C)
        )
        I += I_contrib2

        # Calculate mass contribution
        mass_contrib = (
            density
            * (2 * self.halfThick)
            * self.calculateTriangleArea(Vector3r(0, 0, 0), vv[0], vv[1])
        )
        mass += mass_contrib

        mass_contrib2 = (
            density * (2 * self.halfThick) * self.calculateTriangleArea(vv[0], vv[1], C)
        )
        mass += mass_contrib2

        return mass, I, rotateOk

    def calculateTriangleInertia(self, a, b, c):
        """
        Calculate inertia tensor of a triangle.

        Args:
            a, b, c: Triangle vertices

        Returns:
            Inertia tensor
        """
        # Simplified calculation - would need proper implementation
        area = self.calculateTriangleArea(a, b, c)
        centroid = (a + b + c) / 3.0

        I = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                if i == j:
                    I[i, i] = area * (centroid.norm_squared() - centroid[i] ** 2)
                else:
                    I[i, j] = -area * centroid[i] * centroid[j]

        return I

    def calculateTriangleArea(self, a, b, c):
        """
        Calculate area of a triangle.

        Args:
            a, b, c: Triangle vertices

        Returns:
            Area of triangle
        """
        return 0.5 * ((b - a).cross(c - a)).norm()

    def asRaw(self, center, radius, nn, raw):
        """
        Convert to raw representation.

        Args:
            center: Center position (output)
            radius: Bounding radius (output)
            nn: Nodes list
            raw: Raw data (output)
        """
        # Calculate circumscribed circle center
        center = self.calculateCircumscribedCircleCenter(
            self.nodes[0].pos, self.nodes[1].pos, self.nodes[2].pos
        )

        # Calculate radius
        radius = (self.nodes[0].pos - center).norm()

        # Store nodal positions as raw data
        raw.resize(10)
        for i in range(3):
            for ax in range(3):
                raw[3 * i + ax] = self.nodes[i].pos[ax]

        raw[9] = self.halfThick

    def setFromRaw(self, center, radius, nn, raw):
        """
        Set from raw representation.

        Args:
            center: Center position
            radius: Bounding radius
            nn: Nodes list
            raw: Raw data
        """
        if len(raw) != 10:
            raise RuntimeError(
                f"Facet::setFromRaw: expected 10 raw values, got {len(raw)}"
            )

        # Create nodes if needed
        if len(self.nodes) != 3:
            self.nodes = [None, None, None]

        # Set node positions
        for i in range(3):
            if i >= len(nn):
                nn.append(Node())
            self.nodes[i] = nn[i]
            for ax in range(3):
                self.nodes[i].pos[ax] = raw[3 * i + ax]

        self.halfThick = raw[9]

    def calculateCircumscribedCircleCenter(self, a, b, c):
        """
        Calculate center of circumscribed circle.

        Args:
            a, b, c: Triangle vertices

        Returns:
            Center of circumscribed circle
        """
        # Vectors from a to b and a to c
        ab = b - a
        ac = c - a

        # Cross product of ab and ac
        abXac = ab.cross(ac)

        # Calculate center
        toCircumcenter = (ab.norm_squared() * ac - ac.norm_squared() * ab).cross(
            abXac
        ) / (2 * abXac.norm_squared())

        return a + toCircumcenter


@DEM_LOGGER
class Bo1_Facet_Aabb(BoundFunctor):
    """
    Creates/updates an Aabb of a Facet.

    This functor computes the axis-aligned bounding box for a facet.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for a Facet.

        Args:
            sh: Facet shape
        """
        f = sh

        # Create bound if it doesn't exist
        if not f.bound:
            f.bound = Aabb()
            f.bound.maxRot = -1  # Ignore node rotation

        aabb = f.bound
        halfThickVec = Vector3r(f.halfThick, f.halfThick, f.halfThick)

        if not self.scene.isPeriodic:
            # Initialize with first node
            aabb.min = f.nodes[0].pos - halfThickVec
            aabb.max = f.nodes[0].pos + halfThickVec

            # Expand for other nodes
            for i in range(1, 3):
                for ax in range(3):
                    aabb.min[ax] = min(
                        aabb.min[ax], f.nodes[i].pos[ax] - halfThickVec[ax]
                    )
                    aabb.max[ax] = max(
                        aabb.max[ax], f.nodes[i].pos[ax] + halfThickVec[ax]
                    )
        else:
            # Periodic cell: unshear everything
            aabb.min = self.scene.cell.unshearPt(f.nodes[0].pos) - halfThickVec
            aabb.max = self.scene.cell.unshearPt(f.nodes[0].pos) + halfThickVec

            for i in range(1, 3):
                v = self.scene.cell.unshearPt(f.nodes[i].pos)
                for ax in range(3):
                    aabb.min[ax] = min(aabb.min[ax], v[ax] - halfThickVec[ax])
                    aabb.max[ax] = max(aabb.max[ax], v[ax] + halfThickVec[ax])


@DEM_LOGGER
class In2_Facet(IntraFunctor):
    """
    Distribute force on Facet to its nodes.

    The algorithm is purely geometrical since Facet has no internal forces,
    therefore can accept any kind of material.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def addIntraStiffnesses(self, p, n, ktrans, krot):
        """
        Add internal stiffness contributions to a node.

        Args:
            p: Particle
            n: Node
            ktrans: Translational stiffness vector (modified in-place)
            krot: Rotational stiffness vector (modified in-place)
        """
        # No internal stiffness in Facets
        pass

    def go(self, sh, m, p):
        """
        Apply internal forces.

        Args:
            sh: Shape
            m: Material
            p: Particle
        """
        if not p.contacts:
            return

        self.distributeForces(p, sh, True)

    def distributeForces(self, particle, f, bary):
        """
        Distribute forces to nodes.

        Args:
            particle: Particle
            f: Facet shape
            bary: Whether to use barycentric coordinates
        """
        normal = f.getNormal() if bary else Vector3r(0, 0, 0)

        for C in particle.contacts.values():
            if not C.isReal():
                continue

            F, T, xc = Vector3r(0, 0, 0), Vector3r(0, 0, 0), Vector3r(0, 0, 0)
            weights = Vector3r(0, 0, 0)

            if bary:
                # Find barycentric coordinates of the projected contact point
                c = C.geom.node.pos
                p = c - (c - f.nodes[0].pos).dot(normal) * normal
                weights = CompUtils.triangleBarycentrics(
                    p, f.nodes[0].pos, f.nodes[1].pos, f.nodes[2].pos
                )
            else:
                # Distribute equally
                weights = Vector3r(1 / 3.0, 1 / 3.0, 1 / 3.0)

            # Apply forces to nodes
            for i in range(3):
                F, T, xc = C.getForceTorqueBranch(particle, i, self.scene)
                F *= weights[i]
                T *= weights[i]
                f.nodes[i].getDataTyped(DEMData).addForceTorque(F, xc.cross(F) + T)


@DEM_LOGGER
class Cg2_Facet_Sphere_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between Facet and Sphere.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Facet and Sphere.

        Args:
            sh1: Facet shape
            sh2: Sphere shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        f = sh1
        s = sh2

        sC = s.nodes[0].pos + shift2
        fNormal = f.getNormal()
        planeDist = (sC - f.nodes[0].pos).dot(fNormal)

        # Check if too far
        if abs(planeDist) > (s.radius + f.halfThick) and not C.isReal() and not force:
            return False

        # Project sphere center to facet plane
        fC = sC - planeDist * fNormal

        # Get outer vectors
        outVec = f.getOuterVectors()

        # Check which region the point is in
        ll = [0, 0, 0]
        for i in range(3):
            ll[i] = outVec[i].dot(fC - f.nodes[i].pos)

        w = (1 if ll[0] > 0 else 0) + (2 if ll[1] > 0 else 0) + (4 if ll[2] > 0 else 0)

        # Find contact point based on region
        if w == 0:
            contPt = fC  # Inside triangle
        elif w == 1:
            contPt = CompUtils.closestSegmentPt(
                fC, f.nodes[0].pos, f.nodes[1].pos
            )  # Edge 0-1
        elif w == 2:
            contPt = CompUtils.closestSegmentPt(
                fC, f.nodes[1].pos, f.nodes[2].pos
            )  # Edge 1-2
        elif w == 4:
            contPt = CompUtils.closestSegmentPt(
                fC, f.nodes[2].pos, f.nodes[0].pos
            )  # Edge 2-0
        elif w == 3:
            contPt = f.nodes[1].pos  # Vertex 1
        elif w == 5:
            contPt = f.nodes[0].pos  # Vertex 0
        elif w == 6:
            contPt = f.nodes[2].pos  # Vertex 2
        elif w == 7:
            raise RuntimeError(
                "Cg2_Facet_Sphere_L6Geom: Impossible sphere-facet intersection."
            )
        else:
            raise RuntimeError("Cg2_Facet_Sphere_L6Geom: Nonsense intersection value.")

        # Calculate normal and penetration depth
        normal = sC - contPt

        if (
            normal.norm_squared() > (s.radius + f.halfThick) ** 2
            and not C.isReal()
            and not force
        ):
            return False

        dist = normal.norm()
        if dist == 0:
            self.fatal(
                f"dist==0.0 between Facet #{C.leakPA().id} and Sphere #{C.leakPB().id}"
            )

        normal = normal / dist
        uN = dist - (s.radius + f.halfThick)

        # Get dynamics data
        sphereDyn = s.nodes[0].getDataTyped(DEMData)

        # Get facet velocity at contact point
        facetVel, facetAngVel = f.interpolatePtLinAngVel(contPt)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            contPt,
            facetVel,
            facetAngVel,
            sC,
            sphereDyn.vel,
            sphereDyn.angVel,
            normal,
            contPt + 0.5 * uN * normal,
            uN,
            f.halfThick,
            s.radius,
        )

        return True

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = -1


@DEM_LOGGER
class Cg2_Facet_Facet_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between two Facet shapes.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between two Facets.

        Args:
            sh1: First Facet shape
            sh2: Second Facet shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # This is a placeholder - facet-facet collision is complex
        # and would need a more detailed implementation
        self.warning("Facet-Facet collision not fully implemented")
        return False

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = -1


@DEM_LOGGER
class Cg2_Facet_InfCylinder_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact between Facet and InfCylinder.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Facet and InfCylinder.

        Args:
            sh1: Facet shape
            sh2: InfCylinder shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # This is a placeholder - facet-infcylinder collision is complex
        # and would need a more detailed implementation
        self.warning("Facet-InfCylinder collision not fully implemented")
        return False

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = -1


# # Register classes for dispatching
# Facet.registerClass()
# Bo1_Facet_Aabb.registerClass()
# In2_Facet.registerClass()
# Cg2_Facet_Sphere_L6Geom.registerClass()
# Cg2_Facet_Facet_L6Geom.registerClass()
# Cg2_Facet_InfCylinder_L6Geom.registerClass()
