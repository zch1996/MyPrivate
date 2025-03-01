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
    AlignedBox3r,
    Real,
    INF,
    NAN,
)
from pydem.src.DEMData import DEMData
from pydem.src.Sphere import Sphere
from pydem.src.Wall import Wall
from pydem.src.Facet import Facet
from pydem.src.InfCylinder import InfCylinder
from pydem.src.Node import Node


class Capsule(Shape):
    """
    Cylinder with half-spherical caps on both sides, Minkowski sum of segment with sphere.

    This shape represents a capsule formed by a cylindrical shaft with
    hemispherical caps on both ends.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.radius = NAN  # Radius of the capsule (caps and middle part)
        self.shaft = NAN  # Length of the middle segment

    def numNodes(self):
        """Return number of nodes needed by this shape."""
        return 1

    def selfTest(self, p):
        """
        Perform self-test to verify shape validity.

        Args:
            p: Particle using this shape
        """
        if not (self.radius > 0) or not (self.shaft >= 0):
            raise RuntimeError(
                f"Capsule #{p.getId()}: radius must be positive and shaft non-negative "
                f"(current: radius={self.radius}, shaft={self.shaft})."
            )

        if not self.numNodesOk():
            raise RuntimeError(
                f"Capsule #{p.getId()}: numNodesOk() failed: must be 1, not {len(self.nodes)}."
            )

        super().selfTest(p)

    def volume(self):
        """Calculate volume of the capsule."""
        return (
            4 / 3
        ) * math.pi * self.radius**3 + math.pi * self.radius**2 * self.shaft

    def equivRadius(self):
        """Calculate equivalent radius of the capsule."""
        return (self.volume() * 3 / (4 * math.pi)) ** (1 / 3)

    def isInside(self, pt):
        """
        Check if a point is inside the capsule.

        Args:
            pt: Point to check

        Returns:
            True if point is inside, False otherwise
        """
        # Transform to local coordinates
        l = self.nodes[0].glob2loc(pt)

        # Cylinder test
        if abs(l[0]) < self.shaft / 2:
            return l[1] ** 2 + l[2] ** 2 < self.radius**2

        # Cap test (distance to endpoint)
        endpoint = Vector3r(-self.shaft / 2 if l[0] < 0 else self.shaft / 2, 0, 0)
        return (endpoint - l).norm_squared() < self.radius**2

    def applyScale(self, scale):
        """
        Apply scaling to the capsule.

        Args:
            scale: Scale factor
        """
        self.radius *= scale
        self.shaft *= scale

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
        if n != self.nodes[0]:
            return mass, I, rotateOk  # Not our node

        rotateOk = False  # Node may not be rotated without geometry change
        self.checkNodesHaveDemData()

        r2 = self.radius**2
        r3 = self.radius**3

        # Mass of caps and shaft
        mCaps = (4 / 3) * math.pi * r3 * density
        mShaft = math.pi * r2 * self.shaft * density

        # Distance between centroid and the cap's centroid
        distCap = 0.5 * self.shaft + (3 / 8) * self.radius

        # Moments of inertia
        Ix = 2 * mCaps * r2 / 5 + 0.5 * mShaft * r2
        Iy = (
            (83 / 320) * mCaps * r2
            + mCaps * distCap**2
            + (1 / 12) * mShaft * (3 * r2 + self.shaft**2)
        )

        # Update inertia tensor
        I[0, 0] += Ix
        I[1, 1] += Iy
        I[2, 2] += Iy

        # Update mass
        mass += mCaps + mShaft

        return mass, I, rotateOk

    def alignedBox(self):
        """
        Compute axis-aligned bounding box.

        Returns:
            AlignedBox3r: Axis-aligned bounding box
        """
        pos = self.nodes[0].pos
        ori = self.nodes[0].ori
        dShaft = ori.rotate(Vector3r(0.5 * self.shaft, 0, 0))

        ret = AlignedBox3r()
        for a in [-1, 1]:
            for b in [-1, 1]:
                ret.extend(pos + a * dShaft + b * self.radius * Vector3r(1, 1, 1))

        return ret

    def asRaw(self, center, radius, nn, raw):
        """
        Convert to raw representation.

        Args:
            center: Center position (output)
            radius: Bounding radius (output)
            nn: Nodes list
            raw: Raw data vector (output)
        """
        center = self.nodes[0].pos
        radius = self.radius + 0.5 * self.shaft

        # Store as half-shaft vector
        raw = [0, 0, 0]
        hShaft = self.nodes[0].ori.rotate(Vector3r(0.5 * self.shaft, 0, 0))
        raw[0] = hShaft[0]
        raw[1] = hShaft[1]
        raw[2] = hShaft[2]

        return center, radius, nn, raw

    def setFromRaw(self, center, radius, nn, raw):
        """
        Set from raw representation.

        Args:
            center: Center position
            radius: Bounding radius
            nn: Nodes list
            raw: Raw data vector
        """
        self.setFromRaw_helper_checkRaw_makeNodes(raw, 3)

        hShaft = Vector3r(raw[0], raw[1], raw[2])
        self.nodes[0].pos = center
        self.shaft = 2 * hShaft.norm()
        self.radius = radius - 0.5 * self.shaft

        if self.shaft > 0:
            self.nodes[0].ori.setFromTwoVectors(Vector3r(1, 0, 0), hShaft.normalized())
        else:
            # Zero shaft, orientation doesn't matter
            self.nodes[0].ori = Quaternionr(1, 0, 0, 0)

    def endPt(self, i):
        """
        Return one of capsule endpoints.

        Args:
            i: Index (0 for negative end, 1 for positive end)

        Returns:
            Vector3r: Endpoint position
        """
        return self.nodes[0].loc2glob(
            Vector3r((-0.5 if i == 0 else 0.5) * self.shaft, 0, 0)
        )


@DEM_LOGGER
class Bo1_Capsule_Aabb(BoundFunctor):
    """
    Creates/updates an Aabb of a Capsule.

    This functor computes the axis-aligned bounding box for a capsule.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for a Capsule.

        Args:
            sh: Capsule shape
        """
        cap = sh

        # Create bound if it doesn't exist
        if not cap.bound:
            cap.bound = Aabb()

        assert cap.numNodesOk()
        aabb = cap.bound

        # Get capsule properties
        pos = cap.nodes[0].pos
        ori = cap.nodes[0].ori
        radius = cap.radius
        shaft = cap.shaft

        # Compute half-shaft vector
        hShaft = ori.rotate(Vector3r(0.5 * shaft, 0, 0))

        # Compute endpoints
        A = pos - hShaft
        B = pos + hShaft

        # Set AABB dimensions
        aabb.min = Vector3r.min(A, B) - Vector3r(radius, radius, radius)
        aabb.max = Vector3r.max(A, B) + Vector3r(radius, radius, radius)

        # Set rotation parameters
        if shaft > 0:
            maxArm = max(radius, 0.5 * shaft)
            aabb.maxRot = math.atan2(radius, maxArm)
        else:
            aabb.maxRot = math.pi  # Sphere can rotate freely


@DEM_LOGGER
class Cg2_Capsule_Capsule_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Collision of two Capsule shapes.

    This functor handles collision detection and geometry computation
    between two capsules.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between two Capsules.

        Args:
            sh1: First Capsule shape
            sh2: Second Capsule shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        c1 = sh1
        c2 = sh2

        assert c1.numNodesOk()
        assert c2.numNodesOk()

        # Get properties
        pos1 = c1.nodes[0].pos
        ori1 = c1.nodes[0].ori
        rad1 = c1.radius
        shaft1 = c1.shaft
        dyn1 = c1.nodes[0].getDataTyped(DEMData)

        pos2 = c2.nodes[0].pos + shift2
        ori2 = c2.nodes[0].ori
        rad2 = c2.radius
        shaft2 = c2.shaft
        dyn2 = c2.nodes[0].getDataTyped(DEMData)

        # Calculate capsule segment endpoints
        A1 = pos1 - ori1.rotate(Vector3r(0.5 * shaft1, 0, 0))
        B1 = pos1 + ori1.rotate(Vector3r(0.5 * shaft1, 0, 0))

        A2 = pos2 - ori2.rotate(Vector3r(0.5 * shaft2, 0, 0))
        B2 = pos2 + ori2.rotate(Vector3r(0.5 * shaft2, 0, 0))

        # Find closest points between segments
        s = Vector3r(0, 0, 0)
        t = Vector3r(0, 0, 0)

        p1, p2 = self.closestSegmentSegment(A1, B1, A2, B2, s, t)

        # Calculate distance and normal
        dist_vec = p2 - p1
        dist = dist_vec.norm()

        # Check if objects are too far apart
        if not C.isReal() and dist > (rad1 + rad2) and not force:
            return False

        # Calculate normal and penetration depth
        normal = dist_vec / dist if dist > 0 else Vector3r(1, 0, 0)
        uN = dist - (rad1 + rad2)

        # Calculate contact point
        contPt = p1 + (rad1 + 0.5 * uN) * normal

        # Calculate velocities at contact points
        vel1 = dyn1.vel + dyn1.angVel.cross(p1 - pos1)
        vel2 = dyn2.vel + dyn2.angVel.cross(p2 - pos2)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            p1,
            vel1,
            dyn1.angVel,
            p2,
            vel2,
            dyn2.angVel,
            normal,
            contPt,
            uN,
            rad1,
            rad2,
        )

        return True

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = (s1.nodes[0].pos - s2.nodes[0].pos).norm_squared()

    def closestSegmentSegment(self, p1, q1, p2, q2, s, t):
        """
        Find closest points between two line segments.

        Args:
            p1, q1: First segment endpoints
            p2, q2: Second segment endpoints
            s, t: Parameters to store relative positions

        Returns:
            Tuple of closest points on each segment
        """
        d1 = q1 - p1  # Direction vector of segment S1
        d2 = q2 - p2  # Direction vector of segment S2
        r = p1 - p2

        a = d1.dot(d1)  # Squared length of segment S1
        e = d2.dot(d2)  # Squared length of segment S2
        f = d2.dot(r)

        # Check if either segment is a point
        if a <= 1e-10 and e <= 1e-10:
            s = 0.0
            t = 0.0
            return p1, p2

        if a <= 1e-10:
            s = 0.0
            t = f / e
            t = max(0.0, min(1.0, t))
        else:
            c = d1.dot(r)

            if e <= 1e-10:
                t = 0.0
                s = -c / a
                s = max(0.0, min(1.0, s))
            else:
                b = d1.dot(d2)
                denom = a * e - b * b

                if denom != 0.0:
                    s = (b * f - c * e) / denom
                    s = max(0.0, min(1.0, s))
                else:
                    s = 0.0

                t = (b * s + f) / e

                if t < 0.0:
                    t = 0.0
                    s = -c / a
                    s = max(0.0, min(1.0, s))
                elif t > 1.0:
                    t = 1.0
                    s = (b - c) / a
                    s = max(0.0, min(1.0, s))

        return p1 + s * d1, p2 + t * d2


@DEM_LOGGER
class Cg2_Sphere_Capsule_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of Capsule and Sphere.

    This functor handles collision detection and geometry computation
    between a capsule and a sphere.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Sphere and Capsule.

        Args:
            sh1: Sphere shape
            sh2: Capsule shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        sphere = sh1
        cap = sh2

        assert sphere.numNodesOk()
        assert cap.numNodesOk()

        # Get properties
        sphPos = sphere.nodes[0].pos
        sphRad = sphere.radius
        sphDyn = sphere.nodes[0].getDataTyped(DEMData)

        capPos = cap.nodes[0].pos + shift2
        capOri = cap.nodes[0].ori
        capRad = cap.radius
        capShaft = cap.shaft
        capDyn = cap.nodes[0].getDataTyped(DEMData)

        # Calculate capsule segment endpoints
        A = capPos - capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))
        B = capPos + capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))

        # Find closest point on segment to sphere center
        AB = B - A
        t = (sphPos - A).dot(AB) / AB.dot(AB) if AB.dot(AB) > 0 else 0
        t = max(0.0, min(1.0, t))

        closestPt = A + t * AB

        # Calculate distance and normal
        dist_vec = sphPos - closestPt
        dist = dist_vec.norm()

        # Check if objects are too far apart
        if not C.isReal() and dist > (sphRad + capRad) and not force:
            return False

        # Calculate normal and penetration depth
        normal = dist_vec / dist if dist > 0 else Vector3r(1, 0, 0)
        uN = dist - (sphRad + capRad)

        # Calculate contact point
        contPt = closestPt + (capRad + 0.5 * uN) * normal

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            sphPos,
            sphDyn.vel,
            sphDyn.angVel,
            closestPt,
            capDyn.vel + capDyn.angVel.cross(closestPt - capPos),
            capDyn.angVel,
            normal,
            contPt,
            uN,
            sphRad,
            capRad,
        )

        return True

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = (s1.nodes[0].pos - s2.nodes[0].pos).norm_squared()


@DEM_LOGGER
class Cg2_Wall_Capsule_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of Capsule and Wall.

    This functor handles collision detection and geometry computation
    between a capsule and a wall.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Wall and Capsule.

        Args:
            sh1: Wall shape
            sh2: Capsule shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        wall = sh1
        cap = sh2

        assert wall.numNodesOk()
        assert cap.numNodesOk()

        # Get properties
        wallPos = wall.nodes[0].pos
        ax = wall.axis
        sense = wall.sense

        capPos = cap.nodes[0].pos + shift2
        capOri = cap.nodes[0].ori
        capRad = cap.radius
        capShaft = cap.shaft

        # Compute capsule segment endpoints
        A = capPos - capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))
        B = capPos + capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))

        # Calculate distance to wall
        dist = capPos[ax] - wallPos[ax]

        # Determine normal direction
        normal = Vector3r(0, 0, 0)
        normal[ax] = 1.0 if dist > 0 else -1.0

        # Check sense
        if sense != 0:
            if (sense > 0 and normal[ax] < 0) or (sense < 0 and normal[ax] > 0):
                return False

        # Find closest point on segment to wall
        if abs(normal[ax] * (B[ax] - A[ax])) < 1e-10:
            # Segment is parallel to wall, use midpoint
            closestPt = 0.5 * (A + B)
        else:
            # Find intersection of segment with wall plane
            t = (wallPos[ax] - A[ax]) / (B[ax] - A[ax])

            if t < 0 or t > 1:
                # Use endpoint closest to wall
                closestPt = (
                    A if abs(A[ax] - wallPos[ax]) < abs(B[ax] - wallPos[ax]) else B
                )
            else:
                # Use intersection point
                closestPt = A + t * (B - A)

        # Calculate penetration depth
        uN = normal[ax] * (closestPt[ax] - wallPos[ax]) - capRad

        # Check if objects are too far apart
        if not C.isReal() and uN > 0 and not force:
            return False

        # Calculate contact point
        contPt = closestPt.copy()
        contPt[ax] = wallPos[ax] + 0.5 * uN * normal[ax]

        # Get dynamics data
        wallDyn = wall.nodes[0].getDataTyped(DEMData)
        capDyn = cap.nodes[0].getDataTyped(DEMData)

        # Calculate velocity at closest point on capsule
        capVel = capDyn.vel + capDyn.angVel.cross(closestPt - capPos)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            wallPos,
            wallDyn.vel,
            wallDyn.angVel,
            closestPt,
            capVel,
            capDyn.angVel,
            normal,
            contPt,
            uN,
            -capRad,  # Negative radius for wall
            capRad,
        )

        return True


@DEM_LOGGER
class Cg2_InfCylinder_Capsule_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of Capsule and InfCylinder.

    This functor handles collision detection and geometry computation
    between a capsule and an infinite cylinder.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between InfCylinder and Capsule.

        Args:
            sh1: InfCylinder shape
            sh2: Capsule shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        cyl = sh1
        cap = sh2

        assert cyl.numNodesOk()
        assert cap.numNodesOk()

        # Get properties
        cylPos = cyl.nodes[0].pos
        cylRad = cyl.radius
        cylAx = cyl.axis

        capPos = cap.nodes[0].pos + shift2
        capOri = cap.nodes[0].ori
        capRad = cap.radius
        capShaft = cap.shaft

        # Compute capsule segment endpoints
        A = capPos - capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))
        B = capPos + capOri.rotate(Vector3r(0.5 * capShaft, 0, 0))

        # Project segment onto plane perpendicular to cylinder axis
        A_proj = A.copy()
        B_proj = B.copy()
        A_proj[cylAx] = B_proj[cylAx] = 0
        cylPos_proj = cylPos.copy()
        cylPos_proj[cylAx] = 0

        # Find closest point on projected segment to cylinder axis
        AB = B_proj - A_proj
        t = (cylPos_proj - A_proj).dot(AB) / AB.dot(AB) if AB.dot(AB) > 0 else 0
        t = max(0.0, min(1.0, t))

        closestPt_proj = A_proj + t * AB

        # Calculate distance and normal in projection
        dist_proj = (cylPos_proj - closestPt_proj).norm()

        # Check if objects are too far apart
        if not C.isReal() and dist_proj > (cylRad + capRad) and not force:
            return False

        # Calculate normal in projection
        normal_proj = (
            (cylPos_proj - closestPt_proj).normalized()
            if dist_proj > 0
            else Vector3r(1, 0, 0)
        )

        # Construct full 3D normal (perpendicular to cylinder axis)
        normal = normal_proj.copy()
        normal[cylAx] = 0

        # Find actual 3D closest point on capsule segment
        # This is more complex - we need to find the point on the capsule segment
        # that is closest to the cylinder, considering the 3D geometry

        # For simplicity, use the same t parameter from the projection
        closestPt = A + t * (B - A)

        # Calculate penetration depth
        uN = dist_proj - (cylRad + capRad)

        # Calculate contact point
        contPt = closestPt + (capRad + 0.5 * uN) * normal

        # Get dynamics data
        cylDyn = cyl.nodes[0].getDataTyped(DEMData)
        capDyn = cap.nodes[0].getDataTyped(DEMData)

        # Calculate velocity at closest point on capsule
        capVel = capDyn.vel + capDyn.angVel.cross(closestPt - capPos)

        # Calculate cylinder position at same height as contact point
        cylPos_at_contact = cylPos.copy()
        cylPos_at_contact[cylAx] = contPt[cylAx]

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            cylPos_at_contact,
            cylDyn.vel,
            cylDyn.angVel,
            closestPt,
            capVel,
            capDyn.angVel,
            normal,
            contPt,
            uN,
            cylRad,
            capRad,
        )

        return True


@DEM_LOGGER
class Cg2_Facet_Capsule_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of Capsule and Facet.

    This functor handles collision detection and geometry computation
    between a capsule and a facet.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Facet and Capsule.

        Args:
            sh1: Facet shape
            sh2: Capsule shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        facet = sh1
        cap = sh2

        # Check for periodic boundaries with shear
        if self.scene.isPeriodic and self.scene.cell.hasShear():
            raise RuntimeError(
                "Cg2_Facet_Capsule_L6Geom does not handle periodic boundary conditions with skew."
            )

        # Get capsule properties
        capNode = cap.nodes[0]
        capPos = capNode.pos + shift2
        capOri = capNode.ori
        capDyn = capNode.getDataTyped(DEMData)

        # Get facet normal
        fNormal = facet.getNormal()

        # Calculate capsule segment endpoints
        AB = [
            capPos - capOri.rotate(Vector3r(cap.shaft / 2, 0, 0)),
            capPos + capOri.rotate(Vector3r(cap.shaft / 2, 0, 0)),
        ]

        # Calculate distances from endpoints to facet plane
        f0pos = facet.nodes[0].pos
        planeDists = np.array(
            [(AB[0] - f0pos).dot(fNormal), (AB[1] - f0pos).dot(fNormal)]
        )

        # Check if capsule is too far from facet
        touchDist = facet.halfThick + cap.radius
        mayFail = not C.isReal() and not force

        if (
            mayFail
            and (np.sign(planeDists[0]) == np.sign(planeDists[1]))
            and np.min(np.abs(planeDists)) > touchDist
        ):
            return False

        # Find closest points on facet to segment endpoints
        ffp = [facet.getNearestPt(AB[0]), facet.getNearestPt(AB[1])]

        # Find points on segment closest to those facet points
        rp = np.zeros(2)
        ccp = [
            self.closestSegmentPt(ffp[0], AB[0], AB[1], rp[0]),
            self.closestSegmentPt(ffp[1], AB[0], AB[1], rp[1]),
        ]

        # Calculate squared distances
        fcd2 = np.array(
            [(ccp[0] - ffp[0]).norm_squared(), (ccp[1] - ffp[1]).norm_squared()]
        )

        # Check if too far from triangle
        if mayFail and np.min(fcd2) > touchDist**2:
            return False

        # Calculate distances
        fcd = np.sqrt(fcd2)

        # Calculate penetration depth
        uN = np.min(fcd) - touchDist

        # Determine normal vector and contact point
        normal = Vector3r(0, 0, 0)
        contPt = Vector3r(0, 0, 0)

        if np.max(fcd) > touchDist:
            # One point is beyond physical touch, use the closer one
            ix = np.argmin(fcd)
            normal = (ccp[ix] - ffp[ix]).normalized()
            contPt = ffp[ix] + (facet.halfThick + 0.5 * uN) * normal
        else:
            # Interpolate between points
            weights = fcd - touchDist
            weights = weights / np.sum(weights)

            normal = (
                (weights[0] * (ccp[0] - ffp[0]) + weights[1] * (ccp[1] - ffp[1]))
            ).normalized()
            contPt = (
                weights[0] * ffp[0]
                + weights[1] * ffp[1]
                + (facet.halfThick + 0.5 * uN) * normal
            )

        # Get facet velocity at contact point
        linVel, angVel = facet.interpolatePtLinAngVel(contPt)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            contPt,
            linVel,
            angVel,
            capPos,
            capDyn.vel,
            capDyn.angVel,
            normal,
            contPt,
            uN,
            max(facet.halfThick, cap.radius),
            cap.radius,
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


# Register classes for dispatching
# Capsule.registerClass()
# Bo1_Capsule_Aabb.registerClass()
# Cg2_Capsule_Capsule_L6Geom.registerClass()
# Cg2_Sphere_Capsule_L6Geom.registerClass()
# Cg2_Wall_Capsule_L6Geom.registerClass()
# Cg2_Facet_Capsule_L6Geom.registerClass()
# Cg2_InfCylinder_Capsule_L6Geom.registerClass()
