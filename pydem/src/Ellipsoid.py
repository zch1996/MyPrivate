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
    Matrix3r,
    Quaternionr,
    AngleAxisr,
    AlignedBox3r,
    Real,
    INF,
    NAN,
)
from pydem.src.DEMData import DEMData
from pydem.src.Sphere import Sphere, Bo1_Sphere_Aabb
from pydem.src.Wall import Wall
from pydem.src.Facet import Facet
from pydem.src.Node import Node


class Ellipsoid(Shape):
    """
    Ellipsoidal particle.

    This shape represents an ellipsoid with three semi-principal axes.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.semiAxes = Vector3r(NAN, NAN, NAN)  # Semi-principal axes

    def numNodes(self):
        """Return number of nodes needed by this shape."""
        return 1

    def selfTest(self, p):
        """
        Perform self-test to verify shape validity.

        Args:
            p: Particle using this shape
        """
        if not (self.semiAxes.min() > 0):
            raise RuntimeError(
                f"Ellipsoid #{p.getId()}: all semi-principal semiAxes must be positive "
                f"(current minimum is {self.semiAxes.min()})"
            )

        if not self.numNodesOk():
            raise RuntimeError(
                f"Ellipsoid #{p.getId()}: numNodesOk() failed: must be 1, not {len(self.nodes)}."
            )

        super().selfTest(p)

    def equivRadius(self):
        """Calculate volume-based equivalent radius."""
        return (self.semiAxes.prod()) ** (1 / 3)

    def volume(self):
        """Calculate volume of the ellipsoid."""
        return (4 / 3) * math.pi * self.semiAxes.prod()

    def applyScale(self, scale):
        """
        Apply scaling to the ellipsoid.

        Args:
            scale: Scale factor
        """
        self.semiAxes *= scale

    def isInside(self, pt):
        """
        Check if a point is inside the ellipsoid.

        Args:
            pt: Point to check

        Returns:
            True if point is inside, False otherwise
        """
        l = self.nodes[0].glob2loc(pt)
        return (
            (l[0] / self.semiAxes[0]) ** 2
            + (l[1] / self.semiAxes[1]) ** 2
            + (l[2] / self.semiAxes[2]) ** 2
        ) <= 1

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
        rotateOk = False
        self.checkNodesHaveDemData()

        m = (4 / 3) * math.pi * self.semiAxes.prod() * density
        mass += m

        # Diagonal inertia tensor for ellipsoid
        I[0, 0] += (1 / 5) * m * (self.semiAxes[1] ** 2 + self.semiAxes[2] ** 2)
        I[1, 1] += (1 / 5) * m * (self.semiAxes[2] ** 2 + self.semiAxes[0] ** 2)
        I[2, 2] += (1 / 5) * m * (self.semiAxes[0] ** 2 + self.semiAxes[1] ** 2)

        return mass, I, rotateOk

    def trsfFromUnitSphere(self):
        """
        Return matrix transforming unit sphere to this ellipsoid.

        Returns:
            Transformation matrix
        """
        M = Matrix3r.zeros()
        for i in range(3):
            M[:, i] = self.nodes[0].ori.rotate(self.semiAxes[i] * Vector3r.unit(i))
        return M

    def trsfFromUnitSphere(self, ori):
        """
        Return matrix transforming unit sphere to this ellipsoid with additional rotation.

        Args:
            ori: Additional orientation

        Returns:
            Transformation matrix
        """
        M = Matrix3r.zeros()
        for i in range(3):
            M[:, i] = ori.rotate(
                self.nodes[0].ori.rotate(self.semiAxes[i] * Vector3r.unit(i))
            )
        return M

    def axisExtent(self, axis):
        """
        Return extent along one global axis.

        Args:
            axis: Axis index (0,1,2)

        Returns:
            Extent along the axis
        """
        M = self.trsfFromUnitSphere()
        return np.linalg.norm(M[axis, :])

    def rotatedExtent(self, axis, ori):
        """
        Return extent along one global axis with additional rotation.

        Args:
            axis: Axis index (0,1,2)
            ori: Additional orientation

        Returns:
            Extent along the axis
        """
        M = self.trsfFromUnitSphere(ori)
        return np.linalg.norm(M[axis, :])

    def alignedBox(self):
        """
        Compute axis-aligned bounding box.

        Returns:
            Axis-aligned bounding box
        """
        M = self.trsfFromUnitSphere()
        pos = self.nodes[0].pos
        delta = Vector3r(
            np.linalg.norm(M[0, :]), np.linalg.norm(M[1, :]), np.linalg.norm(M[2, :])
        )
        return AlignedBox3r(pos - delta, pos + delta)

    def asRaw(self, center, radius, nn, raw):
        """
        Convert to raw data.

        Args:
            center: Center position (output)
            radius: Radius (output)
            nn: Nodes (output)
            raw: Raw data (output)
        """
        center = self.nodes[0].pos
        radius = self.semiAxes.max()

        aa = AngleAxisr(self.nodes[0].ori)
        raw.resize(6)

        # Store orientation as axis*angle
        rawOri = raw[0:3]
        rawOri = aa.axis() * aa.angle()

        # Store semi-axes
        rawSemiAxes = raw[3:6]
        rawSemiAxes = self.semiAxes

    def setFromRaw(self, center, radius, nn, raw):
        """
        Set from raw data.

        Args:
            center: Center position
            radius: Radius
            nn: Nodes
            raw: Raw data
        """
        if len(raw) != 6:
            raise RuntimeError(
                f"Ellipsoid::setFromRaw: expected 6 raw values, got {len(raw)}"
            )

        # Create node if needed
        if len(self.nodes) == 0:
            self.nodes.append(Node())

        # Set position and orientation
        self.nodes[0].pos = center

        # Set orientation from axis*angle
        rawOri = raw[0:3]
        n = np.linalg.norm(rawOri)
        if n == 0:
            self.nodes[0].ori = Quaternionr.identity()
        else:
            self.nodes[0].ori = Quaternionr(AngleAxisr(n, rawOri / n))

        nn.append(self.nodes[0])

        # Set semi-axes
        self.semiAxes = Vector3r(raw[3], raw[4], raw[5])


@DEM_LOGGER
class Bo1_Ellipsoid_Aabb(BoundFunctor):
    """
    Functor creating Aabb from Ellipsoid.

    Note: Does not handle rotation detected by verlet distance.
    Warning: DemField.distFactor is ignored.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh):
        """
        Compute Aabb for an Ellipsoid.

        Args:
            sh: Ellipsoid shape
        """
        if not sh.bound:
            sh.bound = Aabb()
            sh.bound.maxRot = 0.0  # Consider rotation

        aabb = sh.bound
        pos = sh.nodes[0].pos

        if not self.scene.isPeriodic or not self.scene.cell.hasShear():
            # Simple case: no periodic boundaries or no shear
            M = sh.trsfFromUnitSphere()
            delta = Vector3r(
                np.linalg.norm(M[0, :]),
                np.linalg.norm(M[1, :]),
                np.linalg.norm(M[2, :]),
            )
            aabb.min = pos - delta
            aabb.max = pos + delta
        else:
            # Complex case: periodic boundaries with shear
            extents = Vector3r(0, 0, 0)
            sT = self.scene.cell.getShearTrsf()

            for i in range(3):
                q = Quaternionr()
                q.setFromTwoVectors(
                    Vector3r.unit(i), np.cross(sT[:, (i + 1) % 3], sT[:, (i + 2) % 3])
                )
                M = sh.trsfFromUnitSphere(q.conjugate())
                extents[i] = np.linalg.norm(M[i, :])

            extents = self.scene.cell.shearAlignedExtents(extents)
            aabb.min = self.scene.cell.unshearPt(pos) - extents
            aabb.max = self.scene.cell.unshearPt(pos) + extents


@DEM_LOGGER
class Cg2_Ellipsoid_Ellipsoid_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Incrementally compute L6Geom for contact of 2 ellipsoids.

    Uses the Perram-Wertheim potential function.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.brent = True  # Use Brent iteration for finding maximum of the Perram-Wertheim potential
        self.brentBits = 4 * 8  # Precision for the Brent method, as number of bits

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between two Ellipsoids.

        Args:
            sh1: First Ellipsoid shape
            sh2: Second Ellipsoid shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        return self.go_Ellipsoid_or_Sphere(
            sh1, sh1.semiAxes, sh2, sh2.semiAxes, shift2, force, C
        )

    def go_Ellipsoid_or_Sphere(self, s1, semiAxesA, s2, semiAxesB, shift2, force, C):
        """
        Compute contact geometry between Ellipsoid/Sphere shapes.

        Args:
            s1: First shape
            semiAxesA: Semi-axes of first shape
            s2: Second shape
            semiAxesB: Semi-axes of second shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # Notation from Perram, Rasmussen, Præstgaard, Lebowtz: Ellipsoid contact potential
        ra = s1.nodes[0].pos
        rb = s2.nodes[0].pos + shift2
        a = semiAxesA
        b = semiAxesB
        oa = s1.nodes[0].ori
        ob = s2.nodes[0].ori

        u = [
            oa.rotate(Vector3r.unit(0)),
            oa.rotate(Vector3r.unit(1)),
            oa.rotate(Vector3r.unit(2)),
        ]

        v = [
            ob.rotate(Vector3r.unit(0)),
            ob.rotate(Vector3r.unit(1)),
            ob.rotate(Vector3r.unit(2)),
        ]

        dyn1 = s1.nodes[0].getDataTyped(DEMData)
        dyn2 = s2.nodes[0].getDataTyped(DEMData)

        R = rb - ra  # (2.1)

        # Construct matrices A and B
        A = Matrix3r.zeros()
        B = Matrix3r.zeros()

        for k in range(3):
            A += np.outer(u[k], u[k]) / a[k] ** 2  # (2.2a)
            B += np.outer(v[k], v[k]) / b[k] ** 2  # (2.2b)

        Ainv = np.linalg.inv(A)  # (2.3a)
        Binv = np.linalg.inv(B)  # (2.3b)

        # (2.4), for Brent's maximization: return only the function value
        # Negated for use with minimization algorithm
        def neg_S_lambda_0(l):
            return -l * (1 - l) * R.dot(np.linalg.inv((1 - l) * Ainv + l * Binv).dot(R))

        # Find maximum using Brent's method
        if self.brent:
            from scipy import optimize

            result = optimize.minimize_scalar(
                neg_S_lambda_0, bounds=(0, 1), method="bounded"
            )
            L = result.x
            Fab = -result.fun  # Invert the sign
        else:
            raise RuntimeError(
                "Cg2_Ellipsoid_Ellipsoid_L6Geom::go: Newton-Raphson iteration is not "
                "yet implemented; use Brent's method by saying brent=True."
            )

        # Check if ellipsoids are separated
        if Fab > 1 and not C.isReal() and not force:
            return False

        # Calculate contact geometry
        G = (1 - L) * Ainv + L * Binv  # (2.6)
        nUnnorm = np.linalg.inv(G).dot(R)  # Donev, (19)
        contPt = ra + (1 - L) * Ainv.dot(nUnnorm)  # Donev, (19)

        # Compute penetration depth
        mu = math.sqrt(Fab)
        Rnorm = np.linalg.norm(R)
        nUnit = nUnnorm / np.linalg.norm(nUnnorm)
        rUnit = R / Rnorm

        uN = Rnorm * (1 - 1 / mu) * rUnit.dot(nUnit)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            ra,
            dyn1.vel,
            dyn1.angVel,
            rb,
            dyn2.vel,
            dyn2.angVel,
            nUnit,
            contPt,
            uN,
            np.linalg.norm(contPt - ra),
            np.linalg.norm(contPt - rb),
        )

        return True

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = (s1.semiAxes.max() + s2.semiAxes.max()) ** 2


@DEM_LOGGER
class Cg2_Sphere_Ellipsoid_L6Geom(Cg2_Ellipsoid_Ellipsoid_L6Geom):
    """
    Compute the geometry of Ellipsoid + Sphere collision.

    Uses the code from Cg2_Ellipsoid_Ellipsoid_L6Geom, representing the sphere
    as an ellipsoid with all semiaxes equal to the radius.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Sphere and Ellipsoid.

        Args:
            sh1: Sphere shape
            sh2: Ellipsoid shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        return self.go_Ellipsoid_or_Sphere(
            sh1, sh1.radius * Vector3r(1, 1, 1), sh2, sh2.semiAxes, shift2, force, C
        )

    def setMinDist00Sq(self, s1, s2, C):
        """Set minimum distance between nodes."""
        C.minDist00Sq = (s1.radius + s2.semiAxes.max()) ** 2


@DEM_LOGGER
class Cg2_Wall_Ellipsoid_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of ellipsoid and wall (axis-aligned plane).
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Wall and Ellipsoid.

        Args:
            sh1: Wall shape
            sh2: Ellipsoid shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        wall = sh1
        ell = sh2

        ax = wall.axis
        sense = wall.sense
        wallPos = wall.nodes[0].pos
        ellPos = ell.nodes[0].pos + shift2

        # Check if ellipsoid is too far from wall
        extent = ell.axisExtent(ax)
        if (
            (
                (wallPos[ax] < (ellPos[ax] - extent))
                or (wallPos[ax] > (ellPos[ax] + extent))
            )
            and not C.isReal()
            and not force
        ):
            return False

        # Calculate penetration distance and normal
        dist = (
            ellPos[ax] - wallPos[ax]
        )  # Signed distance: positive for ellipsoid above the wall

        # Determine normal direction
        if wall.sense == 0:
            normAxSgn = 1 if dist > 0 else -1  # Both-side wall
        else:
            normAxSgn = 1 if sense == 1 else -1

        normal = normAxSgn * Vector3r.unit(ax)
        uN = normAxSgn * dist - extent

        # Calculate contact point
        M = ell.trsfFromUnitSphere()
        Mprime = M.transpose()

        for i in range(3):
            Mprime[:, i] = Mprime[:, i] / np.linalg.norm(Mprime[:, i])

        # Contact point is the extremal point of the ellipsoid in the direction of the wall
        contPt = ellPos + (-normAxSgn) * (M.dot(Mprime))[:, ax]
        contPt[ax] = wallPos[ax]

        contR = np.linalg.norm(contPt - ellPos)

        # Get dynamics data
        ellDyn = ell.nodes[0].getDataTyped(DEMData)
        wallDyn = wall.nodes[0].getDataTyped(DEMData)

        # Handle contact
        self.handleSpheresLikeContact(
            C,
            wallPos,
            wallDyn.vel,
            wallDyn.angVel,
            ellPos,
            ellDyn.vel,
            ellDyn.angVel,
            normal,
            contPt,
            uN,
            -contR,  # Negative for wall
            contR,
        )

        return True


@DEM_LOGGER
class Cg2_Facet_Ellipsoid_L6Geom(Cg2_Any_Any_L6Geom__Base):
    """
    Compute L6Geom for contact of ellipsoid and facet.

    Warning: This class does not work correctly (the result is correct only for
    face contact, otherwise bogus).
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def go(self, sh1, sh2, shift2, force, C):
        """
        Compute contact geometry between Facet and Ellipsoid.

        Args:
            sh1: Facet shape
            sh2: Ellipsoid shape
            shift2: Shift vector for periodic boundaries
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact exists, False otherwise
        """
        # This is a placeholder - facet-ellipsoid collision is complex
        # and would need a more detailed implementation
        self.warning(
            "Facet-Ellipsoid collision not fully implemented - only face contact works correctly"
        )
        return False
