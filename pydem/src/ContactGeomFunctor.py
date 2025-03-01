#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from typing import Optional, List, Dict, Tuple
import numpy as np
import math


from .Functor import CGeomFunctor
from .DEMLogging import DEM_LOGGER
from .demmath import Vector3r, Matrix3r, Quaternionr, Real, Vector2r
from .ContactGeom import ContactGeom


@DEM_LOGGER
class Cg2_Any_Any_L6Geom__Base(CGeomFunctor):
    """Base class for contact geometry functors that create L6Geom."""

    # Approximation flags
    class ApproxFlags:
        APPROX_NO_MID_NORMAL = 1  # Use previous normal instead of mid-step normal
        APPROX_NO_RENORM_MID_NORMAL = 2  # Don't renormalize mid-step normal
        APPROX_NO_MID_TRSF = 4  # Use previous rotation instead of mid-step
        APPROX_NO_MID_BRANCH = 8  # Use current branches instead of mid-step

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.approxMask = 0  # Geometric approximation flags
        self.noRatch = False  # FIXME: document what it really does
        self.iniLensTouch = True  # Set L6Geom.lens to touch distance
        self.trsfRenorm = 100  # How often to renormalize trsf

    def go(self, s1, s2, shift2, force, C):
        """
        Process contact geometry between two shapes.

        Args:
            s1: First shape
            s2: Second shape
            shift2: Shift vector for second shape (for periodic boundaries)
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact geometry was created/updated, False otherwise
        """
        raise NotImplementedError(
            "Cg2_Any_Any_L6Geom__Base::go: This is an abstract class "
            "which should not be used directly; use derived classes."
        )

    def goReverse(self, s1, s2, shift2, force, C):
        """
        Process contact geometry with shapes in reverse order.

        Args:
            s1: First shape
            s2: Second shape
            shift2: Shift vector for second shape (for periodic boundaries)
            force: Whether to force contact creation
            C: Contact object

        Returns:
            True if contact geometry was created/updated, False otherwise
        """
        raise NotImplementedError(
            f"ContactLoop should swap interaction arguments, the order is "
            f"{s1.getClassName()}+{s2.getClassName()} "
            f"(goReverse should never be called)."
        )

    def handleSpheresLikeContact(
        self, C, pos1, vel1, angVel1, pos2, vel2, angVel2, normal, contPt, uN, r1, r2
    ):
        """
        Handle sphere-like contacts.

        Args:
            C: Contact object (modified in-place)
            pos1: Position of first particle
            vel1: Velocity of first particle
            angVel1: Angular velocity of first particle
            pos2: Position of second particle
            vel2: Velocity of second particle
            angVel2: Angular velocity of second particle
            normal: Contact normal vector
            contPt: Contact point
            uN: Normal overlap
            r1: Radius of first particle
            r2: Radius of second particle
        """
        # Create new geometry if needed
        if not C.geom:
            from L6Geom import L6Geom

            # Create and configure new geometry
            C.geom = L6Geom()  # Object is modified in-place
            C.geom.contactGeomType = ContactGeom.Type.L6GEOM  # Set the type
            g = C.geom

            g.setInitialLocalCoords(normal)
            g.uN = uN

            # Set lens lengths
            if self.iniLensTouch:
                g.lens = Vector2r(abs(r1), abs(r2))
            else:
                g.lens = Vector2r(abs(r1) + uN / 2, abs(r2) + uN / 2)

            # Handle negative lens values
            if g.lens[0] < 0:
                g.lens[0] = g.lens[1]
            if g.lens[1] < 0:
                g.lens[1] = g.lens[0]

            # Set contact area
            g.contA = (
                math.pi
                * (min(r1, r2) if r1 > 0 and r2 > 0 else (r1 if r1 > 0 else r2)) ** 2
            )

            # Initialize node
            g.getNode().pos = contPt
            g.getNode().ori = Quaternionr(g.trsf)
            return

        # Update existing geometry
        g = C.geom
        currNormal = normal
        prevNormal = g.trsf[:, 0]  # First column of trsf
        prevContPt = C.geom.getNode().pos
        dt = self.scene.dt

        # Handle periodic boundaries
        shiftVel2 = (
            self.scene.cell.intrShiftVel(C.cellDist)
            if self.scene.isPeriodic
            else Vector3r(0, 0, 0)
        )

        # Calculate rotation vectors
        normRotVec = np.cross(prevNormal, currNormal)

        if self.approxMask & self.ApproxFlags.APPROX_NO_MID_NORMAL:
            midNormal = prevNormal
        else:
            midNormal = 0.5 * (prevNormal + currNormal)

        # Handle special case of inverted normal
        if np.linalg.norm(midNormal) == 0:
            midNormal = currNormal
        elif not (
            self.approxMask & self.ApproxFlags.APPROX_NO_RENORM_MID_NORMAL
        ) and not (self.approxMask & self.ApproxFlags.APPROX_NO_MID_NORMAL):
            midNormal = midNormal / np.linalg.norm(midNormal)

        normTwistVec = midNormal * dt * 0.5 * np.dot(midNormal, angVel1 + angVel2)

        # Compute current transformation
        prevTrsf = g.trsf

        if self.approxMask & self.ApproxFlags.APPROX_NO_MID_TRSF:
            midTrsf = np.zeros((3, 3))
            midTrsf[:, 0] = prevNormal
            midTrsf[:, 1] = prevTrsf[:, 1]
        else:
            midTrsf = np.zeros((3, 3))
            midTrsf[:, 0] = midNormal
            midTrsf[:, 1] = (
                prevTrsf[:, 1] - np.cross(prevTrsf[:, 1], normRotVec + normTwistVec) / 2
            )

        midTrsf[:, 2] = np.cross(midTrsf[:, 0], midTrsf[:, 1])

        currTrsf = np.zeros((3, 3))
        currTrsf[:, 0] = currNormal
        currTrsf[:, 1] = prevTrsf[:, 1] - np.cross(
            midTrsf[:, 1], normRotVec + normTwistVec
        )
        currTrsf[:, 2] = np.cross(currTrsf[:, 0], currTrsf[:, 1])

        # Orthonormalize if needed
        if self.trsfRenorm > 0 and (self.scene.step % self.trsfRenorm) == 0:
            currTrsf[:, 0] = currTrsf[:, 0] / np.linalg.norm(currTrsf[:, 0])
            currTrsf[:, 1] -= currTrsf[:, 0] * np.dot(currTrsf[:, 1], currTrsf[:, 0])
            currTrsf[:, 1] = currTrsf[:, 1] / np.linalg.norm(currTrsf[:, 1])
            currTrsf[:, 2] = np.cross(currTrsf[:, 0], currTrsf[:, 1])

            # Check if matrix is far from orthonormal
            if abs(np.linalg.det(currTrsf) - 1) > 0.05:
                self.error(
                    f"##{{C.leakPA().id}}+{{C.leakPB().id}}, |trsf|={{np.linalg.det(currTrsf)}}"
                )
                g.trsf = currTrsf
                raise RuntimeError("Transformation matrix far from orthonormal.")

        # Compute relative velocity
        if self.approxMask & self.ApproxFlags.APPROX_NO_MID_BRANCH:
            midContPt = contPt
            midPos1 = pos1
            midPos2 = pos2
        else:
            midContPt = 0.5 * (prevContPt + contPt)
            midPos1 = pos1 - (dt / 2) * vel1
            midPos2 = pos2 - (dt / 2) * (vel2 + shiftVel2)

        c1x = (r1 * midNormal) if (self.noRatch and r1 > 0) else (midContPt - midPos1)
        c2x = (-r2 * midNormal) if (self.noRatch and r2 > 0) else (midContPt - midPos2)

        relVel = (vel2 + shiftVel2 + np.cross(angVel2, c2x)) - (
            vel1 + np.cross(angVel1, c1x)
        )

        # Update geometry
        g.trsf = currTrsf
        g.vel = midTrsf.T @ relVel
        g.angVel = midTrsf.T @ (angVel2 - angVel1)
        g.uN = uN

        # Update node
        g.getNode().pos = contPt
        g.getNode().ori = Quaternionr(g.trsf)

    def getApproxMask(self):
        """Get approximation mask."""
        return self.approxMask

    def setApproxMask(self, am):
        """
        Set approximation mask.

        Args:
            am: Approximation mask (0-15)
        """
        if am < 0 or am > 15:
            raise ValueError("approxMask must be between 0 and 15")
        self.approxMask = am
