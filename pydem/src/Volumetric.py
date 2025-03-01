#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List

from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr, AlignedBox3r, Real, NAN


class Volumetric:
    """Utility functions for volumetric calculations and transformations."""

    @staticmethod
    def tetraVolume(A: Vector3r, B: Vector3r, C: Vector3r, D: Vector3r) -> Real:
        """
        Compute tetrahedron volume; the volume is signed, positive for points in
        canonical ordering.

        Args:
            A: First vertex
            B: Second vertex
            C: Third vertex
            D: Fourth vertex

        Returns:
            Real: Tetrahedron volume
        """
        # Volume = 1/6 * |det(B-A C-A D-A)|
        M = np.zeros((3, 3))
        M[:, 0] = B - A
        M[:, 1] = C - A
        M[:, 2] = D - A

        return np.abs(np.linalg.det(M)) / 6.0

    @staticmethod
    def tetraInertia(A: Vector3r, B: Vector3r, C: Vector3r, D: Vector3r) -> Matrix3r:
        """
        Compute tetrahedron inertia.

        Args:
            A: First vertex
            B: Second vertex
            C: Third vertex
            D: Fourth vertex

        Returns:
            Matrix3r: Inertia tensor
        """
        # Compute tetrahedron inertia tensor using standard formula
        I = np.zeros((3, 3))
        vol = Volumetric.tetraVolume(A, B, C, D)

        # Vertices relative to centroid
        centroid = (A + B + C + D) / 4.0
        a = A - centroid
        b = B - centroid
        c = C - centroid
        d = D - centroid

        # Compute covariance matrix
        for v in [a, b, c, d]:
            I[0, 0] += v[1] ** 2 + v[2] ** 2
            I[1, 1] += v[0] ** 2 + v[2] ** 2
            I[2, 2] += v[0] ** 2 + v[1] ** 2
            I[0, 1] -= v[0] * v[1]
            I[0, 2] -= v[0] * v[2]
            I[1, 2] -= v[1] * v[2]

        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]

        return (vol / 10.0) * I

    @staticmethod
    def tetraInertia_cov(
        v: List[Vector3r], fixSign: bool = True
    ) -> Tuple[Matrix3r, Real]:
        """
        Compute tetrahedron inertia using covariance method.

        Args:
            v: Array of 4 vertices
            fixSign: Whether to fix sign of volume if negative

        Returns:
            Tuple containing:
                - Matrix3r: Inertia tensor
                - Real: Volume
        """
        C0 = np.zeros((3, 3))  # Separate parts of covariance
        C1 = np.zeros(3)

        for i in range(4):
            C0 += np.outer(v[i], v[i])
            C1 += v[i]

        vol = Volumetric.tetraVolume(v[0], v[1], v[2], v[3])
        if vol < 0 and fixSign:
            vol *= -1

        C = (vol / 20.0) * (C0 + np.outer(C1, C1))
        I = np.trace(C) * np.eye(3) - C

        assert not fixSign or I[0, 0] > 0
        return I, vol

    @staticmethod
    def tetraInertia_grid(v: List[Vector3r], div: int = 100) -> Matrix3r:
        """
        Compute tetrahedron inertia using grid sampling.

        Args:
            v: Array of 4 vertices
            div: Grid division factor

        Returns:
            Matrix3r: Inertia tensor
        """
        b = AlignedBox3r()
        for i in range(4):
            b.extend(v[i])

        dd = min(b.sizes()) / div

        # Point inside test
        M0 = np.zeros((4, 4))
        for i in range(4):
            M0[i, :3] = v[i]
            M0[i, 3] = 1

        D0 = np.linalg.det(M0)
        C = np.zeros((3, 3))
        dV = dd**3

        for x in np.arange(b.min()[0] + dd / 2, b.max()[0], dd):
            for y in np.arange(b.min()[1] + dd / 2, b.max()[1], dd):
                for z in np.arange(b.min()[2] + dd / 2, b.max()[2], dd):
                    xyz = np.array([x, y, z])
                    inside = True

                    for i in range(4):
                        D = M0.copy()
                        D[i, :3] = xyz
                        if np.signbit(np.linalg.det(D)) != np.signbit(D0):
                            inside = False
                            break

                    if inside:
                        C += dV * np.outer(xyz, xyz)

        return np.trace(C) * np.eye(3) - C

    @staticmethod
    def triangleInertia(v0: Vector3r, v1: Vector3r, v2: Vector3r) -> Matrix3r:
        """
        Compute triangle inertia; zero thickness is assumed for inertia as such;
        the density to multiply with at the end is per unit area;
        volumetric density should therefore be multiplied by thickness.

        Args:
            v0: First vertex
            v1: Second vertex
            v2: Third vertex

        Returns:
            Matrix3r: Inertia tensor
        """
        V = np.vstack((v0, v1, v2))  # Rows
        a = np.linalg.norm(np.cross(v1 - v0, v2 - v0))  # Twice the triangle area

        S = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) * (1.0 / 24.0)
        C = a * V.T @ S @ V

        return np.trace(C) * np.eye(3) - C

    @staticmethod
    def triangleArea(v0: Vector3r, v1: Vector3r, v2: Vector3r) -> Real:
        """
        Unsigned (always positive) triangle area given its vertices.

        Args:
            v0: First vertex
            v1: Second vertex
            v2: Third vertex

        Returns:
            Real: Triangle area
        """
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    @staticmethod
    def inertiaTensorTranslate(I: Matrix3r, m: Real, off: Vector3r) -> Matrix3r:
        """
        Recalculates inertia tensor of a body after translation away from
        (default) or towards its centroid.

        Args:
            I: Inertia tensor in the original coordinates
            m: Mass of the body; if positive, translation is away from the
               centroid; if negative, towards centroid
            off: Offset of the new origin from the original origin

        Returns:
            Matrix3r: Inertia tensor in the new coordinate system
        """
        return I + m * (np.dot(off, off) * np.eye(3) - np.outer(off, off))

    @staticmethod
    def inertiaTensorRotate(I, rot):
        """
        Rotate inertia tensor by rotation matrix or quaternion.

        Args:
            I: 3x3 inertia tensor
            rot: Rotation as either a Quaternionr or a 3x3 rotation matrix

        Returns:
            Rotated inertia tensor
        """
        # Check if rot is already a rotation matrix
        if isinstance(rot, np.ndarray) and rot.shape == (3, 3):
            T = rot
        else:
            # Convert quaternion to rotation matrix
            T = rot.toRotationMatrix()

        # Apply rotation: T * I * T^T
        return T @ I @ T.transpose()

    @staticmethod
    def computePrincipalAxes(
        m: Real, Sg: Vector3r, Ig: Matrix3r
    ) -> Tuple[Vector3r, Quaternionr, Vector3r]:
        """
        Compute position, orientation, and inertia given mass, first and second-order momentum.

        Args:
            m: Mass
            Sg: Static moment (first moment of mass)
            Ig: Inertia tensor in global coordinates

        Returns:
            Tuple containing:
                - pos: Position vector (center of mass)
                - ori: Orientation quaternion aligning with principal axes
                - inertia: Principal moments of inertia
        """
        assert m > 0

        # Clump's centroid
        pos = Sg / m

        # Inertia at clump's centroid but with world orientation
        Ic_orientG = Volumetric.inertiaTensorTranslate(Ig, -m, pos)

        # Symmetrize
        Ic_orientG[1, 0] = Ic_orientG[0, 1]
        Ic_orientG[2, 0] = Ic_orientG[0, 2]
        Ic_orientG[2, 1] = Ic_orientG[1, 2]

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(Ic_orientG)

        # Create quaternion from rotation matrix
        ori = Quaternionr.fromRotationMatrix(eigenvectors)
        ori.normalize()

        # Principal moments of inertia
        inertia = Vector3r(eigenvalues[0], eigenvalues[1], eigenvalues[2])

        return pos, ori, inertia
