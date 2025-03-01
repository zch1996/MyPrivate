#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Functor import CPhysFunctor
from pydem.src.DEMLogging import DEM_LOGGER
from typing import Optional
import math
from pydem.src.ContactPhys import FrictPhys


@DEM_LOGGER
class Cp2_FrictMat_FrictPhys(CPhysFunctor):
    """Contact physics functor for frictional materials."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.tanPhi = None  # MatchMaker for friction angle tangent

    def go(self, m1, m2, C):
        """
        Compute contact physics between two materials.

        Args:
            m1: First material
            m2: Second material
            C: Contact object
        """
        if not C.phys:

            C.phys = FrictPhys()

        self.updateFrictPhys(m1, m2, C.phys, C)

    def updateFrictPhys(self, mat1, mat2, ph, C):
        """
        Update frictional physics properties.

        Args:
            mat1: First material
            mat2: Second material
            ph: Physics object (modified in-place)
            C: Contact object
        """
        # Get geometry parameters
        l6g = C.geom
        l0 = l6g.lens[0]
        l1 = l6g.lens[1]
        A = l6g.contA

        # Calculate normal stiffness (direct modification)
        ph.kn = 1 / (1 / (mat1.young * A / l0) + 1 / (mat2.young * A / l1))

        # Calculate tangential stiffness (direct modification)
        ph.kt = 0.5 * (mat1.ktDivKn + mat2.ktDivKn) * ph.kn

        # Calculate friction angle tangent (direct modification)
        if not self.tanPhi:
            ph.tanPhi = min(mat1.tanPhi, mat2.tanPhi)
        else:
            ph.tanPhi = self.tanPhi(mat1.id, mat2.id, mat1.tanPhi, mat2.tanPhi)
