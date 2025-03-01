#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import math
from enum import Enum

from pydem.src.demmath import NAN
from pydem.src.Object import Object


class MatState(Object):
    """Base class for material state information."""

    def __init__(self):
        """Initialize MatState with default values."""
        super().__init__()
        self.mutex = threading.RLock()  # Thread-safe lock

    def clone(self):
        """Create a clone of this material state."""
        return MatState()

    def getNumScalars(self):
        """Get number of scalar values in this state."""
        return 0

    def getScalar(self, index, time=0, step=-1, smoothing=0):
        """Get scalar value by index."""
        return NAN

    def getScalarName(self, index):
        """Get name of scalar value by index."""
        return ""


class MaterialType(Enum):
    """Enumeration of material types."""

    ElastMat = 0
    FrictMat = 1
    Unknown = 2


class Material(Object):
    """Base class for all material types."""

    def __init__(self, density=1000.0, id=-1):
        """Initialize material with density and ID."""
        super().__init__()
        self.density = density
        self.id = id
        self.state = None
        self.materialType = MaterialType.Unknown

    def getDensity(self):
        """Get material density."""
        return self.density

    def setDensity(self, newDensity):
        """Set material density."""
        if newDensity <= 0:
            raise ValueError("Density must be positive")
        self.density = newDensity

    def getId(self):
        """Get material ID."""
        return self.id

    def setId(self, newId):
        """Set material ID."""
        self.id = newId

    def getState(self):
        """Get material state."""
        return self.state

    def setState(self, newState):
        """Set material state."""
        self.state = newState

    def getMaterialType(self):
        """Get material type."""
        return self.materialType


class ElastMat(Material):
    """Elastic material with Young's modulus."""

    def __init__(self, density=1000.0, young=1e9):
        """Initialize elastic material."""
        super().__init__(density)
        if young <= 0:
            raise ValueError("Young's modulus must be positive")
        self.young = young
        self.materialType = MaterialType.ElastMat

    def getYoung(self):
        """Get Young's modulus."""
        return self.young

    def setYoung(self, y):
        """Set Young's modulus."""
        if y <= 0:
            raise ValueError("Young's modulus must be positive")
        self.young = y


class FrictMat(ElastMat):
    """Frictional material with friction angle and stiffness ratio."""

    def __init__(self, density=1000.0, young=1e9, tanPhi=0.5, ktDivKn=0.2):
        """Initialize frictional material."""
        super().__init__(density, young)
        if tanPhi < 0:
            raise ValueError("Tangent of friction angle must be non-negative")
        if ktDivKn <= 0:
            raise ValueError("ktDivKn must be positive")
        self.tanPhi = tanPhi
        self.ktDivKn = ktDivKn
        self.materialType = MaterialType.FrictMat

    def getTanPhi(self):
        """Get tangent of friction angle."""
        return self.tanPhi

    def setTanPhi(self, tp):
        """Set tangent of friction angle."""
        if tp < 0:
            raise ValueError("Tangent of friction angle must be non-negative")
        self.tanPhi = tp

    def getKtDivKn(self):
        """Get ratio of tangential to normal stiffness."""
        return self.ktDivKn

    def setKtDivKn(self, k):
        """Set ratio of tangential to normal stiffness."""
        if k <= 0:
            raise ValueError("ktDivKn must be positive")
        self.ktDivKn = k
