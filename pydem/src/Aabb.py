#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .Bound import Bound


from typing import List
from .demmath import Vector3r, Quaternionr, Real, NAN


class Aabb(Bound):
    """Axis-aligned bounding box."""

    def __init__(self):
        """Initialize Aabb with default values."""
        super().__init__()
        self.nodeLastPos = []  # List of Vector3r
        self.nodeLastOri = []  # List of Quaternionr
        self.maxD2 = 0.0  # Maximum squared distance
        self.maxRot = NAN  # Maximum rotation
