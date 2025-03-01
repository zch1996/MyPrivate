#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .demmath import Vector3r, AlignedBox3r
from .Object import Object


class Bound(Object):
    """Class representing a bounding box for shapes."""

    def __init__(self):
        """Initialize an empty bounding box."""
        super().__init__()
        self.box = AlignedBox3r()
        # Use properties to map min/max directly to box's min/max

    @property
    def min(self):
        """Get minimum corner of bounding box."""
        return self.box.min

    @min.setter
    def min(self, value):
        """Set minimum corner of bounding box."""
        self.box.min = value

    @property
    def max(self):
        """Get maximum corner of bounding box."""
        return self.box.max

    @max.setter
    def max(self, value):
        """Set maximum corner of bounding box."""
        self.box.max = value
