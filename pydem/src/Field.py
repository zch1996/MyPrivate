#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Object import Object
from pydem.src.demmath import AlignedBox3r, Real, INF


class Field(Object):
    """Base class for simulation fields."""

    def __init__(self):
        """Initialize Field with default values."""
        super().__init__()
        self.scene = None
        self.nodes = []  # List of shared_ptr<Node>

    def getScene(self):
        """Get associated scene."""
        return self.scene

    def selfTest(self):
        """Perform self-test. Default implementation does nothing."""
        pass

    def critDt(self):
        """Calculate critical timestep for this field."""
        return INF

    def getRenderingBBox(self):
        """Get bounding box for rendering."""
        bbox = AlignedBox3r()
        for node in self.nodes:
            if node is not None:
                bbox.extend(node.pos)
        return bbox
