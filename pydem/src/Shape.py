#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from enum import Enum
import math
import numpy as np
from typing import List, Optional, Tuple, Any
from pydem.src.demmath import Vector3r, Matrix3r, AlignedBox3r, NAN
from pydem.src.Object import Object
from pydem.src.Bound import Bound
from pydem.src.Node import Node


# ShapeType enum
class ShapeType(Enum):
    Sphere = 0
    Wall = 1
    Capsule = 2
    Cylinder = 3
    Ellipsoid = 4
    Facet = 5
    InfCylinder = 6
    Tetra = 7
    Tet4 = 8
    Polyhedron = 9
    Membrane = 10
    LevelSet = 11
    Unknown = 12


class Shape(Object):
    """Base class for all shape types."""

    def __init__(self):
        """Initialize shape with default attributes."""
        super().__init__()
        self.nodes = []  # List of Node objects
        self.bound = Bound()
        self.color = 0.5  # Default color with encoded visualization properties
        self.shapeType = ShapeType.Unknown

    def getShapeType(self):
        """Get shape type enumeration."""
        return self.shapeType

    def getNumNodes(self):
        """Get number of nodes required for this shape. Must be implemented by subclasses."""
        raise NotImplementedError(f"{self.toString()} does not implement getNumNodes")

    def checkNumNodes(self):
        """Check if shape has correct number of nodes."""
        return self.getNumNodes() == len(self.nodes)

    def checkNodesHaveDemData(self):
        """Check if all nodes have DEM data attached."""
        if not self.checkNumNodes():
            raise RuntimeError(
                f"Shape has incorrect number of nodes: expected "
                f"{self.getNumNodes()} but got {len(self.nodes)}"
            )

        for node in self.nodes:
            if not node.hasDataTyped(DEMData):
                raise RuntimeError("Node is missing DEMData")

    def getNodes(self):
        """Get list of shape nodes."""
        return self.nodes

    def getAverageNodePosition(self):
        """Calculate average position of all nodes."""
        if not self.nodes:
            raise RuntimeError("Shape has no nodes")

        if len(self.nodes) == 1:
            return self.nodes[0].pos.copy()

        avg = Vector3r(0.0, 0.0, 0.0)
        for node in self.nodes:
            avg += node.pos

        return avg / len(self.nodes)

    def isInside(self, point):
        """Check if point is inside shape. Default implementation raises error."""
        raise NotImplementedError(
            f"{self.toString()} does not implement Shape.isInside"
        )

    def getAlignedBox(self):
        """Get aligned bounding box. Default implementation raises error."""
        raise NotImplementedError(
            f"{self.toString()} does not implement Shape.getAlignedBox"
        )

    def applyScale(self, scale):
        """Apply scale factor to shape. Default implementation raises error."""
        raise NotImplementedError(
            f"{self.toString()} does not implement Shape.applyScale"
        )

    def updateMassInertia(self, density):
        """Update mass and inertia of shape based on density."""
        if len(self.nodes) != 1:
            raise RuntimeError(
                "Mass/inertia update only supported for single-node shapes"
            )

        demData = self.nodes[0].getDataTyped(DEMData)
        if len(demData.getParticleRefs()) > 1:
            raise RuntimeError("Mass/inertia update not supported for shared nodes")

        mass = 0.0
        inertia = Matrix3r()
        canRotate = False

        # Capture the return values from lumpMassInertia
        mass, inertia, canRotate = self.lumpMassInertia(
            self.nodes[0], density, mass, inertia, canRotate
        )

        # Check if inertia tensor is diagonal
        if not np.allclose(inertia - np.diag(np.diagonal(inertia)), 0):
            raise RuntimeError("Non-diagonal inertia tensor not supported")

        demData.inertia = Vector3r(inertia[0, 0], inertia[1, 1], inertia[2, 2])
        demData.mass = mass

    def lumpMassInertia(self, node, density, mass, inertia, canRotate):
        """Calculate mass and inertia contributions. Default implementation raises error."""
        raise NotImplementedError(
            f"{self.toString()}: lumpMassInertia not implemented."
        )

    def getEquivalentRadius(self):
        """Get equivalent radius of shape. Default returns NaN."""
        return NAN

    def getVolume(self):
        """Get volume of shape. Default returns NaN."""
        return NAN

    # Color and visualization properties
    def getBaseColor(self):
        """Get base color value (0-1 range)."""
        return abs(self.color) - math.trunc(abs(self.color))

    def setBaseColor(self, c):
        """Set base color value (0-1 range)."""
        if math.isnan(c):
            return

        self.color = math.trunc(self.color) + (1 if self.color >= 0 else -1) * max(
            0.0, min(c, 1.0)
        )

    def isWireframe(self):
        """Check if shape should be rendered as wireframe."""
        return self.color < 0

    def setWireframe(self, wire):
        """Set wireframe rendering mode."""
        self.color = -abs(self.color) if wire else abs(self.color)

    def isHighlighted(self):
        """Check if shape is highlighted."""
        return abs(self.color) >= 1 and abs(self.color) < 2

    def setHighlighted(self, highlight):
        """Set highlighting state."""
        if self.isHighlighted() == highlight:
            return

        self.color = (1 if self.color >= 0 else -1) + self.getBaseColor()

    def isVisible(self):
        """Check if shape is visible."""
        return abs(self.color) <= 2

    def setVisible(self, visible):
        """Set visibility state."""
        if self.isVisible() == visible:
            return

        highlighted = abs(self.color) > 1
        sign = -1 if self.color < 0 else 1
        self.color = sign * (
            (0 if visible else 2) + self.getBaseColor() + (1 if highlighted else 0)
        )

    # Raw data helpers
    def setFromRaw(self, center, radius, nn, raw):
        """Set shape from raw data. Default implementation raises error."""
        raise NotImplementedError(
            f"{self.toString()} does not implement Shape.setFromRaw"
        )

    def asRaw(self, center, radius, nn, raw):
        """Export shape as raw data. Default implementation raises error."""
        raise NotImplementedError(f"{self.toString()} does not implement Shape.asRaw")

    def setFromRaw_helper_nodeFromCoords(self, nn, raw, pos):
        """Helper to create/reuse node from raw coordinates."""
        if len(raw) < pos + 3:
            raise RuntimeError(f"Raw data too short (length {len(raw)}, pos={pos})")

        p = Vector3r(raw[pos], raw[pos + 1], raw[pos + 2])

        if math.isnan(p[0]) or math.isnan(p[1]):
            # Return existing node
            p2i = int(p[2])
            if not (p2i >= 0 and p2i < len(nn)):
                raise RuntimeError(
                    f"Raw coords beginning with NaN signify an existing node, but the "
                    f"index (z-component) is {p[2]} ({p2i} as int), which is not a valid index "
                    f"(0..{len(nn)-1}) of existing nodes."
                )
            return nn[p2i]
        else:

            n = Node()
            n.pos = p
            nn.append(n)
            return n

    def setFromRaw_helper_checkRaw_makeNodes(self, raw, numRaw):
        """Check raw data length and prepare node list."""
        if len(raw) != numRaw:
            raise RuntimeError(
                f"Error setting {self.toString()} from raw data: {numRaw} "
                f"numbers expected, {len(raw)} given."
            )

        # Add/remove nodes as necessary

        while len(self.nodes) < self.getNumNodes():
            self.nodes.append(Node())

        self.nodes = self.nodes[: self.getNumNodes()]

    def asRaw_helper_coordsFromNode(self, nn, raw, pos, nodeNum):
        """Helper to store node coordinates in raw data."""
        pass  # This method is missing from the provided C++ code

    def selfTest(self, particle):
        """Perform self-tests for correctness."""
        if not self.checkNumNodes():
            raise RuntimeError(
                f"Shape has incorrect number of nodes (expected {self.getNumNodes()}, "
                f"got {len(self.nodes)}) for particle #{particle.getId()}"
            )

        radius = self.getEquivalentRadius()
        volume = self.getVolume()

        # Multi-node shapes should return NaN for radius
        if len(self.nodes) != 1 and not math.isnan(radius):
            raise RuntimeError("Multi-node shape returning non-NaN equivalent radius")

        # Radius and volume should both be valid or both NaN
        if math.isnan(radius) != math.isnan(volume):
            raise RuntimeError(
                "Inconsistent radius/volume: both should be valid or both NaN"
            )

        # Valid radius/volume must be positive
        if not math.isnan(radius) and (radius <= 0 or volume <= 0):
            raise RuntimeError("Invalid radius/volume: must be positive when defined")


from pydem.src.DEMData import Node, DEMData
