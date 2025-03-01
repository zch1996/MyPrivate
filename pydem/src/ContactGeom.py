#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional, List, Dict, Tuple, Any
import numpy as np

from pydem.src.Object import Object
from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr, Real, NAN


class ContactGeom(Object):
    """Base class for contact geometry."""

    class Type:
        """Contact geometry types."""

        NONE = 0
        L6GEOM = 1
        G3GEOM = 2
        # Add other types as needed

    def __init__(self):
        """Initialize ContactGeom with default values."""
        super().__init__()
        self.contactGeomType = self.Type.NONE
        self._node = None  # Node for visualization and force application

    def getNode(self):
        """
        Get associated node.

        Returns:
            Node object
        """
        if not self._node:
            from Node import Node

            self._node = Node()
        return self._node

    def getContactGeomType(self):
        """
        Get contact geometry type.

        Returns:
            Contact geometry type
        """
        return self.contactGeomType

    def cast(self, cls):
        """
        Cast to specific geometry type.

        Args:
            cls: Class to cast to

        Returns:
            Self as the specified class

        Raises:
            TypeError: If self is not an instance of cls
        """
        if isinstance(self, cls):
            return self
        else:
            raise TypeError(f"Cannot cast {self.__class__.__name__} to {cls.__name__}")

    def toString(self) -> str:
        """Return string representation."""
        return f"{self.getClassName()}"
