from enum import IntEnum, auto
import threading
import weakref
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, Any
from abc import ABC, abstractmethod

# Import math utilities from demmath
from pydem.src.demmath import Vector3r, Quaternionr, Matrix3r, NAN, INF, Real, EPSILON
from pydem.src.Object import Object
from pydem.src.NodeData import NodeData


class Node(Object):
    """Represents a node in the DEM system."""

    class DataType(IntEnum):
        """Enum for data types stored in nodes."""

        DEM = 0
        CLUMP = 1
        MESH = 2
        SPARC = 3
        LAST = 4

    def __init__(self):
        super().__init__()
        self.pos = Vector3r(0.0, 0.0, 0.0)
        self.ori = Quaternionr(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        self.data = []  # List of NodeData objects

    def hasData(self, dataType: Union[DataType, int]) -> bool:
        """Check if node has data of the specified type."""
        index = int(dataType)
        return index < len(self.data) and self.data[index] is not None

    def setData(self, nd: NodeData, dataType: Union[DataType, int]) -> None:
        """Set data of the specified type."""
        index = int(dataType)
        if index >= len(self.data):
            # Resize data list if needed
            self.data.extend([None] * (index + 1 - len(self.data)))
        self.data[index] = nd

    def getData(self, dataType: Union[DataType, int]) -> NodeData:
        """Get data of the specified type."""
        index = int(dataType)
        if index >= len(self.data):
            raise IndexError("Invalid data type index")
        if self.data[index] is None:
            raise ValueError(f"No data of type {dataType} available")
        return self.data[index]

    # Type-safe data access templates converted to Python methods
    def getDataTyped(self, dataClass: Type) -> Any:
        """Get data of the specified class type."""
        if not issubclass(dataClass, NodeData):
            raise TypeError(f"Class {dataClass.__name__} must inherit from NodeData")

        # Get the associated data type index from the class
        class_name = dataClass.__name__.upper()
        dataType = getattr(NodeData.DataIndex, class_name, None)
        if dataType is None:
            raise ValueError(f"No DataType defined for {dataClass.__name__}")

        data = self.getData(dataType)
        if not isinstance(data, dataClass):
            raise TypeError(f"Expected {dataClass.__name__}, got {data.getClassName()}")
        return data

    def getDataPtr(self, dataClass: Type) -> Optional[Any]:
        """Get data pointer of the specified class type, or None if not present."""
        if not issubclass(dataClass, NodeData):
            raise TypeError(f"Class {dataClass.__name__} must inherit from NodeData")

        class_name = dataClass.__name__.upper()
        dataType = getattr(NodeData.DataIndex, class_name, None)
        if dataType is None:
            raise ValueError(f"No DataType defined for {dataClass.__name__}")

        if not self.hasData(dataType):
            return None

        data = self.getData(dataType)
        if not isinstance(data, dataClass):
            return None
        return data

    def setDataTyped(self, data: NodeData) -> None:
        """Set data by inferring the type from the class."""
        if not isinstance(data, NodeData):
            raise TypeError(f"Data must inherit from NodeData")

        dataClass = data.__class__
        class_name = dataClass.__name__.upper()
        dataType = getattr(NodeData.DataIndex, class_name, None)
        if dataType is None:
            raise ValueError(f"No DataType defined for {dataClass.__name__}")

        self.setData(data, dataType)

    def hasDataTyped(self, dataClass: Type) -> bool:
        """Check if node has data of the specified class type."""
        if not issubclass(dataClass, NodeData):
            raise TypeError(f"Class {dataClass.__name__} must inherit from NodeData")

        class_name = dataClass.__name__.upper()
        dataType = getattr(NodeData.DataIndex, class_name, None)
        if dataType is None:
            return False

        return self.hasData(dataType)

    # Coordinate transformations
    def glob2loc(self, p: Vector3r) -> Vector3r:
        """Transform global coordinates to local coordinates."""
        return self.ori.conjugate() * (p - self.pos)

    def loc2glob(self, p: Vector3r) -> Vector3r:
        """Transform local coordinates to global coordinates."""
        return self.ori * p + self.pos

    # Tensor transformations
    def glob2locRank2(self, g: Matrix3r) -> Matrix3r:
        """Transform rank-2 tensor from global to local coordinates."""
        R = self.ori.toRotationMatrix()
        return R.transpose() @ g @ R

    def loc2globRank2(self, l: Matrix3r) -> Matrix3r:
        """Transform rank-2 tensor from local to global coordinates."""
        R = self.ori.toRotationMatrix()
        return R @ l @ R.transpose()

    def toString(self) -> str:
        """Return string representation of this node."""
        return f"<{self.getClassName()} @ {id(self)}, pos({self.pos[0]},{self.pos[1]},{self.pos[2]})>"

    # Static helpers
    @staticmethod
    def getDataStatic(node: "Node", dataClass: Type) -> Optional[Any]:
        """Static helper to get data from a node."""
        if node is None:
            return None
        return node.getDataPtr(dataClass)

    @staticmethod
    def setDataStatic(node: "Node", data: NodeData) -> None:
        """Static helper to set data on a node."""
        if node is None:
            raise ValueError("Node cannot be None")
        node.setDataTyped(data)

    @property
    def dem(self):
        """Get DEM data for this node"""
        # Use lazy import to avoid circular dependency
        from pydem.src.DEMData import DEMData

        return self.getDataPtr(DEMData)

    @dem.setter
    def dem(self, value):
        """Set DEM data for this node"""
        self.setDataTyped(value)


# Register the DEMData class with NodeData.DataIndex
NodeData.DataIndex.DEMDATA = int(Node.DataType.DEM)
NodeData.DataIndex.CLUMPDATA = int(Node.DataType.CLUMP)
