from pydem.src.Object import Object
import threading


class NodeData(Object):
    """Base class for node-specific data."""

    class DataIndex:

        DEM = 0
        MESH = 1
        SPARC = 2

    def __init__(self):
        super().__init__()
        self.mutex = threading.RLock()  # Recursive mutex for thread safety

    def getterName(self) -> str:
        """Return the getter name for this data type."""
        raise NotImplementedError(
            f"{self.toString()} does not implement NodeData::getterName"
        )

    def setDataOnNode(self, node: "Node") -> None:
        """Set this data on the given node."""
        raise NotImplementedError(
            f"{self.toString()} does not implement NodeData::setDataOnNode"
        )

    # Define accessor methods as needed
    def toString(self) -> str:
        """Return string representation."""
        return f"{self.getClassName()}"
