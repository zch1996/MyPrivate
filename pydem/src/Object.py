class Object:
    """Base class providing common object functionality."""

    def __init__(self):
        pass

    def getClassName(self) -> str:
        """Returns the class name of this instance."""
        return self.__class__.__name__

    def toString(self) -> str:
        """Returns string representation, can be overridden by subclasses."""
        return self.getClassName()

    def isA(self, classType) -> bool:
        """Checks if this object is an instance of the given class type."""
        return isinstance(self, classType)

    def cast(self, classType):
        """Returns this object as the given type if possible, otherwise raises TypeError."""
        if not self.isA(classType):
            raise TypeError(
                f"Cannot cast {self.getClassName()} to {classType.__name__}"
            )
        return self
