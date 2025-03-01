#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import Dict, List, Type, Tuple, Optional, Any
from .DEMLogging import DEM_LOGGER


@DEM_LOGGER
class FunctorFactory:
    """Factory for creating and managing functors."""

    _instance = None
    _lock = threading.RLock()

    @classmethod
    def instance(cls):
        """Get the singleton instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize the functor factory."""
        if FunctorFactory._instance is not None:
            raise RuntimeError(
                "FunctorFactory is a singleton class, use FunctorFactory.instance() instead"
            )

        # Dictionaries to store registered functors by type
        self.boundFunctors = {}  # {shape_type: functor_class}
        self.geomFunctors = {}  # {(shape1_type, shape2_type): functor_class}
        self.physFunctors = {}  # {(mat1_type, mat2_type): functor_class}
        self.lawFunctors = {}  # {(geom_type, phys_type): functor_class}
        self.intraFunctors = {}  # {(shape_type, mat_type): functor_class}

        # Store registered types
        self.shapeTypes = set()
        self.materialTypes = set()
        self.contactGeomTypes = set()
        self.contactPhysTypes = set()
        self.lawmodelTypes = set()

        self.info("FunctorFactory initialized")

    def registerBoundFunctor(self, shape_type, functor_class):
        """
        Register a bound functor class for a shape type.

        Args:
            shape_type: The shape type this functor handles
            functor_class: The functor class to register
        """
        # 确保不会覆盖已有的注册
        if shape_type in self.boundFunctors:
            self.warning(
                f"Overriding existing bound functor for {shape_type.__name__}: "
                f"{self.boundFunctors[shape_type].__name__} -> {functor_class.__name__}"
            )

        self.boundFunctors[shape_type] = functor_class
        self.shapeTypes.add(shape_type)
        self.debug(
            f"Registered BoundFunctor: {functor_class.__name__} for shape {shape_type.__name__}"
        )

    def registerGeomFunctor(self, shape1_type, shape2_type, functor_class):
        """
        Register a geometry functor class for a pair of shape types.

        Args:
            shape1_type: First shape type
            shape2_type: Second shape type
            functor_class: The functor class to register
        """
        key = (shape1_type, shape2_type)
        self.geomFunctors[key] = functor_class
        self.shapeTypes.add(shape1_type)
        self.shapeTypes.add(shape2_type)
        self.debug(
            f"Registered CGeomFunctor: {functor_class.__name__} for shapes {shape1_type.__name__}, {shape2_type.__name__}"
        )

    def registerPhysFunctor(self, mat1_type, mat2_type, functor_class):
        """
        Register a physics functor class for a pair of material types.

        Args:
            mat1_type: First material type
            mat2_type: Second material type
            functor_class: The functor class to register
        """
        key = (mat1_type, mat2_type)
        self.physFunctors[key] = functor_class
        self.materialTypes.add(mat1_type)
        self.materialTypes.add(mat2_type)
        self.debug(
            f"Registered CPhysFunctor: {functor_class.__name__} for materials {mat1_type.__name__}, {mat2_type.__name__}"
        )

    def registerLawFunctor(self, geom_type, phys_type, model_type, functor_class):
        """
        Register a law functor class for a geometry and physics type.

        Args:
            geom_type: Geometry type
            phys_type: Physics type
            functor_class: The functor class to register
        """
        key = (geom_type, phys_type, model_type)
        self.lawFunctors[key] = functor_class
        self.contactGeomTypes.add(geom_type)
        self.contactPhysTypes.add(phys_type)
        self.lawmodelTypes.add(model_type)
        self.debug(
            f"Registered LawFunctor: {functor_class.__name__} for geom {geom_type.__name__}, phys {phys_type.__name__}"
        )

    def registerIntraFunctor(self, shape_type, mat_type, functor_class):
        """
        Register an intra functor class for a shape and material type.

        Args:
            shape_type: Shape type
            mat_type: Material type
            functor_class: The functor class to register
        """
        key = (shape_type, mat_type)
        self.intraFunctors[key] = functor_class
        self.shapeTypes.add(shape_type)
        self.materialTypes.add(mat_type)
        self.debug(
            f"Registered IntraFunctor: {functor_class.__name__} for shape {shape_type.__name__}, material {mat_type.__name__}"
        )

    # Methods to create functors
    def createBoundFunctor(self, shape_type):
        """Create a bound functor for a shape type."""
        functor_class = self.boundFunctors.get(shape_type)
        if functor_class:
            return functor_class()
        return None

    def createGeomFunctor(self, shape1_type, shape2_type):
        """Create a geometry functor for shape types."""
        key = (shape1_type, shape2_type)
        functor_class = self.geomFunctors.get(key)

        if functor_class:
            return functor_class(), False

        # Try reverse order
        key = (shape2_type, shape1_type)
        functor_class = self.geomFunctors.get(key)

        if functor_class:
            return functor_class(), True

        return None, False

    def createPhysFunctor(self, mat1_type, mat2_type):
        """Create a physics functor for material types."""
        key = (mat1_type, mat2_type)
        functor_class = self.physFunctors.get(key)

        if functor_class:
            return functor_class(), False

        # Try reverse order
        key = (mat2_type, mat1_type)
        functor_class = self.physFunctors.get(key)

        if functor_class:
            return functor_class(), True

        return None, False

    def createLawFunctor(self, geom_type, phys_type, model_type):
        """Create a law functor for geometry and physics types."""
        key = (geom_type, phys_type, model_type)
        functor_class = self.lawFunctors.get(key)

        if functor_class:
            return functor_class()

        return None

    def createIntraFunctor(self, shape_type, mat_type):
        """Create an intra functor for shape and material types."""
        key = (shape_type, mat_type)
        functor_class = self.intraFunctors.get(key)

        if functor_class:
            return functor_class()

        return None

    # Accessor methods
    def getRegisteredShapeTypes(self):
        """Get all registered shape types."""
        return list(self.shapeTypes)

    def getRegisteredMaterialTypes(self):
        """Get all registered material types."""
        return list(self.materialTypes)

    def getRegisteredContactGeomTypes(self):
        """Get all registered contact geometry types."""
        return list(self.contactGeomTypes)

    def getRegisteredContactPhysTypes(self):
        """Get all registered contact physics types."""
        return list(self.contactPhysTypes)

    def getRegisteredLawmodelTypes(self):
        """Get all registered law model types."""
        return list(self.lawmodelTypes)
