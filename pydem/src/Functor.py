#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Object import Object
from typing import Dict, List, Optional, Type, Any, Tuple, Union
from enum import Enum, auto
from pydem.src.DEMLogging import DEM_LOGGER


@DEM_LOGGER
class Functor(Object):
    """Base class for all functors in the DEM system."""

    def __init__(self):
        """Initialize functor with default values."""
        super().__init__()
        self.scene = None
        self.field = None

    def updateScenePtr(self, scene, field):
        """Update scene and field pointers."""
        self.scene = scene
        self.field = field

    @classmethod
    def registerClass(cls):
        """Register this functor class with the FunctorFactory."""
        # Import here to avoid circular imports
        from pydem.src.FunctorFactory import FunctorFactory

        factory = FunctorFactory.instance()

        # 确定函数子类型并相应地注册
        if cls.__name__.startswith("Bo1_"):
            # 边界函数子
            parts = cls.__name__.split("_")
            if len(parts) >= 3:
                shape_name = parts[1]
                try:
                    # 动态导入形状类
                    shape_module = __import__(
                        f"pydem.src.{shape_name}", fromlist=[shape_name]
                    )
                    shape_class = getattr(shape_module, shape_name)
                    factory.registerBoundFunctor(shape_class, cls)
                    cls.debug(
                        f"Registered bound functor: {cls.__name__} for shape {shape_class.__name__}"
                    )
                except (ImportError, AttributeError) as e:
                    cls.error(f"Could not register {cls.__name__}: {e}")
            else:
                cls.error(f"Invalid bound functor name format: {cls.__name__}")

        elif cls.__name__.startswith("Cg2_"):
            # 几何函数子
            parts = cls.__name__.split("_")
            if len(parts) >= 4:
                shape1_name = parts[1]
                shape2_name = parts[2]
                try:
                    shape1_module = __import__(
                        f"pydem.src.{shape1_name}", fromlist=[shape1_name]
                    )
                    shape1_class = getattr(shape1_module, shape1_name)
                    shape2_module = __import__(
                        f"pydem.src.{shape2_name}", fromlist=[shape2_name]
                    )
                    shape2_class = getattr(shape2_module, shape2_name)
                    factory.registerGeomFunctor(shape1_class, shape2_class, cls)
                    cls.debug(f"Registered geometry functor: {cls.__name__}")
                except (ImportError, AttributeError) as e:
                    cls.error(f"Could not register {cls.__name__}: {e}")
            else:
                cls.error(f"Invalid geometry functor name format: {cls.__name__}")

        elif cls.__name__.startswith("Cp2_"):
            parts = cls.__name__.split("_")
            if len(parts) >= 3:
                mat_name = parts[1]
                phys_name = parts[2]
                try:
                    mat_module = __import__(f"pydem.src.Material", fromlist=[mat_name])
                    mat_class = getattr(mat_module, mat_name)
                    phys_module = __import__(
                        f"pydem.src.ContactPhys", fromlist=[phys_name]
                    )
                    phys_class = getattr(phys_module, phys_name)
                    factory.registerPhysFunctor(mat_class, phys_class, cls)
                    cls.debug(f"Registered physics functor: {cls.__name__}")
                except (ImportError, AttributeError) as e:
                    cls.error(f"Could not register {cls.__name__}: {e}")
            else:
                cls.error(f"Invalid physics functor name format: {cls.__name__}")

        elif cls.__name__.startswith("Law2_"):
            parts = cls.__name__.split("_")
            if len(parts) >= 4:
                geom_name = parts[1]
                phys_name = parts[2]
                model_name = parts[3]
                try:
                    geom_module = __import__(f"pydem.src.L6Geom", fromlist=[geom_name])
                    geom_class = getattr(geom_module, geom_name)
                    phys_module = __import__(
                        f"pydem.src.ContactPhys", fromlist=[phys_name]
                    )
                    phys_class = getattr(phys_module, phys_name)
                    # model_module = __import__(
                    #     f"pydem.src.ContactModel", fromlist=[model_name]
                    # )
                    # model_class = getattr(model_module, model_name)
                    factory.registerLawFunctor(geom_class, phys_class, model_name, cls)
                    cls.debug(f"Registered law functor: {cls.__name__}")
                except (ImportError, AttributeError) as e:
                    cls.error(f"Could not register {cls.__name__}: {e}")
            else:
                cls.error(f"Invalid law functor name format: {cls.__name__}")

        return cls


class BoundFunctor(Functor):
    """Functor for computing bounding volumes of shapes."""

    def go(self, shape):
        """Compute bounding volume for a shape."""
        raise NotImplementedError("BoundFunctor.go() must be implemented by subclasses")


class IntraFunctor(Functor):
    """Functor for handling internal forces within particles."""

    def addIntraStiffnesses(self, particle, node, ktrans, krot):
        """Add internal stiffness contributions."""
        raise NotImplementedError(
            "IntraFunctor.addIntraStiffnesses() must be implemented by subclasses"
        )

    def go(self, shape, material, particle):
        """Process internal forces for a particle."""
        raise NotImplementedError("IntraFunctor.go() must be implemented by subclasses")


class CGeomFunctor(Functor):
    """Functor for computing contact geometry."""

    def go(self, shape1, shape2, shift2, force, contact):
        """Compute contact geometry between two shapes."""
        raise NotImplementedError("CGeomFunctor.go() must be implemented by subclasses")

    def goReverse(self, shape1, shape2, shift2, force, contact):
        """Compute contact geometry with shapes in reverse order."""
        raise NotImplementedError(
            "CGeomFunctor.goReverse() must be implemented by subclasses"
        )

    def setMinDist00Sq(self, shape1, shape2, contact):
        """Set minimum distance squared between shapes."""
        pass  # Optional implementation


class CPhysFunctor(Functor):
    """Functor for computing contact physics."""

    def go(self, material1, material2, contact):
        """Compute contact physics between two materials."""
        raise NotImplementedError("CPhysFunctor.go() must be implemented by subclasses")


class LawFunctor(Functor):
    """Functor for computing contact laws."""

    def go(self, geom, phys, contact):
        """Apply contact law to a contact."""
        raise NotImplementedError("LawFunctor.go() must be implemented by subclasses")
