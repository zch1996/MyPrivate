#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .Engine import Engine
from .FunctorFactory import FunctorFactory
from typing import Dict, List, Optional, Type, Any, Tuple, Union
from .DEMLogging import DEM_LOGGER


class Dispatcher(Engine):
    """Base class for all dispatchers."""

    def __init__(self):
        """Initialize dispatcher with default values."""
        super().__init__()
        self.functorCache = {}

    def updateScenePtr(self, scene, field):
        """Update scene and field pointers."""
        super().updateScenePtr(scene, field)
        self.updateFunctors(scene, field)

    def updateFunctors(self, scene, field):
        """Update scene and field pointers for all functors."""
        for functor in self.functorCache.values():
            functor.updateScenePtr(scene, field)

    def initializeFunctors(self):
        """Initialize functors from factory."""
        raise NotImplementedError(
            "Dispatcher.initializeFunctors() must be implemented by subclasses"
        )


@DEM_LOGGER
class BoundDispatcher(Dispatcher):
    """Dispatcher for bound functors."""

    def __init__(self):
        """Initialize bound dispatcher."""
        super().__init__()
        self.functorCache = {}  # {ShapeType: BoundFunctor}

    def initializeFunctors(self):
        """Initialize bound functors from factory."""
        factory = FunctorFactory.instance()
        for shapeType in factory.getRegisteredShapeTypes():
            functor = factory.createBoundFunctor(shapeType)
            if functor:
                self.functorCache[shapeType] = functor
                self.debug(f"Initialized BoundFunctor for shape type {shapeType}")

    def __call__(self, shape):
        """Compute bound for a shape."""
        shapeType = shape.getShapeType()
        functor = self.getFunctor(shapeType)
        if functor:
            functor.go(shape)
        else:
            self.error(f"No BoundFunctor available for shape type {shapeType}")

    def getFunctor(self, shapeType):
        """Get bound functor for a shape type."""
        if shapeType in self.functorCache:
            return self.functorCache[shapeType]

        # Try to create on demand
        functor = FunctorFactory.instance().createBoundFunctor(shapeType)
        if functor:
            functor.updateScenePtr(self.scene, self.field)
            self.functorCache[shapeType] = functor
            return functor

        return None


@DEM_LOGGER
class IntraDispatcher(Dispatcher):
    """Dispatcher for intra functors."""

    def __init__(self):
        """Initialize intra dispatcher."""
        super().__init__()
        self.functorCache = {}  # {(ShapeType, MaterialType): IntraFunctor}

    def initializeFunctors(self):
        """Initialize intra functors from factory."""
        factory = FunctorFactory.instance()
        for shapeType in factory.getRegisteredShapeTypes():
            for matType in factory.getRegisteredMaterialTypes():
                functor = factory.createIntraFunctor(shapeType, matType)
                if functor:
                    self.functorCache[(shapeType, matType)] = functor
                    self.debug(
                        f"Initialized IntraFunctor for shape type {shapeType} and material type {matType}"
                    )

    def addIntraStiffness(self, particle, node, ktrans, krot):
        """Add internal stiffness contributions."""
        shape = particle.getShape()
        material = particle.getMaterial()

        if not shape or not material:
            return

        functor = self.getFunctor(shape.getShapeType(), material.getMaterialType())
        if functor:
            functor.addIntraStiffnesses(particle, node, ktrans, krot)

    def __call__(self, shape, material, particle):
        """Process internal forces for a particle."""
        functor = self.getFunctor(shape.getShapeType(), material.getMaterialType())
        if functor:
            functor.go(shape, material, particle)

    def getFunctor(self, shapeType, materialType):
        """Get intra functor for a shape and material type."""
        key = (shapeType, materialType)
        if key in self.functorCache:
            return self.functorCache[key]

        # Try to create on demand
        functor = FunctorFactory.instance().createIntraFunctor(shapeType, materialType)
        if functor:
            functor.updateScenePtr(self.scene, self.field)
            self.functorCache[key] = functor
            return functor

        return None


@DEM_LOGGER
class CGeomDispatcher(Dispatcher):
    """Dispatcher for contact geometry functors."""

    def __init__(self):
        """Initialize contact geometry dispatcher."""
        super().__init__()
        self.functorCache = {}  # {(ShapeType, ShapeType): (CGeomFunctor, bool)}

    def initializeFunctors(self):
        """Initialize contact geometry functors from factory."""
        factory = FunctorFactory.instance()
        for shape1 in factory.getRegisteredShapeTypes():
            for shape2 in factory.getRegisteredShapeTypes():
                functor, swap = factory.createGeomFunctor(shape1, shape2)
                if functor:
                    self.functorCache[(shape1, shape2)] = functor
                    self.debug(
                        f"Initialized CGeomFunctor for shape types {shape1} and {shape2}"
                    )

    def getFunctor(self, shape1Type, shape2Type):
        """Get contact geometry functor for two shape types."""
        key = (shape1Type, shape2Type)
        if key in self.functorCache:
            return self.functorCache[key], True

        # Try to create on demand
        functor, swap = FunctorFactory.instance().createGeomFunctor(
            shape1Type, shape2Type
        )
        if functor:
            functor.updateScenePtr(self.scene, self.field)
            self.functorCache[key] = (functor, swap)
            return functor, swap

        return None, False


@DEM_LOGGER
class CPhysDispatcher(Dispatcher):
    """Dispatcher for contact physics functors."""

    def __init__(self):
        """Initialize contact physics dispatcher."""
        super().__init__()
        self.functorCache = {}  # {(MaterialType, MaterialType): (CPhysFunctor, bool)}

    def initializeFunctors(self):
        """Initialize contact physics functors from factory."""
        factory = FunctorFactory.instance()
        for mat1 in factory.getRegisteredMaterialTypes():
            for mat2 in factory.getRegisteredMaterialTypes():
                functor, swap = factory.createPhysFunctor(mat1, mat2)
                if functor:
                    self.functorCache[(mat1, mat2)] = functor
                    self.debug(
                        f"Initialized CPhysFunctor for material types {mat1} and {mat2}"
                    )

    def explicitAction(self, scene, m1, m2, contact):
        """Explicitly apply contact physics."""
        self.updateScenePtr(scene, self.field)

        if not contact.geom:
            raise ValueError("Contact has no geometry")

        functor, swap = self.getFunctor(m1.getMaterialType(), m2.getMaterialType())
        if not functor:
            raise ValueError(
                f"No suitable physics functor for materials: {m1.getClassName()}, {m2.getClassName()}"
            )

        if swap:
            m1, m2 = m2, m1

        functor.go(m1, m2, contact)

    def getFunctor(self, mat1Type, mat2Type):
        """Get contact physics functor for two material types."""
        key = (mat1Type, mat2Type)
        if key in self.functorCache:
            return self.functorCache[key][0], self.functorCache[key][1]

        # Try to create on demand
        functor, swap = FunctorFactory.instance().createPhysFunctor(mat1Type, mat2Type)
        if functor:
            functor.updateScenePtr(self.scene, self.field)
            self.functorCache[key] = (functor, swap)
            return functor, swap

        return None, False


@DEM_LOGGER
class LawDispatcher(Dispatcher):
    """Dispatcher for contact law functors."""

    def __init__(self):
        """Initialize contact law dispatcher."""
        super().__init__()
        self.functorCache = {}  # {(GeomType, PhysType): LawFunctor}

    def initializeFunctors(self):
        """Initialize contact law functors from factory."""
        factory = FunctorFactory.instance()
        for geomType in factory.getRegisteredContactGeomTypes():
            for physType in factory.getRegisteredContactPhysTypes():
                for modelType in factory.getRegisteredLawmodelTypes():
                    functor = factory.createLawFunctor(geomType, physType, modelType)

                    if functor:
                        self.functorCache[(geomType, physType)] = functor
                        self.debug(
                            f"Initialized LawFunctor for geom type {geomType} and phys type {physType}"
                        )

    def getFunctor(self, geom, phys):
        """
        Get contact law functor for a geometry and physics type.

        Args:
            geom: Contact geometry
            phys: Contact physics

        Returns:
            Functor for the given geometry and physics types
        """
        key = (geom.getContactGeomType(), phys.getContactPhysType())

        # Check if functor is already in cache
        if key in self.functorCache:
            return self.functorCache[key]

        # Create new functor
        functor = FunctorFactory.instance().createLawFunctor(geom, phys)
        if functor:
            # Update functor and add to cache (cache is modified in-place)
            functor.updateScenePtr(self.scene, self.field)
            self.functorCache[key] = functor
            return functor

        return None
