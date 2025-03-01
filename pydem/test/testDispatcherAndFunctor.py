#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Dispatcher and Functor classes in PyDEM.
This script tests the functionality of:
- Functor base classes
- Dispatcher classes
- ContactGeom functors
- ContactPhys functors
- Bound functors
- Sphere-specific functors
"""

import sys
import os
import numpy as np

# Add parent directory to path to import PyDEM modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Functor import (
    Functor,
    BoundFunctor,
    CGeomFunctor,
    CPhysFunctor,
    LawFunctor,
    IntraFunctor,
)
from Dispatcher import (
    Dispatcher,
    BoundDispatcher,
    CGeomDispatcher,
    CPhysDispatcher,
    LawDispatcher,
)
from FunctorFactory import FunctorFactory
from ContactGeomFunctor import Cg2_Any_Any_L6Geom__Base
from Sphere import (
    Sphere,
    Bo1_Sphere_Aabb,
    In2_Sphere_ElastMat,
    Cg2_Sphere_Sphere_L6Geom,
)
from ContactPhysFunctor import Cp2_FrictMat_FrictPhys
from L6Geom import L6Geom
from G3Geom import G3Geom
from ContactGeom import ContactGeom
from Material import Material, FrictMat
from Contact import Contact
from Aabb import Aabb
from Node import Node
from Particle import Particle
from Scene import Scene
from DEMField import DEMField
from demmath import Vector3r, Matrix3r, Quaternionr, Real, Vector2r
from DEMData import DEMData


def print_separator(title):
    """Print a separator with a title."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def test_functor_creation():
    """Test creation of functor base classes."""
    print_separator("Testing Functor Creation")

    # Create functors
    bound_functor = BoundFunctor()
    geom_functor = CGeomFunctor()
    phys_functor = CPhysFunctor()
    law_functor = LawFunctor()
    intra_functor = IntraFunctor()

    # Print functor types
    print(f"BoundFunctor: {bound_functor}")
    print(f"CGeomFunctor: {geom_functor}")
    print(f"CPhysFunctor: {phys_functor}")
    print(f"LawFunctor: {law_functor}")
    print(f"IntraFunctor: {intra_functor}")

    # Check inheritance
    print("\nChecking inheritance:")
    print(f"BoundFunctor is a Functor: {isinstance(bound_functor, Functor)}")
    print(f"CGeomFunctor is a Functor: {isinstance(geom_functor, Functor)}")
    print(f"CPhysFunctor is a Functor: {isinstance(phys_functor, Functor)}")
    print(f"LawFunctor is a Functor: {isinstance(law_functor, Functor)}")
    print(f"IntraFunctor is a Functor: {isinstance(intra_functor, Functor)}")


def test_dispatcher_creation(scene, dem_field):
    """Test creation of dispatcher classes."""
    print_separator("Testing Dispatcher Creation")

    # Create dispatchers
    bound_dispatcher = BoundDispatcher()
    geom_dispatcher = CGeomDispatcher()
    phys_dispatcher = CPhysDispatcher()
    law_dispatcher = LawDispatcher()

    # Print dispatcher types
    print(f"BoundDispatcher: {bound_dispatcher}")
    print(f"CGeomDispatcher: {geom_dispatcher}")
    print(f"CPhysDispatcher: {phys_dispatcher}")
    print(f"LawDispatcher: {law_dispatcher}")

    # Check inheritance
    print("\nChecking inheritance:")
    print(
        f"BoundDispatcher is a Dispatcher: {isinstance(bound_dispatcher, Dispatcher)}"
    )
    print(f"CGeomDispatcher is a Dispatcher: {isinstance(geom_dispatcher, Dispatcher)}")
    print(f"CPhysDispatcher is a Dispatcher: {isinstance(phys_dispatcher, Dispatcher)}")
    print(f"LawDispatcher is a Dispatcher: {isinstance(law_dispatcher, Dispatcher)}")

    # Initialize dispatchers
    print("\nInitializing dispatchers with scene and field...")
    bound_dispatcher.scene = scene
    bound_dispatcher.field = dem_field
    geom_dispatcher.scene = scene
    geom_dispatcher.field = dem_field
    phys_dispatcher.scene = scene
    phys_dispatcher.field = dem_field
    law_dispatcher.scene = scene
    law_dispatcher.field = dem_field

    # Initialize functors
    print("Initializing functors...")
    bound_dispatcher.initializeFunctors()
    geom_dispatcher.initializeFunctors()
    phys_dispatcher.initializeFunctors()
    law_dispatcher.initializeFunctors()

    # Check functor caches
    print(f"BoundDispatcher functor cache size: {len(bound_dispatcher.functorCache)}")
    print(f"CGeomDispatcher functor cache size: {len(geom_dispatcher.functorCache)}")
    print(f"CPhysDispatcher functor cache size: {len(phys_dispatcher.functorCache)}")
    print(f"LawDispatcher functor cache size: {len(law_dispatcher.functorCache)}")


def test_contact_geom_functor(scene, dem_field):
    """Test ContactGeomFunctor classes."""
    print_separator("Testing ContactGeomFunctor")

    # Create spheres
    s1 = Sphere()
    s1.radius = 1.0
    s1.nodes.append(Node())
    s1.nodes[0].pos = Vector3r(0, 0, 0)

    s2 = Sphere()
    s2.radius = 1.0
    s2.nodes.append(Node())
    s2.nodes[0].pos = Vector3r(1.5, 0, 0)

    # Create contact
    contact = Contact()

    # Create functor
    functor = Cg2_Sphere_Sphere_L6Geom()
    functor.scene = scene
    functor.field = dem_field

    # Add DEM data to nodes
    s1.nodes[0].setDataTyped(DEMData())
    s2.nodes[0].setDataTyped(DEMData())

    print(f"Created spheres with radii {s1.radius} and {s2.radius}")
    print(f"Sphere 1 position: {s1.nodes[0].pos}")
    print(f"Sphere 2 position: {s2.nodes[0].pos}")
    print(
        f"Distance between centers: {np.linalg.norm(s2.nodes[0].pos - s1.nodes[0].pos)}"
    )
    print(
        f"Expected overlap: {s1.radius + s2.radius - np.linalg.norm(s2.nodes[0].pos - s1.nodes[0].pos)}"
    )

    print(f"ContactGeom should be None:", contact.geom is None)

    # Process contact
    print("\nProcessing contact...")
    result = functor.go(s1, s2, Vector3r(0, 0, 0), False, contact)

    # Check result
    print(f"Contact created: {result}")
    print(f"Contact geometry exists: {contact.geom is not None}")
    if contact.geom:
        print(f"Contact geometry type: {contact.geom.getContactGeomType()}")
        print(f"Normal overlap (uN): {contact.geom.uN}")
        print(f"Reference lengths: {contact.geom.lens}")

        # Check normal direction
        normal = contact.geom.trsf[:, 0]
        print(f"Contact normal: {normal}")


def test_contact_phys_functor(scene, dem_field):
    """Test ContactPhysFunctor classes."""
    print_separator("Testing ContactPhysFunctor")

    # Create materials
    mat1 = FrictMat()
    mat1.young = 1e7
    mat1.ktDivKn = 0.3
    mat1.tanPhi = 0.5

    mat2 = FrictMat()
    mat2.young = 2e7
    mat2.ktDivKn = 0.4
    mat2.tanPhi = 0.6

    # Create contact with geometry
    contact = Contact()
    contact.geom = L6Geom()
    contact.geom.lens = Vector2r(1.0, 1.0)
    contact.geom.contA = 3.14159  # π

    print(f"Created materials with Young's moduli {mat1.young} and {mat2.young}")
    print(f"Material 1 ktDivKn: {mat1.ktDivKn}, tanPhi: {mat1.tanPhi}")
    print(f"Material 2 ktDivKn: {mat2.ktDivKn}, tanPhi: {mat2.tanPhi}")
    print(f"Contact geometry lens: {contact.geom.lens}, area: {contact.geom.contA}")

    # Create functor
    functor = Cp2_FrictMat_FrictPhys()
    functor.scene = scene
    functor.field = dem_field

    # Process contact
    print("\nProcessing contact...")
    functor.go(mat1, mat2, contact)

    # Check result
    print(f"Contact physics exists: {contact.phys is not None}")
    if contact.phys:
        # Calculate expected values
        expected_kn = 1 / (
            1 / (mat1.young * contact.geom.contA / contact.geom.lens[0])
            + 1 / (mat2.young * contact.geom.contA / contact.geom.lens[1])
        )
        expected_kt = 0.5 * (mat1.ktDivKn + mat2.ktDivKn) * expected_kn
        expected_tanPhi = min(mat1.tanPhi, mat2.tanPhi)

        print(f"Normal stiffness (kn): {contact.phys.kn} (expected: {expected_kn})")
        print(f"Tangential stiffness (kt): {contact.phys.kt} (expected: {expected_kt})")
        print(
            f"Friction angle tangent (tanPhi): {contact.phys.tanPhi} (expected: {expected_tanPhi})"
        )


def test_bound_functor(scene, dem_field):
    """Test BoundFunctor classes."""
    print_separator("Testing BoundFunctor")

    # Set distance factor
    dem_field.distFactor = 0.1  # 10% margin

    # Create sphere
    sphere = Sphere()
    sphere.radius = 1.0
    sphere.nodes.append(Node())
    sphere.nodes[0].pos = Vector3r(0, 0, 0)

    print(f"Created sphere with radius {sphere.radius}")
    print(f"Sphere position: {sphere.nodes[0].pos}")
    print(f"Distance factor: {dem_field.distFactor}")

    # Create functor
    functor = Bo1_Sphere_Aabb()
    functor.scene = scene
    functor.field = dem_field

    # Process bound
    print("\nProcessing bound...")
    functor.go(sphere)

    # Check result
    print(f"Bound exists: {sphere.bound is not None}")
    if sphere.bound:
        print(f"Bound type: {sphere.bound.__class__.__name__}")

        # Calculate expected values
        margin = dem_field.distFactor * sphere.radius
        expected_min = Vector3r(
            -sphere.radius - margin,
            -sphere.radius - margin,
            -sphere.radius - margin,
        )
        expected_max = Vector3r(
            sphere.radius + margin,
            sphere.radius + margin,
            sphere.radius + margin,
        )

        print(f"Bound min: {sphere.bound.min} (expected: {expected_min})")
        print(f"Bound max: {sphere.bound.max} (expected: {expected_max})")


def main():
    """Main function to run all tests."""
    print_separator("PyDEM Dispatcher and Functor Test")

    # Create scene and field
    scene = Scene()
    dem_field = DEMField()
    dem_field.scene = scene
    scene.fields.append(dem_field)

    # Register functors with factory
    factory = FunctorFactory.instance()
    factory.registerBoundFunctor("Sphere", Bo1_Sphere_Aabb)
    factory.registerCGeomFunctor("Sphere", "Sphere", Cg2_Sphere_Sphere_L6Geom)
    factory.registerCPhysFunctor("FrictMat", "FrictMat", Cp2_FrictMat_FrictPhys)

    # Run tests
    test_functor_creation()
    test_dispatcher_creation(scene, dem_field)
    test_contact_geom_functor(scene, dem_field)
    test_contact_phys_functor(scene, dem_field)
    test_bound_functor(scene, dem_field)

    print_separator("All Tests Completed")


if __name__ == "__main__":
    main()
