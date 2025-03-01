#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ContactLoop import ContactLoop, UpdatePhys
from Dispatcher import CGeomDispatcher, CPhysDispatcher, LawDispatcher
from ContactHook import ContactHook
from Scene import Scene
from DEMField import DEMField
from Particle import Particle
from Sphere import *
from Material import FrictMat
from ContactPhysFunctor import Cp2_FrictMat_FrictPhys
from FunctorFactory import FunctorFactory
from demmath import Vector3r, Matrix3r
import numpy as np
import time
from LawFunctor import Law2_L6Geom_FrictPhys_IdealElPl


class TestContactHook(ContactHook):
    """Test implementation of ContactHook for testing."""

    def __init__(self):
        super().__init__()
        self.newContacts = []
        self.delContacts = []
        self.mask = 0xFFFF  # Match all particles

    def hookNew(self, dem, contact):
        """Record new contacts."""
        self.newContacts.append(contact)
        print(f"New contact: {contact.toString()}")

    def hookDel(self, dem, contact):
        """Record deleted contacts."""
        self.delContacts.append(contact)
        print(f"Deleted contact: {contact.toString()}")


def setup_functors():
    """Register functors with the factory."""
    factory = FunctorFactory.instance()

    factory.registerBoundFunctor("Sphere", Bo1_Sphere_Aabb)
    factory.registerCGeomFunctor("Sphere", "Sphere", Cg2_Sphere_Sphere_L6Geom)
    factory.registerCPhysFunctor("FrictMat", "FrictMat", Cp2_FrictMat_FrictPhys)
    factory.registerLawFunctor("L6Geom", "FrictPhys", Law2_L6Geom_FrictPhys_IdealElPl)


def create_particles(dem, count=2, distance=1.0):
    """Create test particles."""
    particles = []
    steel = FrictMat(density=7800.0, young=2e11, tanPhi=0.3)

    for i in range(count):
        # Create a steel sphere
        sp = Sphere()
        sp.radius = 0.1
        steel_sphere = Particle.make(sp, steel)
        steel_sphere.setPos(Vector3r(0.0, 0.0, i * distance))

        particles.append(steel_sphere)

    return particles


def test_contact_loop_basic():
    """Test basic functionality of ContactLoop."""
    print("\n=== Testing ContactLoop Basic Functionality ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5

    # Setup functors
    setup_functors()

    # Create particles
    particles = create_particles(dem, 2, 0.9)  # Slightly overlapping

    # Print particle information
    for p in particles:
        print(f"Particle {p.getId()}:")
        print(f"  Type: {p.getShape().toString()}")
        print(f"  Position: {p.getPos()}")
        print(f"  Velocity: {p.getVel()}")
        print(f"  Mass: {p.getMass()}")
        print(f"  Material: {p.getMaterial().getMaterialType()}")
        print(
            f"  Fixed: {'Yes' if p.getShape().getNodes()[0].getDataTyped(DEMData).isBlockedAll() else 'No'}"
        )

    # Add particles to DEM field properly
    for p in particles:
        dem.particles.add(p)

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Create contact loop
    loop = ContactLoop()
    loop.setGeoDispatcher(geoDisp)
    loop.setPhysDispatcher(phyDisp)
    loop.setLawDispatcher(lawDisp)
    loop.updateScenePtr(scene, dem)

    # Create a contact manually
    from Contact import Contact

    contact = Contact()
    contact.pA = particles[0]
    contact.pB = particles[1]
    contact.stepLastSeen = scene.step
    contact.stepCreated = scene.step

    dem.contacts.add(contact)

    # Run the contact loop
    print("Running contact loop...")
    loop.run()

    # Check results
    print(f"Number of contacts: {len(dem.contacts)}")
    print(f"Number of real contacts: {dem.contacts.countReal()}")

    # Check if contact was processed correctly
    if dem.contacts.countReal() > 0:
        contact = dem.contacts[0]
        print(f"Contact is real: {contact.isReal()}")
        print(f"Contact has geometry: {contact.geom is not None}")
        print(f"Contact has physics: {contact.phys is not None}")

        if contact.geom and contact.phys:
            print(f"Normal force: {contact.phys.getForce()[0]}")

    return scene, dem, loop


def test_contact_hook():
    """Test ContactHook functionality."""
    print("\n=== Testing ContactHook Functionality ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5

    # Setup functors
    setup_functors()

    # Create particles
    particles = create_particles(dem, 2, 0.9)  # Slightly overlapping
    dem.collectNodes()

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Create contact hook
    hook = TestContactHook()

    # Create contact loop with hook
    loop = ContactLoop()
    loop.setGeoDispatcher(geoDisp)
    loop.setPhysDispatcher(phyDisp)
    loop.setLawDispatcher(lawDisp)
    loop.setContactHook(hook)
    loop.updateScenePtr(scene, dem)

    # Run the contact loop
    print("Running contact loop with hook...")
    loop.run()

    # Check hook results
    print(f"New contacts detected: {len(hook.newContacts)}")

    # Now move particles apart to delete contact
    particles[1].setPos(Vector3r(2.0, 0.0, 0.0))

    # Run the loop again
    scene.step += 1
    loop.run()

    # Check hook results for deleted contacts
    print(f"Deleted contacts detected: {len(hook.delContacts)}")

    return scene, dem, loop, hook


def test_update_phys_modes():
    """Test different UpdatePhys modes."""
    print("\n=== Testing UpdatePhys Modes ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5

    # Setup functors
    setup_functors()

    # Create particles
    particles = create_particles(dem, 2, 0.9)  # Slightly overlapping
    dem.collectNodes()

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Test each update mode
    for mode in [
        UpdatePhys.UPDATE_PHYS_NEVER,
        UpdatePhys.UPDATE_PHYS_ALWAYS,
        UpdatePhys.UPDATE_PHYS_ONCE,
    ]:
        # Create contact loop
        loop = ContactLoop()
        loop.setGeoDispatcher(geoDisp)
        loop.setPhysDispatcher(phyDisp)
        loop.setLawDispatcher(lawDisp)
        loop.setUpdatePhys(mode)
        loop.updateScenePtr(scene, dem)

        # Clear contacts
        dem.contacts.clear()

        # Create a contact manually
        from Contact import Contact

        contact = Contact()
        contact.pA = particles[0]
        contact.pB = particles[1]
        contact.stepLastSeen = scene.step
        contact.stepCreated = scene.step
        dem.contacts.add(contact)

        dem.contacts.dirty = False

        # Run the contact loop
        print(f"\nTesting UpdatePhys mode: {mode}")
        loop.run()

        # Check results
        if dem.contacts.countReal() > 0:
            contact = dem.contacts[0]
            print(f"Contact has physics: {contact.phys is not None}")

            # Run again to test UPDATE_PHYS_ONCE behavior
            if mode == UpdatePhys.UPDATE_PHYS_ONCE:
                # Modify material to see if physics gets updated
                particles[0].getMaterial().setYoung(2e7)

                # Check current mode
                print(f"Current updatePhys mode: {loop.updatePhys}")

                # Run loop again
                loop.run()

                # Should have switched to NEVER
                print(f"New updatePhys mode: {loop.updatePhys}")

    return scene, dem


def test_stress_calculation():
    """Test stress calculation functionality."""
    print("\n=== Testing Stress Calculation ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5
    scene.isPeriodic = True  # Enable periodic boundaries

    # Setup cell
    from Cell import Cell

    scene.cell = Cell()
    scene.cell.setBox(Vector3r(10, 10, 10))

    # Setup functors
    setup_functors()

    # Create particles
    particles = create_particles(dem, 2, 0.9)  # Slightly overlapping
    dem.collectNodes()

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Create contact loop with stress evaluation
    loop = ContactLoop()
    loop.setGeoDispatcher(geoDisp)
    loop.setPhysDispatcher(phyDisp)
    loop.setLawDispatcher(lawDisp)
    loop.setEvalStress(True)
    loop.updateScenePtr(scene, dem)

    # Create a contact manually
    from Contact import Contact

    contact = Contact()
    contact.pA = particles[0]
    contact.pB = particles[1]
    contact.stepLastSeen = scene.step
    contact.stepCreated = scene.step
    dem.contacts.add(contact)

    # Run the contact loop
    print("Running contact loop with stress calculation...")
    loop.run()

    # Check stress results
    print(f"Stress matrix:\n{loop.stress}")

    return scene, dem, loop


def test_performance():
    """Test performance with many particles."""
    print("\n=== Testing Performance with Many Particles ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5

    # Setup functors
    setup_functors()

    # Create many particles in a grid
    particle_count = 10  # 10x10x10 = 1000 particles
    particles = []

    m = FrictMat()
    m.young = 1e7
    m.setYoung(1e7)
    m.setDensity(2000)

    for i in range(particle_count):
        for j in range(particle_count):
            for k in range(particle_count):

                # Create particle
                sp = Sphere()
                sp.setRadius(0.5)
                p = Particle.make(sp, m)

                # Position particles in a grid
                pos = Vector3r(i * 1.1, j * 1.1, k * 1.1)
                p.setPos(pos)

                # Add to DEM field
                dem.particles.add(p)
                particles.append(p)

    # dem.collectNodes()
    print(f"Created {len(particles)} particles")

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Create contact loop
    loop = ContactLoop()
    loop.setGeoDispatcher(geoDisp)
    loop.setPhysDispatcher(phyDisp)
    loop.setLawDispatcher(lawDisp)
    loop.updateScenePtr(scene, dem)

    # Create some contacts manually
    # Just create contacts between adjacent particles

    from Contact import Contact

    for i in range(len(particles)):
        for j in range(i + 1, len(particles)):
            # Only create contacts for particles that are close
            dist = np.linalg.norm(particles[i].getPos() - particles[j].getPos())

            if dist < 2.0:  # Only create contacts for nearby particles
                contact = Contact()
                contact.pA = particles[i]
                contact.pB = particles[j]
                contact.stepLastSeen = scene.step
                contact.stepCreated = scene.step
                dem.contacts.add(contact)

    print(f"Created {len(dem.contacts)} contacts")

    # Run the contact loop and measure time
    print("Running contact loop...")
    start_time = time.time()
    loop.run()
    end_time = time.time()

    # Report results
    elapsed = end_time - start_time
    print(f"Contact loop took {elapsed:.4f} seconds")
    print(f"Number of real contacts: {dem.contacts.countReal()}")

    return scene, dem, loop


def test_reordering():
    """Test contact reordering functionality."""
    print("\n=== Testing Contact Reordering ===")

    # Create scene and field
    scene = Scene()
    dem = DEMField()
    scene.setField(dem)
    scene.dt = 1e-5

    # Setup functors
    setup_functors()

    # Create particles
    particles = create_particles(dem, 5, 1.5)  # Not overlapping
    dem.collectNodes()

    # Create dispatchers
    geoDisp = CGeomDispatcher()
    phyDisp = CPhysDispatcher()
    lawDisp = LawDispatcher()

    # Initialize dispatchers
    geoDisp.updateScenePtr(scene, dem)
    phyDisp.updateScenePtr(scene, dem)
    lawDisp.updateScenePtr(scene, dem)

    # Create contact loop with reordering
    loop = ContactLoop()
    loop.setGeoDispatcher(geoDisp)
    loop.setPhysDispatcher(phyDisp)
    loop.setLawDispatcher(lawDisp)
    loop.setReorderEvery(1)  # Reorder every step
    loop.updateScenePtr(scene, dem)

    # Create a mix of real and non-real contacts
    from Contact import Contact

    # Create real contacts (overlapping)
    for i in range(2):
        contact = Contact()
        contact.pA = particles[i]
        contact.pB = particles[i + 1]
        contact.stepLastSeen = scene.step
        contact.stepCreated = scene.step
        dem.contacts.add(contact)

    # Create non-real contacts (not overlapping)
    for i in range(2, 4):
        contact = Contact()
        contact.pA = particles[i]
        contact.pB = particles[i + 1]
        contact.stepLastSeen = scene.step
        contact.stepCreated = -1  # Not real
        dem.contacts.add(contact)

    # Print initial order
    print("Initial contact order:")
    for i, contact in enumerate(dem.contacts.linView):
        print(f"{i}: {contact.toString()} (real: {contact.isReal()})")

    # Run the contact loop
    print("Running contact loop with reordering...")
    loop.run()

    # Print new order
    print("Contact order after reordering:")
    for i, contact in enumerate(dem.contacts.linView):
        print(f"{i}: {contact.toString()} (real: {contact.isReal()})")

    return scene, dem, loop


def run_all_tests():
    """Run all tests."""
    print("=== Running All ContactLoop Tests ===")

    test_contact_loop_basic()
    test_contact_hook()
    test_update_phys_modes()
    test_stress_calculation()
    test_reordering()

    # Performance test is optional as it creates many particles
    # Uncomment to run it
    test_performance()

    print("\n=== All Tests Completed ===")


if __name__ == "__main__":
    run_all_tests()
