#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive test for the DEM system components.
This script tests the interaction between all major components:
- Scene
- DEMField
- ParticleContainer
- ContactContainer
- ClumpData
- Engine
- Volumetric
"""

import os
import sys
import time
import logging
import numpy as np
import weakref
from typing import List, Dict, Set
from demmath import PI

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DEMTest")

# Initialize OpenMP simulation
from OpenMPSimulator import OpenMPSimulator

OpenMPSimulator.initialize()

# Import all needed classes
from Object import Object
from Node import Node, NodeData
from DEMData import DEMData
from ClumpData import ClumpData
from Volumetric import Volumetric
from ContactData import ContactData
from ContactGeom import ContactGeom
from ContactPhys import ContactPhys
from Contact import Contact
from ContactContainer import ContactContainer
from ParticleContainer import ParticleContainer
from DEMField import DEMField
from Engine import Engine
from Scene import Scene, EnergyTracker
from demmath import Vector3r, Quaternionr, Matrix3r, NAN, INF
from Material import Material, ElastMat, FrictMat
from Shape import Shape, ShapeType
from Particle import Particle


# Define a sphere shape class for testing
class Sphere(Shape):
    """Sphere shape implementation."""

    def __init__(self, radius=1.0):
        """Initialize sphere with given radius."""
        super().__init__()
        self.radius = radius
        self.shapeType = ShapeType.Sphere

    def getNumNodes(self):
        """Sphere has a single node."""
        return 1

    def getEquivalentRadius(self):
        """Radius of the sphere."""
        return self.radius

    def getVolume(self):
        """Volume of the sphere."""
        return (4.0 / 3.0) * PI * self.radius**3

    def isInside(self, point):
        """Check if point is inside the sphere."""
        if not self.nodes:
            return False

        nodePos = self.nodes[0].pos
        return np.linalg.norm(point - nodePos) <= self.radius

    def getAlignedBox(self):
        """Get aligned bounding box for the sphere."""
        if not self.nodes:
            raise RuntimeError("Sphere has no nodes")

        center = self.nodes[0].pos
        self.bound.min = Vector3r(
            center[0] - self.radius, center[1] - self.radius, center[2] - self.radius
        )
        self.bound.max = Vector3r(
            center[0] + self.radius, center[1] + self.radius, center[2] + self.radius
        )
        return self.bound.box

    def applyScale(self, scale):
        """Scale the sphere radius."""
        self.radius *= scale

    def lumpMassInertia(self, node, density, mass, inertia, canRotate):
        """Calculate mass and inertia tensor for a sphere."""
        mass = (4.0 / 3.0) * PI * self.radius**3 * density

        # Inertia tensor for a solid sphere
        I = (2.0 / 5.0) * mass * self.radius**2

        # Set diagonal elements of inertia tensor
        inertia[0, 0] = I
        inertia[1, 1] = I
        inertia[2, 2] = I

        # Sphere can rotate
        canRotate = True

        return mass, inertia, canRotate

    def toString(self):
        """String representation of the sphere."""
        return f"Sphere(r={self.radius})"


# Create a class to run clump formation tests
class ClumpTest:
    """Tests for clump formation and manipulation."""

    @staticmethod
    def create_single_node_clump():
        """Test creating a single-node clump."""
        # Create a node with DEMData
        node = Node()
        demData = DEMData()
        node.setDataTyped(demData)

        # Setup node properties
        node.pos = Vector3r(1.0, 2.0, 3.0)
        node.ori = Quaternionr(1.0, 0.0, 0.0, 0.0)
        demData.mass = 10.0
        demData.inertia = Vector3r(2.0, 2.0, 2.0)

        # Create a particle to reference the node
        particle = Particle.make(Sphere(0.5), Material(1000.0))
        demData.addParticleRef(particle)

        # Create a clump with the node
        clumpNode = ClumpData.makeClump([node])

        # Verify clump properties
        assert node.getDataTyped(
            DEMData
        ).isClumped(), "Node should be marked as clumped"
        assert clumpNode.getDataTyped(
            DEMData
        ).isClump(), "Clump node should be marked as clump"
        assert np.allclose(
            clumpNode.pos, node.pos
        ), "Clump should have same position as node"

        logger.info("Single-node clump test passed")
        return clumpNode, node

    @staticmethod
    def create_multi_node_clump():
        """Test creating a multi-node clump."""
        # Create nodes with DEMData
        nodes = []
        for i in range(3):
            node = Node()
            demData = DEMData()
            node.setDataTyped(demData)

            # Setup node properties
            node.pos = Vector3r(i * 1.0, i * 0.5, i * 0.2)
            node.ori = Quaternionr(1.0, 0.0, 0.0, 0.0)
            demData.mass = 5.0
            demData.inertia = Vector3r(1.0, 1.0, 1.0)

            # Create a particle to reference the node
            particle = Particle.make(Sphere(0.4), Material(1000.0))
            demData.addParticleRef(particle)

            nodes.append(node)

        # Create a clump with the nodes
        clumpNode = ClumpData.makeClump(nodes)

        # Verify clump properties
        for node in nodes:
            assert node.getDataTyped(
                DEMData
            ).isClumped(), "Node should be marked as clumped"
            assert (
                node.getDataTyped(DEMData).getMaster() is clumpNode
            ), "Node should reference clump node as master"

        assert clumpNode.getDataTyped(
            DEMData
        ).isClump(), "Clump node should be marked as clump"
        assert isinstance(
            clumpNode.getDataTyped(DEMData), ClumpData
        ), "Clump node should have ClumpData"

        clumpData = clumpNode.getDataTyped(ClumpData)
        assert len(clumpData.getNodes()) == 3, "Clump should have 3 nodes"

        logger.info("Multi-node clump test passed")
        return clumpNode, nodes


# Create a test for contact detection and management
class ContactTest:
    """Tests for contact creation and management."""

    @staticmethod
    def create_contacts():
        """Create test contacts between particles."""
        # Create particles
        p1 = Particle.make(Sphere(0.5), Material(1000.0))
        p1.id = 0
        p1.getShape().getNodes()[0].pos = Vector3r(0.0, 0.0, 0.0)

        p2 = Particle.make(Sphere(0.5), Material(1000.0))
        p2.id = 1
        p2.getShape().getNodes()[0].pos = Vector3r(0.9, 0.0, 0.0)

        p3 = Particle.make(Sphere(0.5), Material(1000.0))
        p3.id = 2
        p3.getShape().getNodes()[0].pos = Vector3r(0.0, 0.9, 0.0)

        # Create contacts
        c12 = Contact()
        c12.pA = weakref.ref(p1)
        c12.pB = weakref.ref(p2)

        c13 = Contact()
        c13.pA = weakref.ref(p1)
        c13.pB = weakref.ref(p3)

        # Add geometry and physics to make contacts "real"
        c12.geom = ContactGeom()
        c12.phys = ContactPhys()
        c12.geom.node = Node()
        c12.geom.node.pos = Vector3r(0.45, 0.0, 0.0)
        c12.phys.force = Vector3r(10.0, 0.0, 0.0)

        c13.geom = ContactGeom()
        c13.phys = ContactPhys()
        c13.geom.node = Node()
        c13.geom.node.pos = Vector3r(0.0, 0.45, 0.0)
        c13.phys.force = Vector3r(0.0, 10.0, 0.0)

        logger.info("Created test contacts")
        return [p1, p2, p3], [c12, c13]

    @staticmethod
    def test_contact_container(particles, contacts):
        """Test contact container functionality."""
        # Create container with mock particle container
        container = ContactContainer()
        particleContainer = ParticleContainer()

        # Add particles to particle container
        for p in particles:
            particleContainer.insertAt(p, p.id)

        container.particles = particleContainer

        # Add contacts
        for c in contacts:
            assert container.add(c), f"Failed to add {c.toString()}"

        # Test container state
        assert container.size() == len(
            contacts
        ), f"Container size should be {len(contacts)}"
        assert container.countReal() == len(contacts), "All contacts should be real"

        # Test finding contacts
        p1, p2, p3 = particles
        assert container.exists(p1.id, p2.id), "Contact between p1 and p2 should exist"
        assert container.exists(p1.id, p3.id), "Contact between p1 and p3 should exist"
        assert not container.exists(
            p2.id, p3.id
        ), "Contact between p2 and p3 should not exist"

        # Test iterations
        count = 0
        for c in container:
            count += 1
            assert c.isReal(), "Iterated contact should be real"
        assert count == len(contacts), f"Iterator should yield {len(contacts)} contacts"

        logger.info("Contact container tests passed")
        return container


# Create a test for the DEMField
class DEMFieldTest:
    """Tests for DEMField functionality."""

    @staticmethod
    def setup_field(particles, contacts):
        """Set up a DEM field with particles and contacts."""
        # Create field
        field = DEMField()

        # Add particles to field
        for p in particles:
            field.particles.insertAt(p, p.id)

        # Add contacts to field
        for c in contacts:
            field.contacts.add(c)

        # Collect nodes
        added = field.collectNodes()
        logger.info(f"Added {added} nodes to field")

        return field

    @staticmethod
    def test_field_operations(field):
        """Test field operations."""
        # Test gravity
        gravity = Vector3r(0.0, 0.0, -9.81)
        field.setGravity(gravity)
        assert np.allclose(
            field.getGravity(), gravity
        ), "Gravity should be set correctly"

        # Test particle removal
        particleCount = len(field.particles)
        print("particleCount", particleCount)
        field.removeParticle(2)  # Remove particle with ID 2
        assert (
            len(field.particles) == particleCount - 1
        ), "One particle should be removed"
        assert not field.particles.exists(2), "Particle 3 should not exist"

        # Test self test
        try:
            field.selfTest()
            logger.info("Field self-test passed")
        except Exception as e:
            logger.error(f"Field self-test failed: {str(e)}")
            raise

        return field


# Create a test for Scene and engines
class SceneTest:
    """Tests for Scene and engines."""

    @staticmethod
    def setup_scene(field):
        """Set up a scene with the given field."""
        # Create scene
        scene = Scene()
        scene.setField(field)

        # Add test engine
        engine = TestEngine()
        scene.addEngine(engine)

        # Enable energy tracking
        scene.enableEnergyTracking(True)

        return scene

    @staticmethod
    def run_simulation(scene, steps=10):
        """Run a simulation for the given number of steps."""
        # Run simulation
        startTime = scene.time
        startStep = scene.step

        logger.info(f"Starting simulation at step {startStep}, time {startTime}")
        scene.run(steps=steps, wait=True)

        endTime = scene.time
        endStep = scene.step

        logger.info(f"Simulation completed at step {endStep}, time {endTime}")
        logger.info(
            f"Ran for {endStep - startStep} steps, {endTime - startTime} seconds"
        )

        return scene


# Create a test for Volumetric
class VolumetricTest:
    """Tests for Volumetric utilities."""

    @staticmethod
    def test_volumetric_calculations():
        """Test volumetric calculations."""
        # Test tetrahedron volume
        a = Vector3r(0.0, 0.0, 0.0)
        b = Vector3r(1.0, 0.0, 0.0)
        c = Vector3r(0.0, 1.0, 0.0)
        d = Vector3r(0.0, 0.0, 1.0)

        volume = Volumetric.tetraVolume(a, b, c, d)
        expected = 1.0 / 6.0
        assert (
            abs(volume - expected) < 1e-6
        ), f"Tetrahedron volume should be {expected}, got {volume}"

        # Test inertia calculations
        inertia = Volumetric.tetraInertia(a, b, c, d)
        assert inertia.shape == (3, 3), "Inertia tensor should be 3x3"

        # Test triangle area
        area = Volumetric.triangleArea(a, b, c)
        expected = 0.5
        assert (
            abs(area - expected) < 1e-6
        ), f"Triangle area should be {expected}, got {area}"

        # Test principal axes computation
        mass = 10.0
        staticMoment = Vector3r(10.0, 20.0, 30.0)
        inertiaTensor = np.eye(3) * 100.0

        pos, ori, principalInertia = Volumetric.computePrincipalAxes(
            mass, staticMoment, inertiaTensor
        )
        assert np.allclose(
            pos, staticMoment / mass
        ), "Position should be center of mass"

        logger.info("Volumetric tests passed")


# Main test function
def run_dem_system_test():
    """Run a comprehensive test of the DEM system."""
    try:
        logger.info("Starting DEM system test")

        # Test clump formation
        logger.info("Testing clump formation...")
        single_clump, single_node = ClumpTest.create_single_node_clump()
        multi_clump, multi_nodes = ClumpTest.create_multi_node_clump()

        # Test contact creation and management
        logger.info("Testing contact system...")
        particles, contacts = ContactTest.create_contacts()
        container = ContactTest.test_contact_container(particles, contacts)

        # Test DEMField
        logger.info("Testing DEMField...")
        field = DEMFieldTest.setup_field(particles, contacts)
        field = DEMFieldTest.test_field_operations(field)

        # Test Volumetric
        logger.info("Testing Volumetric utilities...")
        VolumetricTest.test_volumetric_calculations()

        # All tests passed
        logger.info("All DEM system tests passed")
        return True

    except Exception as e:
        logger.error(f"DEM system test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run the test
    success = run_dem_system_test()
    sys.exit(0 if success else 1)
