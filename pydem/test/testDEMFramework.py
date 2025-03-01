#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import math
import numpy as np
from demmath import Vector3r, Quaternionr, Real

# Import DEM framework classes
from Object import Object
from Node import Node
from NodeData import NodeData
from DEMData import DEMData
from Bound import Bound
from Shape import Shape, ShapeType
from Material import Material, ElastMat, FrictMat, MaterialType
from Impose import Impose, ImposeVelocity, ImposeForce, CombinedImpose
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
        return (4.0 / 3.0) * math.pi * self.radius**3

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
        mass = (4.0 / 3.0) * math.pi * self.radius**3 * density

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


# Define a Box shape class for testing
class Box(Shape):
    """Box shape implementation."""

    def __init__(self, hx=1.0, hy=1.0, hz=1.0):
        """Initialize box with half-extents."""
        super().__init__()
        self.hx = hx  # Half-extent in x
        self.hy = hy  # Half-extent in y
        self.hz = hz  # Half-extent in z
        self.shapeType = ShapeType.Unknown  # Not a standard shape type

    def getNumNodes(self):
        """Box has a single node."""
        return 1

    def getEquivalentRadius(self):
        """Equivalent radius of the box."""
        return math.sqrt(self.hx**2 + self.hy**2 + self.hz**2)

    def getVolume(self):
        """Volume of the box."""
        return 8.0 * self.hx * self.hy * self.hz

    def isInside(self, point):
        """Check if point is inside the box."""
        if not self.nodes:
            return False

        # Transform point to local coordinates
        local_point = self.nodes[0].glob2loc(point)

        # Check if point is within box half-extents
        return (
            abs(local_point[0]) <= self.hx
            and abs(local_point[1]) <= self.hy
            and abs(local_point[2]) <= self.hz
        )

    def getAlignedBox(self):
        """Get aligned bounding box for the box."""
        if not self.nodes:
            raise RuntimeError("Box has no nodes")

        # For a rotated box, we need to find the corners
        R = self.nodes[0].ori.toRotationMatrix()
        center = self.nodes[0].pos

        # Corners in local coordinates
        corners_local = [
            Vector3r(sx * self.hx, sy * self.hy, sz * self.hz)
            for sx in [-1, 1]
            for sy in [-1, 1]
            for sz in [-1, 1]
        ]

        # Corners in global coordinates
        corners_global = [center + R @ corner for corner in corners_local]

        # Find min and max
        min_corner = Vector3r(
            min(c[0] for c in corners_global),
            min(c[1] for c in corners_global),
            min(c[2] for c in corners_global),
        )
        max_corner = Vector3r(
            max(c[0] for c in corners_global),
            max(c[1] for c in corners_global),
            max(c[2] for c in corners_global),
        )

        self.bound.min = min_corner
        self.bound.max = max_corner
        return self.bound.box

    def applyScale(self, scale):
        """Scale the box dimensions."""
        self.hx *= scale
        self.hy *= scale
        self.hz *= scale

    def lumpMassInertia(self, node, density, mass, inertia, canRotate):
        """Calculate mass and inertia tensor for a box."""
        mass = 8.0 * self.hx * self.hy * self.hz * density

        # Inertia tensor for a solid box
        Ixx = (1.0 / 3.0) * mass * (self.hy**2 + self.hz**2)
        Iyy = (1.0 / 3.0) * mass * (self.hx**2 + self.hz**2)
        Izz = (1.0 / 3.0) * mass * (self.hx**2 + self.hy**2)

        # Set diagonal elements of inertia tensor
        inertia[0, 0] = Ixx
        inertia[1, 1] = Iyy
        inertia[2, 2] = Izz

        # Box can rotate
        canRotate = True

        return mass, inertia, canRotate

    def toString(self):
        """String representation of the box."""
        return f"Box(hx={self.hx}, hy={self.hy}, hz={self.hz})"


def run_dem_test():
    """Run tests for the DEM framework."""
    print("Starting DEM framework test")

    # Test 1: Create basic objects
    print("\n--- Test 1: Create basic objects ---")
    test_object()

    # Test 2: Test Node and NodeData
    print("\n--- Test 2: Test Node and NodeData ---")
    test_node_and_data()

    # Test 3: Test Shape classes
    print("\n--- Test 3: Test Shape classes ---")
    test_shapes()

    # Test 4: Test Material classes
    print("\n--- Test 4: Test Material classes ---")
    test_materials()

    # Test 5: Test Impose classes
    print("\n--- Test 5: Test Impose classes ---")
    test_impose()

    # Test 6: Test Particle creation and properties
    print("\n--- Test 6: Test Particle creation and properties ---")
    test_particle()

    # Test 7: Test complete simulation setup
    print("\n--- Test 7: Test complete simulation setup ---")
    test_simulation_setup()

    print("\nAll tests completed successfully!")


def test_object():
    """Test the base Object class."""
    obj = Object()
    print(f"Created object: {obj.toString()}")
    print(f"Class name: {obj.getClassName()}")

    # Test type checking
    assert obj.isA(Object), "isA() should return True for correct type"
    print("isA() works correctly")

    # Test cast
    try:
        obj.cast(Object)
        print("cast() works correctly")
    except Exception as e:
        print(f"Error in cast(): {e}")


def test_node_and_data():
    """Test Node and NodeData classes."""
    # Create a node
    node = Node()
    print(f"Created node: {node.toString()}")

    # Set position and orientation
    node.pos = Vector3r(1.0, 2.0, 3.0)
    node.ori = Quaternionr(0.707, 0.0, 0.707, 0.0)  # 90 degree rotation around Y
    print(f"Node position: {node.pos}")
    print(f"Node orientation: ({node.ori.w}, {node.ori.x}, {node.ori.y}, {node.ori.z})")

    # Test coordinate transformations
    test_point = Vector3r(4.0, 5.0, 6.0)
    local_point = node.glob2loc(test_point)
    global_point = node.loc2glob(local_point)
    print(f"Global point: {test_point}")
    print(f"Transformed to local: {local_point}")
    print(f"Transformed back to global: {global_point}")

    # Create and attach DEMData
    dem_data = DEMData()
    node.setDataTyped(dem_data)

    # Test data access
    if node.hasDataTyped(DEMData):
        data = node.getDataTyped(DEMData)
        print(f"Node has DEMData: {data.toString()}")
    else:
        print("Error: Node should have DEMData")

    # Test DOF operations
    data.setBlockedDOFs("xZ")
    print(f"Blocked DOFs: {data.getBlockedDOFs()}")
    print(f"Is X blocked? {data.isBlockedAxisDOF(0, False)}")
    print(f"Is Y blocked? {data.isBlockedAxisDOF(1, False)}")
    print(f"Is Z rotation blocked? {data.isBlockedAxisDOF(2, True)}")

    # Test force application
    data.addForce(Vector3r(10.0, 0.0, 0.0))
    data.addForce(Vector3r(0.0, 5.0, 0.0))
    print(f"Applied force: {data.force}")


def test_shapes():
    """Test Shape classes."""
    # Create a sphere
    sphere = Sphere(radius=2.0)
    print(f"Created sphere: {sphere.toString()}")
    print(f"Shape type: {sphere.getShapeType()}")
    print(f"Radius: {sphere.getEquivalentRadius()}")
    print(f"Volume: {sphere.getVolume()}")

    # Create a node for the sphere
    node = Node()
    node.pos = Vector3r(0.0, 0.0, 0.0)
    sphere.nodes.append(node)

    # Test node access
    print(f"Number of nodes: {len(sphere.getNodes())}")
    print(f"Expected number of nodes: {sphere.getNumNodes()}")

    # Test bounding box
    box = sphere.getAlignedBox()
    print(f"Bounding box min: {box.min}")
    print(f"Bounding box max: {box.max}")

    # Test inside check
    inside_point = Vector3r(1.0, 1.0, 1.0)
    outside_point = Vector3r(3.0, 0.0, 0.0)
    print(f"Is point {inside_point} inside? {sphere.isInside(inside_point)}")
    print(f"Is point {outside_point} inside? {sphere.isInside(outside_point)}")

    # Create a box
    box = Box(hx=1.0, hy=2.0, hz=3.0)
    print(f"Created box: {box.toString()}")
    print(f"Equivalent radius: {box.getEquivalentRadius()}")
    print(f"Volume: {box.getVolume()}")

    # Create a node for the box
    node = Node()
    node.pos = Vector3r(0.0, 0.0, 0.0)
    box.nodes.append(node)

    # Test scaling
    box.applyScale(2.0)
    print(f"Box after scaling: {box.toString()}")


def test_materials():
    """Test Material classes."""
    # Create elastic material
    elast_mat = ElastMat(density=2000.0, young=1e9)
    print(f"Created elastic material: {elast_mat.toString()}")
    print(f"Material type: {elast_mat.getMaterialType()}")
    print(f"Density: {elast_mat.getDensity()}")
    print(f"Young's modulus: {elast_mat.getYoung()}")

    # Create frictional material
    frict_mat = FrictMat(density=2500.0, young=2e9, tanPhi=0.6, ktDivKn=0.3)
    print(f"Created frictional material: {frict_mat.toString()}")
    print(f"Material type: {frict_mat.getMaterialType()}")
    print(f"Density: {frict_mat.getDensity()}")
    print(f"Young's modulus: {frict_mat.getYoung()}")
    print(f"Tangent of friction angle: {frict_mat.getTanPhi()}")
    print(f"Kt/Kn ratio: {frict_mat.getKtDivKn()}")

    # Test property changes
    frict_mat.setDensity(3000.0)
    frict_mat.setYoung(3e9)
    frict_mat.setTanPhi(0.8)
    print(f"Updated material density: {frict_mat.getDensity()}")
    print(f"Updated Young's modulus: {frict_mat.getYoung()}")
    print(f"Updated friction angle: {frict_mat.getTanPhi()}")


def test_impose():
    """Test Impose classes."""
    # Create velocity impose
    vel_impose = ImposeVelocity(Vector3r(1.0, 0.0, 0.0))
    print(f"Created velocity impose: {vel_impose.toString()}")
    print(f"Velocity: {vel_impose.getVel()}")

    # Create force impose
    force_impose = ImposeForce(Vector3r(0.0, 100.0, 0.0))
    print(f"Created force impose: {force_impose.toString()}")
    print(f"Force: {force_impose.getForce()}")

    # Test combined impose
    combined = vel_impose.combine(force_impose)
    print(f"Created combined impose: {combined.toString()}")
    print(
        f"Number of imposes: {len(combined.getImposes()) if hasattr(combined, 'getImposes') else 'N/A'}"
    )

    # Test type checking
    print(
        f"Velocity impose has VELOCITY type: {vel_impose.hasType(Impose.Type.VELOCITY)}"
    )
    print(f"Force impose has FORCE type: {force_impose.hasType(Impose.Type.FORCE)}")
    print(
        f"Combined impose has VELOCITY type: {combined.hasType(Impose.Type.VELOCITY)}"
    )
    print(f"Combined impose has FORCE type: {combined.hasType(Impose.Type.FORCE)}")


def test_particle():
    """Test Particle class."""
    # Create shape and material
    sphere = Sphere(radius=1.0)
    material = FrictMat(density=2500.0)

    # Create node for shape
    node = Node()
    node.pos = Vector3r(0.0, 0.0, 0.0)
    sphere.nodes.append(node)

    # Create a particle
    particle = Particle()
    particle.id = 1
    particle.setShape(sphere)
    particle.setMaterial(material)

    print(f"Created particle: {particle.toString()}")
    print(f"ID: {particle.getId()}")

    # Test DEMData creation for node
    dem_data = DEMData()
    node.setDataTyped(dem_data)
    dem_data.addParticleRef(particle)

    # Update mass and inertia
    particle.updateMassInertia()
    print(f"Particle mass: {particle.getMass()}")
    print(f"Particle inertia: {particle.getInertia()}")

    # Test position and orientation
    particle.setPos(Vector3r(1.0, 2.0, 3.0))
    print(f"Particle position: {particle.getPos()}")

    # Test velocity
    particle.setVel(Vector3r(0.5, 1.0, 1.5))
    print(f"Particle velocity: {particle.getVel()}")

    # Test kinetic energy
    energy = particle.getKineticEnergy()
    print(f"Particle kinetic energy: {energy}")

    # Test particle factory method
    factory_particle = Particle.make(Sphere(radius=2.0), ElastMat(density=2000.0))
    print(f"Factory-created particle: {factory_particle.toString()}")
    print(f"Factory particle mass: {factory_particle.getMass()}")

    # Test self-test function
    try:
        particle.selfTest()
        print("Particle passed self-test")
    except Exception as e:
        print(f"Self-test failed: {e}")


def test_simulation_setup():
    """Test a complete simulation setup."""
    print("Setting up a simple DEM simulation with multiple particles")

    # Create materials
    steel = FrictMat(density=7800.0, young=2e11, tanPhi=0.3)
    rubber = FrictMat(density=1200.0, young=5e6, tanPhi=0.9, ktDivKn=0.4)

    # Create particles
    particles = []

    # Create a steel sphere
    steel_sphere = Particle.make(Sphere(radius=0.1), steel)
    steel_sphere.id = 1
    steel_sphere.setPos(Vector3r(0.0, 0.0, 0.3))
    steel_sphere.setVel(Vector3r(0.0, 0.0, -1.0))
    particles.append(steel_sphere)

    # Create a rubber box
    rubber_box = Particle.make(Box(hx=0.2, hy=0.2, hz=0.02), rubber, fixed=True)
    rubber_box.id = 2
    rubber_box.setPos(Vector3r(0.0, 0.0, 0.0))
    particles.append(rubber_box)

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

    # Simulate a single time step
    print("\nSimulating one time step...")
    dt = 0.001  # Time step in seconds
    gravity = Vector3r(0.0, 0.0, -9.81)  # Gravity vector

    # Apply gravity and update positions
    for p in particles:
        if not p.getShape().getNodes()[0].getDataTyped(DEMData).isBlockedAll():
            p.getShape().getNodes()[0].getDataTyped(DEMData).addForce(
                gravity * p.getMass()
            )

            # Very simple explicit Euler integration
            vel = p.getVel()
            force = p.getForce()
            mass = p.getMass()

            # Update velocity
            new_vel = vel + (force / mass) * dt
            p.setVel(new_vel)

            # Update position
            pos = p.getPos()
            new_pos = pos + new_vel * dt
            p.setPos(new_pos)

    # Print updated particle information
    print("\nParticles after time step:")
    for p in particles:
        print(f"Particle {p.getId()}:")
        print(f"  Position: {p.getPos()}")
        print(f"  Velocity: {p.getVel()}")


if __name__ == "__main__":
    run_dem_test()
