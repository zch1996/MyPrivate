#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for Contact and ContactContainer classes.
"""

import weakref
import logging
import numpy as np

# Configure logging to see debug messages
logging.basicConfig(level=logging.DEBUG)

# Initialize the OpenMP simulation
from OpenMPSimulator import OpenMPSimulator

OpenMPSimulator.initialize()


# Create mock classes for testing
class MockNode:
    def __init__(self):
        self.pos = np.array([0.0, 0.0, 0.0])
        self.ori = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion


class MockShape:
    def __init__(self):
        self.node = MockNode()

    def getNodes(self):
        return [self.node]


class MockParticle:
    def __init__(self, id):
        self.id = id
        self.contacts = {}
        self.shape = MockShape()

    def getId(self):
        return self.id

    def getShape(self):
        return self.shape

    def getNodes(self):
        return self.shape.getNodes()

    def checkNodes(self, *args):
        pass

    def toString(self):
        return f"Particle #{self.id}"


class MockParticleContainer(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def exists(self, id):
        return 0 <= id < len(self) and self[id] is not None


# Import Contact classes
from Contact import Contact
from ContactContainer import ContactContainer


def test_contact_creation():
    print("Testing Contact creation...")

    # Create particles
    p1 = MockParticle(1)
    p2 = MockParticle(2)

    # Create contact
    c = Contact()
    c.pA = weakref.ref(p1)
    c.pB = weakref.ref(p2)

    # Check basic properties
    assert c.isReal() == False, "New contact should not be real"
    assert c.isColliding() == False, "New contact should not be colliding"
    assert c.getParticleA() == p1, "ParticleA should be p1"
    assert c.getParticleB() == p2, "ParticleB should be p2"

    print("Contact creation test passed!")


def test_contact_container():
    print("Testing ContactContainer...")

    # Create particles
    particles = MockParticleContainer(
        [MockParticle(0), MockParticle(1), MockParticle(2)]  # Index 0 is empty
    )

    # Create contacts
    c12 = Contact()
    c12.pA = weakref.ref(particles[1])
    c12.pB = weakref.ref(particles[2])

    c13 = Contact()
    c13.pA = weakref.ref(particles[1])
    c13.pB = weakref.ref(particles[0])

    # Create container
    container = ContactContainer()
    container.particles = particles

    # Add contacts
    assert container.add(c12) == True, "Should successfully add c12"
    assert container.add(c13) == True, "Should successfully add c13"

    # Check container state
    assert len(container) == 2, "Container should have 2 contacts"
    assert container.exists(1, 2) == True, "Contact (1,2) should exist"
    assert container.exists(1, 0) == True, "Contact (1,3) should exist"
    assert container.exists(2, 0) == False, "Contact (2,3) should not exist"

    # Test removal
    assert container.remove(c12) == True, "Should successfully remove c12"
    assert len(container) == 1, "Container should have 1 contact after removal"
    assert container.exists(1, 2) == False, "Contact (1,2) should no longer exist"

    # Test pending removals
    container.requestRemoval(c13)
    assert container.removeAllPending() == 1, "Should remove 1 pending contact"
    assert len(container) == 0, "Container should be empty after removing all pending"

    print("ContactContainer test passed!")


def run_all_tests():
    print("=== Starting Contact/ContactContainer Tests ===")
    test_contact_creation()
    test_contact_container()
    print("=== All tests passed! ===")


if __name__ == "__main__":
    run_all_tests()
