#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any

from pydem.src.Field import Field
from pydem.src.demmath import Vector3r, Real, NAN, AlignedBox3r
from pydem.src.ParticleContainer import ParticleContainer
from pydem.src.ContactContainer import ContactContainer
from pydem.src.DEMData import DEMData
from pydem.src.ClumpData import ClumpData
from pydem.src.DEMFuncs import DemFuncs
from pydem.src.DEMLogging import DEM_LOGGER


@DEM_LOGGER
class DEMField(Field):
    """DEM field for particle-based simulations."""

    # Particle group masks
    MASK_MOVABLE = 0b0001
    MASK_BOUNDARY = 0b0010
    MASK_DELETABLE = 0b0100
    MASK_STATIC = 0b1000

    # Common mask combinations
    MASK_LONE = MASK_BOUNDARY
    MASK_MOVABLE_ALL = MASK_MOVABLE | MASK_DELETABLE
    MASK_BOUNDARY_ALL = MASK_BOUNDARY | MASK_MOVABLE
    MASK_STATIC_ALL = MASK_STATIC | MASK_BOUNDARY | MASK_MOVABLE
    MASK_INLET = MASK_MOVABLE_ALL
    MASK_OUTLET = MASK_DELETABLE

    def __init__(self):
        """Initialize DEM field with default values."""
        super().__init__()
        self.particles = ParticleContainer()
        self.contacts = ContactContainer()
        self.loneMask = self.MASK_LONE
        self.gravity = Vector3r(0.0, 0.0, 0.0)
        self.distFactor = -1.0
        self.saveDead = False
        self.deadNodes = []
        self.deadParticles = []
        self.nodesMutex = threading.RLock()

        # Initialize references
        self.particles.dem = self
        self.contacts.dem = self
        self.contacts.particles = self.particles

        self.debug("DEMField initialized")

    def clearDead(self):
        """Clear dead nodes and particles."""
        self.deadNodes.clear()
        self.deadParticles.clear()
        self.debug("Cleared dead nodes and particles")

    def collectNodes(self, fromCxx=True) -> int:
        """
        Collect nodes from particles.

        Args:
            fromCxx: Whether called from C++ (ignored in Python implementation)

        Returns:
            int: Number of nodes added
        """
        seen = set()
        added = 0

        # First mark existing nodes
        for node in self.nodes:
            seen.add(node)

        # Collect nodes from particles
        for particle in self.particles:
            if particle is None or particle.getShape() is None:
                continue

            for node in particle.getShape().getNodes():
                if node in seen:
                    continue

                if not node.hasDataTyped(DEMData):
                    raise RuntimeError(
                        f"Node {node.toString()} in particle #{particle.getId()} does not have DEMData"
                    )

                seen.add(node)
                node.getDataTyped(DEMData).linIx = len(self.nodes)
                self.nodes.append(node)
                added += 1

        self.debug(f"Collected {added} new nodes from particles")
        return added

    def addNode(self, node):
        """
        Add a node to the field.

        Args:
            node: Node to add

        Raises:
            ValueError: If node is None or invalid
        """
        if node is None:
            raise ValueError("Cannot add null node")

        if not node.hasDataTyped(DEMData):
            raise RuntimeError("Node must have DEMData")

        demData = node.getDataTyped(DEMData)

        if (
            demData.linIx >= 0
            and demData.linIx < len(self.nodes)
            and self.nodes[demData.linIx] is node
        ):
            raise RuntimeError(
                f"Node {node.toString()} already exists at index {demData.linIx}"
            )

        with self.nodesMutex:
            demData.linIx = len(self.nodes)
            self.nodes.append(node)
            self.debug(f"Added node {node.toString()} at index {demData.linIx}")

    def addNodes(self, newNodes):
        """
        Add multiple nodes to the field.

        Args:
            newNodes: List of nodes to add
        """
        for node in newNodes:
            self.addNode(node)

    def addNodesFromParticles(self, particles):
        """
        Add nodes from particles.

        Args:
            particles: List of particles whose nodes should be added
        """
        seen = set()

        for particle in particles:
            if particle.getShape() is None:
                continue

            for node in particle.getShape().getNodes():
                if node in seen:
                    continue
                self.addNode(node)
                seen.add(node)

        self.debug(f"Added nodes from {len(particles)} particles")

    def splitNode(self, node, particles, massMult=None, inertiaMult=None):
        """
        Split a node to create separate nodes for different particles.

        Args:
            node: Node to split
            particles: List of particles to be assigned to the new node
            massMult: Optional mass multiplier
            inertiaMult: Optional inertia multiplier

        Returns:
            List[Node]: List containing original node and the new node
        """
        demData = node.getDataTyped(DEMData)

        if demData.isClump():
            raise RuntimeError("Cannot split clump node")

        if demData.isClumped():
            raise RuntimeError("Cannot split clumped node")

        # Create new node as clone of original
        from Scene import Scene

        clone = Scene.getMaster().deepCopy(node)
        cloneData = clone.getDataTyped(DEMData)
        cloneData.linIx = -1  # Not yet in nodes list

        result = [node, clone]

        # Update particle references
        for particle in particles:
            shape = particle.getShape()
            found = False

            particleRefs = demData.getParticleRefs()
            for i, particleRef in enumerate(particleRefs):
                if particleRef != particle:
                    continue

                # Update particle's node reference
                nodeIndex = -1
                for j, n in enumerate(shape.getNodes()):
                    if n == node:
                        nodeIndex = j
                        break

                if nodeIndex == -1:
                    raise RuntimeError("Node not found in particle's shape nodes")

                shape.getNodes()[nodeIndex] = clone
                particleRefs.pop(i)
                cloneData.addParticleRef(particle)
                found = True
                break

            if not found:
                raise RuntimeError("Particle not found in node's references")

        # Update mass and inertia if specified
        if massMult is not None:
            demData.mass *= massMult
            cloneData.mass *= massMult

        if inertiaMult is not None:
            demData.inertia *= inertiaMult
            cloneData.inertia *= inertiaMult

        self.addNode(clone)
        self.debug(f"Split node {node.toString()} into two nodes")
        return result

    def removeParticle(self, id):
        """
        Remove a particle by ID.

        Args:
            id: Particle ID to remove
        """
        self.debug(f"Removing particle #{id}")

        assert id >= 0
        assert id < len(self.particles)

        particle = self.particles[id]

        if particle is None:
            return

        # Check for clumped nodes
        if particle.getShape() is not None and len(particle.getShape().getNodes()) > 0:
            for node in particle.getShape().getNodes():
                if node.getDataTyped(DEMData).isClumped():
                    raise RuntimeError(
                        f"Cannot remove particle #{id}: contains clumped node"
                    )
        else:
            self.debug(f"Removing #{id} without shape or nodes")
            if self.saveDead:
                self.deadParticles.append(self.particles[id])
            self.particles.remove(id)
            return

        # Remove particle's nodes if no longer used
        for node in particle.getShape().getNodes():
            demData = node.getDataTyped(DEMData)

            if not demData.getParticleRefs():
                raise RuntimeError(f"#{id} has node which back-references no particle!")

            # Remove particle reference from node
            if particle in demData.getParticleRefs():
                demData.getParticleRefs().remove(particle)
            else:
                raise RuntimeError("Node does not reference its particle")

            # Remove node if no longer referenced
            if not demData.getParticleRefs() and demData.linIx >= 0:
                with self.nodesMutex:
                    if self.saveDead:
                        self.deadNodes.append(node)

                    # Move last node to this position
                    if demData.linIx < len(self.nodes) - 1:
                        self.nodes[demData.linIx] = self.nodes[-1]
                        self.nodes[demData.linIx].getDataTyped(
                            DEMData
                        ).linIx = demData.linIx

                    self.nodes.pop()

        # Remove particle's contacts
        for otherId, contact in list(particle.getContacts().items()):
            self.contacts.remove(contact)

        if self.saveDead:
            self.deadParticles.append(particle)

        self.particles.remove(id)

    def removeClump(self, id):
        """
        Remove a clump node and its associated particles.

        Args:
            id: Index of the clump node to remove

        Raises:
            IndexError: If ID is invalid
            RuntimeError: If node is not a clump
        """
        if id >= len(self.nodes):
            raise IndexError("Invalid clump node index")

        node = self.nodes[id]
        if node is None:
            raise RuntimeError(f"Null node at index {id}")

        try:
            clumpData = node.getDataTyped(ClumpData)
        except:
            raise RuntimeError("Node is not a clump")

        # Collect particles to remove
        particlesToRemove = set()
        for clumpedNode in clumpData.getNodes():
            for particle in clumpedNode.getDataTyped(DEMData).getParticleRefs():
                particlesToRemove.add(particle.getId())

        # Remove particles
        for particleId in particlesToRemove:
            if self.saveDead:
                self.deadParticles.append(self.particles[particleId])
            self.removeParticle(particleId)

        # Remove clump node
        if self.saveDead:
            self.deadNodes.append(node)

        with self.nodesMutex:
            if id < len(self.nodes) - 1:
                self.nodes[id] = self.nodes[-1]
                self.nodes[id].getDataTyped(DEMData).linIx = id
            self.nodes.pop()

        self.debug(
            f"Removed clump node {id} and {len(particlesToRemove)} associated particles"
        )

    def setNodesRefPos(self):
        """Set reference positions for visualization."""
        # This would be implemented for visualization systems
        pass

    def getRenderingBBox(self):
        """
        Get bounding box for rendering.

        Returns:
            AlignedBox3r: Bounding box enclosing all nodes
        """
        box = AlignedBox3r()

        for particle in self.particles:
            if particle is None or particle.getShape() is None:
                continue

            for node in particle.getShape().getNodes():
                box.extend(node.pos)

        for node in self.nodes:
            if node is not None:
                box.extend(node.pos)

        return box

    def critDt(self):
        """
        Calculate critical timestep.

        Returns:
            Real: Critical timestep value
        """
        return DemFuncs.pWaveDt(self, True)

    def selfTest(self):
        """
        Run self-tests for the DEM field.

        Raises:
            RuntimeError: If any test fails
        """
        # Test particles
        for i in range(len(self.particles)):
            particle = self.particles[i]
            if particle is None:
                continue

            if particle.getId() != i:
                raise RuntimeError(
                    f"Particle index mismatch: stored={particle.getId()}, actual={i}"
                )

            if particle.getShape() is None:
                raise RuntimeError(f"Particle #{i} has no shape")

            if particle.getMaterial() is None:
                raise RuntimeError(f"Particle #{i} has no material")

            if particle.getMaterial().getDensity() <= 0:
                raise RuntimeError(
                    f"Particle #{i} has invalid density: {particle.getMaterial().getDensity()}"
                )

            # Test shape nodes
            shape = particle.getShape()
            if not shape.checkNumNodes():
                raise RuntimeError(f"Particle #{i} shape has incorrect number of nodes")

            particle.selfTest()

            # Check node references
            for node in shape.getNodes():
                if not node.hasDataTyped(DEMData):
                    raise RuntimeError(f"Particle #{i} node has no DEMData")

                demData = node.getDataTyped(DEMData)

                # Check particle back-reference
                if particle not in demData.getParticleRefs():
                    raise RuntimeError(
                        f"Node does not back-reference its particle #{i}"
                    )

                # Check clump status
                if demData.isClumped() and not demData.getMaster():
                    raise RuntimeError(
                        f"Clumped node has no master node in particle #{i}"
                    )

                if not demData.isClumped() and demData.getMaster():
                    raise RuntimeError(
                        f"Non-clumped node has master node in particle #{i}"
                    )

                # Check node status consistency
                if demData.linIx < 0 and (
                    not np.allclose(demData.vel, Vector3r(0, 0, 0))
                    or not np.allclose(demData.angVel, Vector3r(0, 0, 0))
                    or demData.impose is not None
                ):
                    raise RuntimeError(
                        "Node has velocity/angular velocity/impose but not in nodes list"
                    )

        # Test nodes list consistency
        for i, node in enumerate(self.nodes):
            if node is None:
                raise RuntimeError(f"Null node at index {i}")

            if not node.hasDataTyped(DEMData):
                raise RuntimeError(f"Node at index {i} has no DEMData")

            demData = node.getDataTyped(DEMData)

            if demData.linIx != i:
                raise RuntimeError(
                    f"Node index mismatch at {i}: stored={demData.linIx}"
                )

            # Check clump data
            if demData.isClump():
                try:
                    clumpData = demData
                    # In Python we would use isinstance instead of dynamic_cast
                    if not isinstance(clumpData, ClumpData):
                        raise RuntimeError("Node marked as clump but has no ClumpData")

                    # Test clump nodes
                    for j, clumpedNode in enumerate(clumpData.getNodes()):
                        if clumpedNode is None:
                            raise RuntimeError(f"Null clumped node at index {j}")

                        if not clumpedNode.hasDataTyped(DEMData):
                            raise RuntimeError("Clumped node has no DEMData")

                        clumpedData = clumpedNode.getDataTyped(DEMData)

                        if not clumpedData.isClumped():
                            raise RuntimeError("Node in clump not marked as clumped")

                        if not clumpedData.getMaster():
                            raise RuntimeError("Clumped node has no master reference")

                        if clumpedData.getMaster() is not node:
                            raise RuntimeError("Clumped node references wrong master")
                except:
                    raise RuntimeError("Node marked as clump but has no ClumpData")

            # Check particle references
            for j, particleRef in enumerate(demData.getParticleRefs()):
                if particleRef is None:
                    raise RuntimeError(f"Null particle reference at index {j}")

                if particleRef.getShape() is None:
                    raise RuntimeError("Referenced particle has no shape")

                # Check that particle references this node
                if node not in particleRef.getShape().getNodes():
                    raise RuntimeError("Particle does not reference back its node")

            # Test node physics
            demData.selfTest(node, f"DemField.nodes[{i}]")

        self.debug("Self-test completed successfully")

    # Accessors
    def getParticles(self):
        """Get particle container."""
        return self.particles

    def getContacts(self):
        """Get contact container."""
        return self.contacts

    def getLoneMask(self):
        """Get lone mask."""
        return self.loneMask

    def getGravity(self):
        """Get gravity vector."""
        return self.gravity

    def setGravity(self, g):
        """Set gravity vector."""
        self.gravity = g

    def getDistFactor(self):
        """Get distance factor."""
        return self.distFactor

    def isSaveDead(self):
        """Check if dead objects are saved."""
        return self.saveDead

    def getDeadNodes(self):
        """Get list of dead nodes."""
        return self.deadNodes

    def getDeadParticles(self):
        """Get list of dead particles."""
        return self.deadParticles
