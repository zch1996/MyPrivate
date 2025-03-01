#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Dict, List, Optional, Set, Any
import weakref

# Add this import at the top of the file
from pydem.src.Object import Object
from pydem.src.demmath import Vector3r, Quaternionr, NAN
from pydem.src.DEMData import DEMData


class Particle(Object):
    """Class representing a physical particle in the simulation."""

    # Class constants
    ID_NONE = -1

    def __init__(self):
        """Initialize a particle with default values."""
        super().__init__()
        self.id = self.ID_NONE
        self.mask = 1  # Default collision mask
        self.shape = None
        self.material = None
        self.matState = None
        self.contacts = {}  # Dictionary of {particle_id: contact_obj}

    @staticmethod
    def make(shape, material, fixed=False):
        """Factory method to create a new particle with shape and material."""
        if shape is None or material is None:
            raise ValueError("Shape and material must not be null")

        particle = Particle()
        particle.shape = shape
        particle.material = material

        nodes = shape.getNodes()
        # Resize nodes list to match expected size
        while len(nodes) < shape.getNumNodes():
            from Node import Node

            nodes.append(Node())
        nodes = nodes[: shape.getNumNodes()]

        # Import DEMData here to avoid circular import issues
        from DEMData import DEMData

        for i in range(len(nodes)):
            if nodes[i] is None:
                from Node import Node

                nodes[i] = Node()

            if not nodes[i].hasDataTyped(DEMData):
                dem_data = DEMData()
                nodes[i].setDataTyped(dem_data)

            if fixed:
                nodes[i].getDataTyped(DEMData).setBlockedAll()

            nodes[i].getDataTyped(DEMData).addParticleRef(particle)

        shape.updateMassInertia(material.getDensity())
        return particle

    def checkNodes(self, checkDemData=True, checkUninodal=False):
        """Verify that nodes exist and have required properties."""
        if self.shape is None:
            raise RuntimeError(f"Particle #{self.id} has no shape")

        nodes = self.shape.getNodes()
        if not nodes:
            raise RuntimeError(f"Particle #{self.id} has no nodes")

        if checkUninodal and len(nodes) != 1:
            raise RuntimeError(
                f"Particle #{self.id} should be uninodal but has {len(nodes)} nodes"
            )

        if checkDemData:
            if not nodes[0].hasDataTyped(DEMData):
                raise RuntimeError(f"Particle #{self.id} node has no DEMData")

    def getNodes(self):
        """Get list of nodes associated with this particle."""
        self.checkNodes(False, False)
        return self.shape.getNodes()

    def findContactWith(self, other):
        """Find contact with another particle."""
        if other is None:
            return None

        if other.id in self.contacts:
            return self.contacts[other.id]

        return None

    def countRealContacts(self):
        """Count number of real (non-virtual) contacts."""
        return sum(1 for c in self.contacts.values() if c.isReal())

    def updateMassInertia(self):
        """Update mass and inertia based on shape and material."""
        if self.shape is None:
            raise RuntimeError("Cannot update mass/inertia: no shape")

        if self.material is None:
            raise RuntimeError("Cannot update mass/inertia: no material")

        self.shape.updateMassInertia(self.material.getDensity())

    def getKineticEnergy(self, trans=True, rot=True, clumped=True):
        """Calculate kinetic energy of the particle."""
        if self.shape is None:
            return 0

        energy = 0
        for node in self.shape.getNodes():
            if node is None or not node.hasDataTyped(DEMData):
                continue

            demData = node.getDataTyped(DEMData)
            if not clumped and demData.isClumped():
                continue

            # Use Scene from a global Omega instance if available
            scene = None
            try:
                from Omega import Omega

                scene = Omega.instance().getScene()
            except:
                pass

            energy += DEMData.getEkAny(node, trans, rot, scene)

        return energy

    def setShape(self, shape):
        """Set particle shape."""
        if shape is None:
            raise ValueError("Cannot set null shape")
        self.shape = shape

    def getShape(self):
        """Get particle shape."""
        return self.shape

    def setMaterial(self, material):
        """Set particle material."""
        if material is None:
            raise ValueError("Cannot set null material")
        self.material = material

    def getMaterial(self):
        """Get particle material."""
        return self.material

    def getContacts(self):
        """Get all contacts with this particle."""
        return self.contacts

    def getId(self):
        """Get particle ID."""
        return self.id

    def getMask(self):
        """Get collision mask."""
        return self.mask

    def setMask(self, mask):
        """Set collision mask."""
        self.mask = mask

    def getPos(self):
        """Get position of the particle."""
        self.checkNodes(False, True)
        return self.shape.getNodes()[0].pos

    def setPos(self, pos):
        """Set position of the particle."""
        self.checkNodes(False, True)
        self.shape.getNodes()[0].pos = pos

    def getOri(self):
        """Get orientation of the particle."""
        self.checkNodes(False, True)
        return self.shape.getNodes()[0].ori

    def setOri(self, ori):
        """Set orientation of the particle."""
        self.checkNodes(False, True)
        self.shape.getNodes()[0].ori = ori

    def getRefPos(self):
        """Get reference position (for visualization)."""
        # This would typically be used with OpenGL visualization
        # Since we don't have that context, return NaN vector
        return Vector3r(NAN, NAN, NAN)

    def setRefPos(self, pos):
        """Set reference position (for visualization)."""
        # This would typically be used with OpenGL visualization
        raise RuntimeError("RefPos only supported with OpenGL enabled")

    def getVel(self):
        """Get velocity of the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).vel

    def setVel(self, vel):
        """Set velocity of the particle."""
        self.checkNodes(True, True)
        self.shape.getNodes()[0].getDataTyped(DEMData).vel = vel

    def getAngVel(self):
        """Get angular velocity of the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).angVel

    def setAngVel(self, angVel):
        """Set angular velocity of the particle."""
        self.checkNodes(True, True)
        self.shape.getNodes()[0].getDataTyped(DEMData).setAngVel(angVel)

    def getImpose(self):
        """Get imposed constraints on the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).impose

    def setImpose(self, impose):
        """Set imposed constraints on the particle."""
        self.checkNodes(True, True)
        self.shape.getNodes()[0].getDataTyped(DEMData).impose = impose

    def getBlocked(self):
        """Get blocked degrees of freedom as string."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).getBlockedDOFs()

    def setBlocked(self, blocked):
        """Set blocked degrees of freedom from string."""
        self.checkNodes(True, True)
        self.shape.getNodes()[0].getDataTyped(DEMData).setBlockedDOFs(blocked)

    def getMass(self):
        """Get mass of the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).mass

    def getInertia(self):
        """Get inertia tensor of the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).inertia

    def getForce(self):
        """Get force applied to the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).force

    def getTorque(self):
        """Get torque applied to the particle."""
        self.checkNodes(True, True)
        return self.shape.getNodes()[0].getDataTyped(DEMData).torque

    def selfTest(self):
        """Perform self-test to validate particle state."""
        # Basic validation
        if self.shape is None:
            raise RuntimeError(f"Particle #{self.id} has no shape")

        if self.material is None:
            raise RuntimeError(f"Particle #{self.id} has no material")

        # Check material properties
        if self.material.getDensity() <= 0:
            raise RuntimeError(
                f"Particle #{self.id} has invalid density: {self.material.getDensity()}"
            )

        # Check shape
        self.shape.selfTest(self)

        # Check nodes
        for node in self.shape.getNodes():
            if not node.hasDataTyped(DEMData):
                raise RuntimeError(f"Node in particle #{self.id} has no DEMData")

            demData = node.getDataTyped(DEMData)

            # Check particle back-reference
            found = False
            for p in demData.getParticleRefs():
                if p is self:
                    found = True
                    break

            if not found:
                raise RuntimeError(
                    f"Node {node.toString()} does not reference back to particle #{self.id}"
                )

        # Check contacts
        for otherId, contact in self.contacts.items():
            if contact is None:
                raise RuntimeError(
                    f"Particle #{self.id} has null contact with #{otherId}"
                )

            pA = contact.getParticleA()
            pB = contact.getParticleB()
            isFirst = pA is self
            isSecond = pB is self

            if not isFirst and not isSecond:
                raise RuntimeError("Contact does not reference this particle")

    def postLoad(self, obj, attr):
        """Post-load processing."""
        if attr is None:
            if self.shape is None:
                return

            for node in self.shape.getNodes():
                if not node.hasDataTyped(DEMData):
                    continue

                node.getDataTyped(DEMData).addParticleRef(self)

    def toString(self):
        """Return string representation of the particle."""
        result = f"Particle #{self.id}"
        if self.shape:
            result += f" [{self.shape.getClassName()}]"
        else:
            result += " [no-shape]"
        result += f" @ {id(self)}"
        return result
