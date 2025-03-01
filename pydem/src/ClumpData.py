#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Optional, Dict, Set, Tuple

from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr, NAN
from pydem.src.DEMData import DEMData
from pydem.src.Node import Node, NodeData
from pydem.src.DEMLogging import DEM_LOGGER, DEM_IMPL_LOGGER
from pydem.src.Volumetric import Volumetric


@DEM_LOGGER
class ClumpData(DEMData):
    """Class for managing clumped particles in DEM simulations."""

    def __init__(self):
        """Initialize ClumpData with default values."""
        super().__init__()
        self.nodes = []  # List of Node objects
        self.relPos = []  # Relative positions of members
        self.relOri = []  # Relative orientations of members
        self.equivRad = NAN  # Equivalent radius for statistics

    @staticmethod
    def makeClump(
        nodes: List[Node],
        centralNode: Optional[Node] = None,
        intersecting: bool = False,
    ) -> Node:
        """
        Create a clump from a list of nodes.

        Args:
            nodes: List of nodes to include in the clump
            centralNode: Optional central node to use as the clump center
            intersecting: Whether to allow self-intersecting clumps

        Returns:
            Node: The clump master node
        """
        if not nodes:
            raise RuntimeError("Cannot create clump with 0 nodes")

        if intersecting:
            raise RuntimeError("Self-intersecting clumps not yet implemented")

        # Create or validate central node
        clump = None
        cNode = centralNode if centralNode else Node()

        if not centralNode or not centralNode.hasDataTyped(DEMData):
            clump = ClumpData()
            clump.setClump()
            # Register the ClumpData as both CLUMPDATA and DEMDATA
            cNode.setData(clump, Node.DataType.CLUMP)  # Set as ClumpData
            cNode.setData(clump, Node.DataType.DEM)  # Also set as DEMData
        else:
            if not isinstance(centralNode.getDataTyped(DEMData), ClumpData):
                raise RuntimeError("Central node must have ClumpData attached")
            clump = centralNode.getDataTyped(DEMData)
            clump.setClump()

        # Special case: single node clump
        if len(nodes) == 1:
            clump.debug("Creating single-node clump")
            cNode.pos = nodes[0].pos.copy()
            cNode.ori = nodes[0].ori.copy()
            clump.nodes.append(nodes[0])

            memberData = nodes[0].getDataTyped(DEMData)
            memberData.setClumped(cNode)

            clump.relPos.append(Vector3r(0.0, 0.0, 0.0))
            clump.relOri.append(Quaternionr(1.0, 0.0, 0.0, 0.0))  # Identity quaternion
            clump.mass = memberData.mass
            clump.inertia = memberData.inertia.copy()
            clump.equivRad = np.sqrt(np.sum(clump.inertia) / (3.0 * clump.mass))
            return cNode

        # Multi-node clump
        totalMass = 0.0
        staticMoment = Vector3r(0.0, 0.0, 0.0)
        inertiaTensor = np.zeros((3, 3))

        # First pass: compute mass properties
        for node in nodes:
            memberData = node.getDataTyped(DEMData)

            if memberData.isClumped():
                raise RuntimeError(f"Node {node.toString()} is already clumped")

            if not memberData.getParticleRefs():
                raise RuntimeError(f"Node {node.toString()} has no particle references")

            totalMass += memberData.mass
            staticMoment += memberData.mass * node.pos

            # Transform inertia tensor
            inertiaTensor += Volumetric.inertiaTensorTranslate(
                Volumetric.inertiaTensorRotate(
                    np.diag(memberData.inertia), node.ori.conjugate()
                ),
                memberData.mass,
                -1.0 * node.pos,
            )

        # Compute principal properties
        if totalMass > 0:
            cNode.pos, cNode.ori, clump.inertia = Volumetric.computePrincipalAxes(
                totalMass, staticMoment, inertiaTensor
            )
            clump.mass = totalMass
            clump.equivRad = np.sqrt(np.sum(clump.inertia) / (3.0 * clump.mass))
        else:
            if not centralNode:
                raise RuntimeError("Massless clump requires central node specification")
            clump.equivRad = NAN

        # Block massless clumps
        if not centralNode and not (clump.mass > 0):
            clump.setBlockedAll()

        # Second pass: compute relative positions and orientations
        for node in nodes:
            memberData = node.getDataTyped(DEMData)

            clump.nodes.append(node)
            clump.relPos.append(cNode.ori.conjugate() * (node.pos - cNode.pos))
            clump.relOri.append(cNode.ori.conjugate() * node.ori)

            memberData.setClumped(cNode)

        return cNode

    @staticmethod
    def checkIsClumpNode(node: Node) -> None:
        """
        Check if a node is a clump master node.

        Args:
            node: Node to check

        Raises:
            RuntimeError: If node is not a valid clump master node
        """
        if not node.hasDataTyped(DEMData):
            raise RuntimeError("Node has no DEMData")

        if not isinstance(node.getDataTyped(DEMData), ClumpData):
            raise RuntimeError("Node is not a clump master node")

        if not node.getDataTyped(DEMData).isClump():
            raise RuntimeError(
                "Invalid state: node has ClumpData but is not marked as clump"
            )

    @staticmethod
    def forceTorqueFromMembers(node: Node, force: Vector3r, torque: Vector3r) -> None:
        """
        Compute force and torque from member nodes.

        Args:
            node: Clump master node
            force: Force vector to update
            torque: Torque vector to update
        """
        clump = node.getDataTyped(ClumpData)
        if not isinstance(clump, ClumpData):
            raise TypeError("Node doesn't have ClumpData")

        for memberNode in clump.nodes:
            memberData = memberNode.getDataTyped(DEMData)
            force += memberData.force
            torque += memberData.torque + np.cross(
                memberNode.pos - node.pos, memberData.force
            )

    @staticmethod
    def applyToMembers(node: Node, reset: bool = False) -> None:
        """
        Apply clump properties to member nodes.

        Args:
            node: Clump master node
            reset: Whether to reset force and torque on members
        """
        clump = node.getDataTyped(DEMData)
        if not isinstance(clump, ClumpData):
            raise TypeError("Node doesn't have ClumpData")

        for i, memberNode in enumerate(clump.nodes):
            memberData = memberNode.getDataTyped(DEMData)

            if not memberData.isClumped():
                raise RuntimeError(
                    f"Node {memberNode.toString()} should be marked as clumped"
                )

            # Update position and orientation
            memberNode.pos = node.pos + node.ori * clump.relPos[i]
            memberNode.ori = node.ori * clump.relOri[i]

            # Update velocities
            memberData.vel = clump.vel + np.cross(
                clump.angVel, memberNode.pos - node.pos
            )
            memberData.angVel = clump.angVel

            # Reset forces if requested
            if reset:
                memberData.force = Vector3r(0.0, 0.0, 0.0)
                memberData.torque = Vector3r(0.0, 0.0, 0.0)

    @staticmethod
    def resetForceTorque(node: Node) -> None:
        """
        Reset force and torque on all member nodes.

        Args:
            node: Clump master node
        """
        clump = node.getDataTyped(DEMData)
        if not isinstance(clump, ClumpData):
            raise TypeError("Node doesn't have ClumpData")

        for memberNode in clump.nodes:
            memberData = memberNode.getDataTyped(DEMData)
            memberData.force = Vector3r(0.0, 0.0, 0.0)
            memberData.torque = Vector3r(0.0, 0.0, 0.0)

    def getNodes(self) -> List[Node]:
        """Get list of member nodes."""
        return self.nodes

    def getRelativePositions(self) -> List[Vector3r]:
        """Get relative positions of member nodes."""
        return self.relPos

    def getRelativeOrientations(self) -> List[Quaternionr]:
        """Get relative orientations of member nodes."""
        return self.relOri

    def getEquivalentRadius(self) -> float:
        """Get equivalent radius of the clump."""
        return self.equivRad


# Register ClumpData with NodeData.DataIndex
NodeData.DataIndex.CLUMPDATA = 1  # Use a unique index different from DEMDATA
