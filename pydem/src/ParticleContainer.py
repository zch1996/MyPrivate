#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
from typing import List, Optional, Dict, Any
from pydem.src.Object import Object


class ParticleContainer(Object):
    """Container for managing particles with efficient ID allocation."""

    def __init__(self):
        """Initialize with empty particle storage."""
        super().__init__()
        self.parts = []  # List of particles
        self.freeIds = []  # List of free IDs
        self.manipMutex = threading.RLock()  # Mutex for thread safety
        self.dem = None  # Reference to DEM field (set by DemField.postLoad)

    def clear(self):
        """Clear all particles and free IDs."""
        with self.manipMutex:
            self.parts.clear()
            self.freeIds.clear()

    def size(self):
        """Get number of particles (including None entries)."""
        return len(self.parts)

    def exists(self, id):
        """Check if particle with given ID exists."""
        return (id >= 0) and (id < len(self.parts)) and (self.parts[id] is not None)

    def safeGet(self, id):
        """Get particle by ID, with error checking."""
        if not self.exists(id):
            raise ValueError(f"No such particle: #{id}")
        return self.parts[id]

    def findFreeId(self):
        """Find next available ID."""
        size = len(self.parts)

        # Try to reuse IDs from the free list
        while self.freeIds:
            id = self.freeIds[0]
            self.freeIds.pop(0)

            if id < 0:
                raise RuntimeError("Invalid negative ID in freeIds")

            if id <= size:
                if self.parts[id] is not None:
                    raise RuntimeError(
                        f"ParticleContainer.findFreeId: freeIds contained {id}, but it is occupied"
                    )
                return id

        # If no free IDs, use the next available
        return size

    def insertAt(self, particle, id):
        """Insert particle at specified ID."""
        if id < 0:
            raise ValueError("Cannot insert at negative index")

        with self.manipMutex:
            # Resize parts list if needed
            if id >= len(self.parts):
                self.parts.extend([None] * (id + 1 - len(self.parts)))

            # Set particle ID and store it
            if particle is not None:
                particle.id = id
            self.parts[id] = particle

    def add(self, particle, nodes=-1):
        """Add particle and optionally its nodes to the field."""
        if particle is None:
            print("Particle to be added is None.")

        if particle.id >= 0:
            print(f"Particle already has ID {particle.id}")

        if nodes != -1 and nodes != 0 and nodes != 1:
            print(f"nodes must be ∈ {{-1,0,1}} (not {nodes}).")

        if nodes != 0:
            if particle.getShape() is None:
                print("Particle has no shape.")

            if not particle.getShape().checkNumNodes():
                print("Particle shape has wrong number of nodes.")

            for node in particle.getShape().getNodes():
                demData = node.getDataTyped(DEMData)
                if demData.linIx >= 0:
                    # Node already in DemField.nodes, don't add again
                    if (
                        demData.linIx < len(self.dem.nodes)
                        and self.dem.nodes[demData.linIx] is node
                    ):
                        continue

                    # Node not in DemField.nodes, or says it is somewhere else
                    if demData.linIx < len(self.dem.nodes):
                        raise RuntimeError(
                            f"{node.toString()}: Node.dem.linIx={demData.linIx}, "
                            f"but DemField.nodes[{demData.linIx}]="
                            f"{self.dem.nodes[demData.linIx].toString() if self.dem.nodes[demData.linIx] else 'is empty (programming error!?)'}"
                        )
                    raise RuntimeError(
                        f"{node.toString()}: Node.dem.linIx={demData.linIx}, "
                        f"which is out of range for DemField.nodes (size {len(self.dem.nodes)})"
                    )
                else:
                    # Maybe add node
                    if nodes == 1 or (nodes == -1 and demData.guessMoving()):
                        demData.linIx = len(self.dem.nodes)
                        self.dem.nodes.append(node)

        return self.insert(particle)

    def insert(self, particle):
        """Insert particle using next available ID."""
        if particle is None:
            raise ValueError("Cannot insert null particle")

        if particle.id >= 0:
            raise ValueError(f"Particle already has ID {particle.id}")

        id = self.findFreeId()
        self.insertAt(particle, id)
        return id

    def remove(self, id):
        """Remove particle with specified ID."""
        if not self.exists(id):
            return False

        with self.manipMutex:
            # Add to free IDs list
            self.freeIds.append(id)

            # Reset particle pointer
            self.parts[id] = None

            # Shrink container if possible
            if id + 1 == len(self.parts):
                while id >= 0 and self.parts[id] is None:
                    id -= 1
                self.parts = self.parts[: id + 1]

            return True

    def __getitem__(self, id):
        """Access particle by ID."""
        return self.parts[id]

    def __setitem__(self, id, particle):
        """Set particle at ID."""
        self.insertAt(particle, id)

    def __iter__(self):
        """Iterate over non-None particles."""
        return (p for p in self.parts if p is not None)

    def __len__(self):
        """Count non-None particles."""
        return sum(1 for p in self.parts if p is not None)


# Import at the end to avoid circular imports
from pydem.src.DEMData import DEMData
