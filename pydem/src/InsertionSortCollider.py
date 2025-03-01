#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import threading
from typing import List, Dict, Tuple, Optional, Any
import weakref

from pydem.src.Collision import AabbCollider
from pydem.src.Contact import Contact
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import Vector3r, Vector3i, Real, INF, NAN
from pydem.src.OpenMPSimulator import omp_get_thread_num, omp_get_max_threads


@DEM_LOGGER
class InsertionSortCollider(AabbCollider):
    """
    Collider using insertion sort algorithm for collision detection.

    This collider maintains sorted lists of particle bounds along each axis
    and detects collisions by tracking bound inversions.
    """

    class Bounds:
        """Struct for storing bounds of bodies."""

        def __init__(self, coord=0.0, id=0, is_min=False):
            """Initialize bounds with default values."""
            self.coord = coord  # Coordinate along sort axis
            self.id = id  # Particle ID
            self.period = 0  # Periodic cell coordinate

            # Flags
            self.hasBB = False  # Has bounding box
            self.isMin = is_min  # Is minimum bound
            self.isInf = False  # Is infinite

        def __lt__(self, other):
            """Less than comparison for sorting."""
            if self.id == other.id and self.coord == other.coord:
                return self.isMin
            return self.coord < other.coord

        def __gt__(self, other):
            """Greater than comparison for sorting."""
            if self.id == other.id and self.coord == other.coord:
                return not self.isMin
            return self.coord > other.coord

    class VecBounds:
        """Container for bounds along one axis."""

        def __init__(self, axis=-1):
            """Initialize with default values."""
            self.axis = axis  # Axis (0,1,2)
            self.vec = []  # Bounds vector
            self.cellDim = 0.0  # Cell dimension
            self.size = 0  # Cache of vector size
            self.loIdx = 0  # Lowest coordinate index

        def __getitem__(self, idx):
            """Get bounds at index."""
            assert idx < self.size and idx >= 0
            return self.vec[idx]

        def __setitem__(self, idx, value):
            """Set bounds at index."""
            assert idx < self.size and idx >= 0
            self.vec[idx] = value

        def updatePeriodicity(self, scene):
            """Update cell dimension for periodicity."""
            assert scene.isPeriodic
            assert 0 <= self.axis <= 2
            self.cellDim = scene.cell.getSize()[self.axis]

        def clear(self):
            """Clear bounds vector."""
            self.vec.clear()
            self.size = 0
            self.loIdx = 0

        def norm(self, i):
            """Normalize index for periodic boundaries."""
            if i < 0:
                i += self.size
            ret = i % self.size
            assert ret >= 0 and ret < self.size
            return ret

        def __str__(self):
            """String representation."""
            s = f"VecBounds for axis {self.axis} with {self.size} entries\n"
            for i in range(self.size):
                s += f"{self.vec[i].coord}\n"
            return s

    def __init__(self, *args, **kwargs):
        """Initialize InsertionSortCollider with default values."""
        super().__init__()

        # Initialize thread-local storage for contacts
        n_threads = omp_get_max_threads()
        self.mmakeContacts = [[] for _ in range(n_threads)]
        self.rremoveContacts = [[] for _ in range(n_threads)]

        # Initialize bounds arrays
        self.BB = [self.VecBounds(i) for i in range(3)]
        self.maxima = []  # Maximum bounds
        self.minima = []  # Minimum bounds

        # Configuration
        self.forceInitSort = False
        self.sortAxis = 0
        self.sortThenCollide = False
        self.maxVel2 = 0.0
        self.numReinit = 0
        self.stepInvs = Vector3i(0, 0, 0)
        self.numInvs = Vector3i(0, 0, 0)
        self.ompTuneSort = Vector3i(1, 1000, 0)
        self.sortChunks = -1
        self.paraPeri = False
        self.periDbgNew = False
        self.maxSortPass = -100

        # Internal state
        self.particles = None
        self.dem = None
        self.periodic = False

    def removeContactLater(self, contact):
        """Queue contact for removal."""
        self.rremoveContacts[omp_get_thread_num()].append(contact)

    def makeContactLater(self, pA, pB, cellDist=None):
        """Queue contact for creation."""
        if cellDist is None:
            cellDist = Vector3i(0, 0, 0)

        C = Contact()
        C.pA = weakref.ref(pA)
        C.pB = weakref.ref(pB)
        C.cellDist = cellDist
        C.stepCreated = self.scene.step
        self.mmakeContacts[omp_get_thread_num()].append(C)

    def makeRemoveContactLater_process(self):
        """Process queued contact operations."""
        # Use a lock to ensure thread safety
        with self.dem.getContacts().manipMutex:
            # Process removals
            for removeContacts in self.rremoveContacts:
                for C in removeContacts:
                    self.dem.getContacts().removeMaybe_fast(C)
                removeContacts.clear()

            # Process creations
            for makeContacts in self.mmakeContacts:
                for C in makeContacts:
                    self.dem.getContacts().addMaybe_fast(C)
                makeContacts.clear()

    def spatialOverlap(self, id1, id2):
        """Check if two particles overlap in space."""
        assert not self.periodic
        return (
            self.minima[3 * id1 + 0] <= self.maxima[3 * id2 + 0]
            and self.maxima[3 * id1 + 0] >= self.minima[3 * id2 + 0]
            and self.minima[3 * id1 + 1] <= self.maxima[3 * id2 + 1]
            and self.maxima[3 * id1 + 1] >= self.minima[3 * id2 + 1]
            and self.minima[3 * id1 + 2] <= self.maxima[3 * id2 + 2]
            and self.maxima[3 * id1 + 2] >= self.minima[3 * id2 + 2]
        )

    def handleBoundInversion(self, id1, id2, separating):
        """Handle bound inversion between two particles."""
        assert not self.periodic
        assert id1 != id2

        overlap = False if separating else self.spatialOverlap(id1, id2)

        C = self.dem.getContacts().find(id1, id2)
        hasCon = C is not None

        if not overlap and not hasCon:
            return
        if overlap and hasCon:
            return

        if overlap:
            p1 = self.particles[id1]
            p2 = self.particles[id2]
            if not self.mayCollide(self.dem, p1, p2):
                return
            self.makeContactLater(p1, p2, Vector3i(0, 0, 0))
            return

        if not overlap and hasCon:
            if not C.isReal():
                self.removeContactLater(C)
            return

    def throwTooManyPasses(self):
        """Throw exception when sort doesn't converge."""
        if self.maxSortPass > 0:
            msg = f"maxSortPass={self.maxSortPass} partial sort passes"
        else:
            msg = f"{-self.maxSortPass * omp_get_max_threads()} partial sort passes (maxSortPass={self.maxSortPass} with {omp_get_max_threads()} OpenMP threads)"

        raise RuntimeError(
            f"InsertionSort: sort not done after {msg}. "
            "If motion is out-of-control, increasing Scene.dtSafety might help."
        )

    def insertionSort(self, v, doCollide=True, ax=0):
        """Sort bounds along an axis."""
        assert not self.periodic
        assert v.size == len(v.vec)
        if v.size == 0:
            return

        # Parallel sort configuration
        chunks = self.sortChunks
        if chunks <= 0:
            # Auto-determine number of chunks
            if self.ompTuneSort[0] <= 0:
                chunks = 1
            else:
                chunks = max(
                    1, min(omp_get_max_threads(), v.size // self.ompTuneSort[0])
                )
                if chunks > 1 and v.size < self.ompTuneSort[1]:
                    chunks = 1
                if self.ompTuneSort[2] > 0:
                    chunks = min(chunks, self.ompTuneSort[2])

        # Single-threaded case
        if chunks == 1:
            self.insertionSort_part(v, doCollide, ax, 0, v.size, 0)
            return

        # Multi-threaded case
        splits = [0] * (chunks + 1)
        for i in range(1, chunks):
            splits[i] = (i * v.size) // chunks
        splits[chunks] = v.size

        # Parallel sort
        for chunk in range(chunks):
            self.insertionSort_part(
                v, doCollide, ax, splits[chunk], splits[chunk + 1], splits[chunk]
            )

    def insertionSort_part(self, v, doCollide, ax, iBegin, iEnd, iStart):
        """Sort a part of the bounds vector."""
        assert not self.periodic
        assert v.size > 0
        assert iBegin < iEnd

        earlyStop = iBegin != iStart

        for i in range(max(iStart, iBegin), iEnd):
            i_1 = i - 1

            if v[i_1].coord <= v[i].coord:
                if earlyStop:
                    return
                continue

            j = i_1
            vi = v[i]
            viHasBB = vi.hasBB
            viIsMin = vi.isMin
            viIsInf = vi.isInf

            while j >= iBegin and v[j].coord > vi.coord:
                v[j + 1] = v[j]

                if viIsMin != v[j].isMin and doCollide and viHasBB and v[j].hasBB:
                    if vi.id != v[j].id:
                        self.handleBoundInversion(
                            vi.id,
                            v[j].id,
                            not viIsMin and not viIsInf and not v[j].isInf,
                        )

                j -= 1

            v[j + 1] = vi

    def shouldBeRemoved(self, C, scene):
        """Check if a contact should be removed."""
        if C.pA is None or C.pB is None or C.pA() is None or C.pB() is None:
            return True  # Remove contact if either particle has been deleted

        id1 = C.leakPA().getId()
        id2 = C.leakPB().getId()

        if not self.periodic:
            return not self.spatialOverlap(id1, id2)
        else:
            periods = Vector3i(0, 0, 0)
            return not self.spatialOverlapPeri(id1, id2, scene, periods)

    def updateScenePtr(self, scene, field):
        """Update scene and field pointers."""
        super().updateScenePtr(scene, field)

        self.dem = field
        self.particles = field.getParticles()
        self.periodic = scene.isPeriodic

    def run(self):
        """Run the collision detection algorithm."""
        # Check if we need to do a full run
        if not self.prologue_doFullRun():
            return

        # Update bounding boxes
        if not self.updateBboxes_doFullRun():
            return

        # Initialize bounds arrays
        n_particles = len(self.particles)
        self.minima = np.zeros(3 * n_particles)
        self.maxima = np.zeros(3 * n_particles)

        # Fill bounds arrays
        for i, p in enumerate(self.particles):
            if not p or not p.getShape() or not p.getShape().bound:
                continue

            aabb = p.getShape().bound
            self.minima[3 * i : 3 * i + 3] = aabb.min
            self.maxima[3 * i : 3 * i + 3] = aabb.max

        # Initialize bounds vectors
        for ax in range(3):
            self.BB[ax].clear()

            for i, p in enumerate(self.particles):
                if not p or not p.getShape() or not p.getShape().bound:
                    continue

                # Add min bound
                bmin = self.Bounds(self.minima[3 * i + ax], i, True)
                bmin.hasBB = True
                self.BB[ax].vec.append(bmin)

                # Add max bound
                bmax = self.Bounds(self.maxima[3 * i + ax], i, False)
                bmax.hasBB = True
                self.BB[ax].vec.append(bmax)

            self.BB[ax].size = len(self.BB[ax].vec)

        # Sort bounds
        if self.periodic:
            for ax in range(3):
                self.BB[ax].updatePeriodicity(self.scene)
                self.insertionSortPeri(self.BB[ax], ax != 0, ax)
        else:
            for ax in range(3):
                self.insertionSort(self.BB[ax], ax != 0, ax)

        # Process contacts
        self.makeRemoveContactLater_process()

        # Count this as a full run
        self.nFullRuns += 1

    def prologue_doFullRun(self):
        """Check if we need to do a full collision detection run."""
        # Always do a full run if forced
        if self.forceInitSort:
            self.forceInitSort = False
            return True

        # Check if we have particles
        if not self.particles or len(self.particles) == 0:
            return False

        # Check if we need to reinitialize
        if self.numReinit > 0:
            self.numReinit -= 1
            return True

        # Check maximum velocity
        if self.maxVel2 > 0:
            maxVelSq = 0.0
            for p in self.particles:
                if not p or not p.getShape():
                    continue

                for node in p.getShape().getNodes():
                    if not node.hasDataTyped(DEMData):
                        continue

                    velSq = np.sum(node.getDataTyped(DEMData).vel ** 2)
                    maxVelSq = max(maxVelSq, velSq)

            if maxVelSq > self.maxVel2:
                return True

        # Count inversions
        invs = self.countInversions()
        self.stepInvs = invs
        self.numInvs += invs

        # Do full run if we have inversions
        return invs[0] > 0 or invs[1] > 0 or invs[2] > 0

    def updateBboxes_doFullRun(self):
        """Update bounding boxes and check if we need to do a full run."""
        # Check if we have particles
        if not self.particles or len(self.particles) == 0:
            return False

        # Update Verlet distance
        self.setVerletDist(self.scene, self.dem)

        # Update bounding boxes
        for p in self.particles:
            if not p or not p.getShape():
                continue

            if not p.getShape().bound or self.aabbIsDirty(p):
                self.updateAabb(p)

        return True

    def countInversions(self):
        """Count inversions in bounds arrays."""
        invs = Vector3i(0, 0, 0)

        for ax in range(3):
            v = self.BB[ax]
            if v.size == 0:
                continue

            if self.periodic:
                # Count periodic inversions
                for i in range(v.size):
                    i_1 = v.norm(i - 1)
                    if (i == v.loIdx and v[i].coord < 0) or v[i_1].coord > v[
                        i
                    ].coord + (i == v.loIdx and v.cellDim or 0):
                        invs[ax] += 1
            else:
                # Count non-periodic inversions
                for i in range(1, v.size):
                    if v[i - 1].coord > v[i].coord:
                        invs[ax] += 1

        return invs

    def probeAabb(self, min_corner, max_corner):
        """Find particles within an AABB."""
        ret = []

        # Check each particle
        for i in range(len(self.particles)):
            if (
                self.minima[3 * i] <= max_corner[0]
                and self.maxima[3 * i] >= min_corner[0]
                and self.minima[3 * i + 1] <= max_corner[1]
                and self.maxima[3 * i + 1] >= min_corner[1]
                and self.minima[3 * i + 2] <= max_corner[2]
                and self.maxima[3 * i + 2] >= min_corner[2]
            ):
                ret.append(i)

        return ret

    # Periodic boundary methods
    def spatialOverlapPeri(self, id1, id2, scene, periods):
        """Check if two particles overlap with periodic boundary conditions."""
        assert self.periodic

        # Initialize periods vector
        periods.fill(0)

        # Check overlap along each axis
        for axis in range(3):
            min1 = self.minima[3 * id1 + axis]
            max1 = self.maxima[3 * id1 + axis]
            min2 = self.minima[3 * id2 + axis]
            max2 = self.maxima[3 * id2 + axis]

            # Get cell dimension for this axis
            dim = scene.cell.getSize()[axis]

            # Check overlap along this axis
            period = 0
            if not self.spatialOverlapPeri_axis(
                axis, id1, id2, min1, max1, min2, max2, dim, period
            ):
                return False

            periods[axis] = period

        return True

    def spatialOverlapPeri_axis(
        self, axis, id1, id2, min1, max1, min2, max2, dim, period
    ):
        """Check overlap along one axis with periodic boundaries."""
        # Direct overlap without wrapping
        if min1 <= max2 and max1 >= min2:
            period = 0
            return True

        # Particle 2 overlaps with particle 1 across lower boundary
        if min2 + dim <= max1:
            period = -1
            return True

        # Particle 2 overlaps with particle 1 across upper boundary
        if max2 - dim >= min1:
            period = 1
            return True

        return False

    def cellWrap(self, x, x0, x1, period):
        """Wrap coordinate to periodic cell."""
        if x < x0:
            period -= 1
            return x + (x1 - x0)
        elif x > x1:
            period += 1
            return x - (x1 - x0)
        return x

    def cellWrapRel(self, x, x0, x1, period):
        """Wrap relative coordinate to periodic cell with period tracking."""
        dim = x1 - x0
        if x < x0:
            period -= 1
            return x + dim
        elif x > x1:
            period += 1
            return x - dim
        return x

    def insertionSortPeri(self, v, doCollide=True, ax=0):
        """Sort bounds with periodic boundaries."""
        assert self.periodic
        assert v.size == len(v.vec)
        if v.size == 0:
            return

        # Parallel sort configuration
        chunks = self.sortChunks
        if chunks <= 0:
            # Auto-determine number of chunks
            if self.ompTuneSort[0] <= 0:
                chunks = 1
            else:
                chunks = max(
                    1, min(omp_get_max_threads(), v.size // self.ompTuneSort[0])
                )
                if chunks > 1 and v.size < self.ompTuneSort[1]:
                    chunks = 1
                if self.ompTuneSort[2] > 0:
                    chunks = min(chunks, self.ompTuneSort[2])

        # Single-threaded case
        if chunks == 1 or not self.paraPeri:
            self.insertionSortPeri_part(v, doCollide, ax, 0, v.size, 0)
            return

        # Multi-threaded case
        splits0 = [0] * (chunks + 1)
        splits1 = [0] * (chunks + 1)

        for i in range(1, chunks):
            splits0[i] = (i * v.size) // chunks
        splits0[chunks] = v.size
        splits1[0] = 0

        for i in range(1, chunks + 1):
            splits1[i] = splits1[0] + v.size

        isOrdered = False
        pass_count = 0
        max_passes = (
            self.maxSortPass
            if self.maxSortPass > 0
            else (-self.maxSortPass * omp_get_max_threads())
        )

        while not isOrdered and pass_count < max_passes:
            even = pass_count % 2 == 0
            splits = splits0 if even else splits1

            # Parallel sort
            for chunk in range(chunks):
                start = (
                    splits[chunk]
                    if pass_count == 0
                    else (splits1[chunk] if even else splits0[chunk + 1])
                )
                self.insertionSortPeri_part(
                    v, doCollide, ax, splits[chunk], splits[chunk + 1], start
                )

            # Check if sorted
            isOrdered = True
            for chunk in range(chunks):
                i = v.norm(splits[chunk])
                i_1 = v.norm(i - 1)
                if (i == v.loIdx and v[i].coord < 0) or v[i_1].coord > v[i].coord + (
                    i == v.loIdx and v.cellDim or 0
                ):
                    isOrdered = False
                    break

            pass_count += 1

        if not isOrdered:
            self.throwTooManyPasses()

    def insertionSortPeri_part(self, v, doCollide, ax, iBegin, iEnd, iStart):
        """Sort a part of the bounds vector with periodic boundaries."""
        assert self.periodic
        assert v.size > 0
        assert iBegin < iEnd
        assert v.norm(iBegin) == iBegin
        assert 0 <= iEnd
        assert iStart < iEnd

        earlyStop = iBegin != iStart
        partial = v.norm(iBegin) != v.norm(iEnd)

        for _i in range(max(iStart, iBegin), iEnd):
            i = v.norm(_i)
            i_1 = v.norm(i - 1)

            if i == v.loIdx and v[i].coord < 0 and (not partial or i != iEnd - 1):
                v[i].period -= 1
                v[i].coord += v.cellDim
                v.loIdx = v.norm(v.loIdx + 1)

            iCmpCoord = v[i].coord + (i == v.loIdx and v.cellDim or 0)
            if v[i_1].coord <= iCmpCoord:
                if earlyStop:
                    return
                continue

            j = i_1
            vi = v[i]
            viHasBB = vi.hasBB
            viIsMin = vi.isMin
            viIsInf = vi.isInf

            while (not partial or j >= iBegin) and v[j].coord > vi.coord + (
                v.norm(j + 1) == v.loIdx and v.cellDim or 0
            ):
                j1 = v.norm(j + 1)
                vNew = v[j1] = v[j]

                if j == v.loIdx and vi.coord < 0:
                    vi.period -= 1
                    vi.coord += v.cellDim
                    v.loIdx = v.norm(v.loIdx + 1)
                elif j1 == v.loIdx:
                    vNew.period += 1
                    vNew.coord -= v.cellDim
                    v.loIdx = v.norm(v.loIdx - 1)

                if viIsMin != v[j].isMin and doCollide and viHasBB and v[j].hasBB:
                    if vi.id != vNew.id:
                        self.handleBoundInversionPeri(
                            vi.id,
                            vNew.id,
                            not viIsMin and not viIsInf and not v[j].isInf,
                        )

                j = v.norm(j - 1)

            v[v.norm(j + 1)] = vi

    def handleBoundInversionPeri(self, id1, id2, separating):
        """Handle bound inversion with periodic boundaries."""
        assert self.periodic
        assert id1 != id2

        periods = Vector3i(0, 0, 0)
        overlap = (
            False
            if separating
            else self.spatialOverlapPeri(id1, id2, self.scene, periods)
        )

        C = self.dem.getContacts().find(id1, id2)
        hasCon = C is not None

        if not overlap and not hasCon:
            return
        if overlap and hasCon:
            return

        if overlap:
            p1 = self.particles[id1]
            p2 = self.particles[id2]
            if not self.mayCollide(self.dem, p1, p2):
                return
            self.makeContactLater(p1, p2, periods)
            return

        if not overlap and hasCon:
            if not C.isReal():
                self.removeContactLater(C)
            return


# Import at the end to avoid circular imports
from pydem.src.DEMData import DEMData
