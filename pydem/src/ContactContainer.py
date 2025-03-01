#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterator, TypeVar, Generic
from collections import defaultdict
import weakref

from pydem.src.Object import Object
from pydem.src.Contact import Contact
from pydem.src.demmath import Real
from pydem.src.FilterIterator import filter_iterator
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.OpenMPSimulator import omp_get_thread_num, omp_get_max_threads
from pydem.src.utils import ptr_to_string


@DEM_LOGGER
class ContactContainer(Object):
    """Container for storing and managing contacts between particles."""

    class PendingContact:
        """Helper class for storing pending contact removal."""

        def __init__(self, contact, force=False):
            self.contact = contact
            self.force = force

    class IsReal:
        """Helper class for filtering real contacts."""

        def __call__(self, c):
            return c is not None and c.isReal()

    # Type definitions equivalent to C++ using statements
    T = TypeVar("T")
    ContactVector = List[Contact]
    iterator = filter_iterator
    const_iterator = filter_iterator

    def __init__(self):
        """Initialize contact container with default values."""
        super().__init__()
        self.linView = []  # Linear storage of contacts
        self.nullContactPtr = None  # For returning when not found
        self.dirty = False  # Flag for collider notification
        self.stepColliderLastRun = -1  # Last step when collider was run
        self.dem = None  # Reference to DEM field
        self.particles = None  # Reference to particle container

        self.manipMutex = threading.RLock()  # Mutex for thread-safe operations

        # Initialize thread-specific pending lists
        max_threads = omp_get_max_threads()
        self.threadsPending = [[] for _ in range(max_threads)]

        self.debug(f"ContactContainer initialized with {max_threads} thread slots")

    def add(self, c, threadSafe=False):
        """Add a contact to the container."""
        if c is None:
            return False

        with self.manipMutex:
            pA = c.leakPA()
            pB = c.leakPB()

            if pA is None or pB is None:
                self.warning(f"Skipping contact with null particle: {c.toString()}")
                return False

            # Create weak references to particles
            c.pA = weakref.ref(pA) if not callable(c.pA) else c.pA
            c.pB = weakref.ref(pB) if not callable(c.pB) else c.pB

            # Initialize contacts dict if needed
            if not hasattr(pA, "contacts"):
                pA.contacts = {}
            if not hasattr(pB, "contacts"):
                pB.contacts = {}

            # Check for existing contacts
            if not threadSafe:
                if pB.getId() in pA.contacts or pA.getId() in pB.contacts:
                    self.warning(f"Contact ##{pA.getId()}+{pB.getId()} already exists!")
                    return False

            # Store contacts in particles
            pA.contacts[pB.getId()] = c
            pB.contacts[pA.getId()] = c

            # Add to linear view
            c.linIx = len(self.linView)
            self.linView.append(c)
            # self.dirty = True
            return True

    def addMaybe_fast(self, c):
        """Add a contact if it doesn't exist (non-thread-safe version)."""
        pA = c.leakPA()
        pB = c.leakPB()

        if pA is None or pB is None:
            return

        if pB.getId() in self.particles[pA.getId()].contacts:
            return

        pA.contacts[pB.getId()] = c
        pB.contacts[pA.getId()] = c
        self.linView.append(c)
        c.linIx = len(self.linView) - 1

        self.dirty = True

        return

    def remove(self, c, threadSafe=False):
        """Remove a contact from the container."""
        if c is None:
            return False

        with self.manipMutex:
            pA = c.getParticleA()
            pB = c.getParticleB()

            # Handle particle removal
            for p in [pA, pB]:
                if p is None:
                    continue

                id = p.getId()
                id2 = -1
                if pA is not None and pB is not None:
                    id2 = pB.getId() if p is pA else pA.getId()

                if id2 >= 0:
                    if id2 not in p.contacts:
                        if not threadSafe:
                            if (
                                id2 >= len(self.particles)
                                or p is not self.particles[id2]
                            ):
                                return False
                            self.fatal(f"Contact ##{id}+{id2} vanished from particle!")
                        return False

                    del p.contacts[id2]

            if pA is None and pB is None:
                if len(self.linView) > c.linIx and self.linView[c.linIx] is c:
                    raise RuntimeError(
                        f"Contact @ {ptr_to_string(c)} exists in linView but both particles vanished"
                    )
                return False

            self.linView_remove(c.linIx)
            return True

    def removeMaybe_fast(self, c):
        """Remove a contact if it exists (non-thread-safe version)."""
        pA = c.leakPA()
        pB = c.leakPB()

        if pA is None or pB is None:
            return

        if pB.getId() not in pA.contacts:
            return

        del pA.contacts[pB.getId()]
        del pB.contacts[pA.getId()]
        self.linView_remove(c.linIx)

    def linView_remove(self, ix):
        """Remove a contact from linear view by index."""
        if ix < 0 or ix >= len(self.linView):
            self.warning(
                f"Invalid index {ix} for linView_remove (size={len(self.linView)})"
            )
            return

        if ix < len(self.linView) - 1:
            self.linView[ix] = self.linView[-1]
            self.linView[ix].linIx = ix

        self.linView.pop()

    def clear(self):
        """Clear all contacts from the container."""
        with self.manipMutex:
            count = len(self.linView)
            if self.particles:
                for p in self.particles:
                    if p is not None:
                        p.contacts.clear()

            self.linView.clear()
            self.clearPending()
            self.dirty = True
            self.debug(f"Cleared {count} contacts")

    def find(self, idA, idB):
        """Find a contact between two particles by their IDs."""
        if not self.particles or not self.particles.exists(idA):
            return self.nullContactPtr

        if idB in self.particles[idA].contacts:
            return self.particles[idA].contacts[idB]

        return self.nullContactPtr

    def exists(self, idA, idB):
        """Check if a contact exists between two particles by their IDs."""
        if not self.particles or not self.particles.exists(idA):
            return False

        return idB in self.particles[idA].contacts

    def existsReal(self, idA, idB):
        """Check if a real contact exists between two particles by their IDs."""
        c = self.find(idA, idB)
        return c is not None and c.isReal()

    def requestRemoval(self, c, force=False):
        """Request removal of a contact."""
        if c is None:
            self.warning("Attempt to request removal of null contact")
            return

        c.reset()

        # Store in thread-specific pending list
        pending_contact = self.PendingContact(c, force)

        thread_id = omp_get_thread_num()
        self.threadsPending[thread_id].append(pending_contact)
        self.debug(f"Contact {c.toString()} queued for removal in thread {thread_id}")

    def clearPending(self):
        """Clear all pending contact removals."""
        count = sum(len(pending) for pending in self.threadsPending)
        for pending in self.threadsPending:
            pending.clear()
        self.debug(f"Cleared {count} pending removals")

    def removeAllPending(self):
        """Remove all pending contacts."""
        removed = 0
        for pending in self.threadsPending:
            for p in pending:
                if self.remove(p.contact):
                    removed += 1
            pending.clear()

        if removed > 0:
            self.debug(f"Removed {removed} pending contacts")
        return removed

    def removeNonReal(self):
        """Remove all non-real contacts."""
        to_remove = []
        for c in self.linView:
            if not c.isReal():
                to_remove.append(c)

        count = len(to_remove)
        for c in to_remove:
            self.remove(c)

        if count > 0:
            self.debug(f"Removed {count} non-real contacts")
        return count

    def countReal(self):
        """Count the number of real contacts."""
        return sum(1 for c in self.linView if c is not None and c.isReal())

    def realRatio(self):
        """Get the ratio of real contacts to total contacts."""
        if not self.linView:
            return 0
        return float(self.countReal()) / len(self.linView)

    def removePending(self, pred, scene):
        """Remove pending contacts based on a predicate."""
        removed = 0
        for pending in self.threadsPending:
            to_remove = []
            for i, p in enumerate(pending):
                if p.force or pred.shouldBeRemoved(p.contact, scene):
                    to_remove.append(i)
                    if self.remove(p.contact):
                        removed += 1

            # Remove from list in reverse order to avoid index issues
            for i in reversed(to_remove):
                pending.pop(i)

        if removed > 0:
            self.debug(f"Removed {removed} contacts via predicate")
        return removed

    def __getitem__(self, ix):
        """Get contact by index."""
        if ix < 0 or ix >= len(self.linView):
            raise IndexError(
                f"ContactContainer index {ix} out of range (0-{len(self.linView)-1})"
            )
        return self.linView[ix]

    def size(self):
        """Get the number of contacts."""
        return len(self.linView)

    # Iterator methods
    def begin(self):
        """Get iterator to the beginning of real contacts."""
        it = filter_iterator(
            0, len(self.linView), lambda i: self.IsReal()(self.linView[i])
        )
        it.container = self.linView
        return it

    def end(self):
        """Get iterator to the end of real contacts."""
        it = filter_iterator(
            len(self.linView),
            len(self.linView),
            lambda i: self.IsReal()(self.linView[i]),
        )
        it.container = self.linView
        return it

    def __iter__(self):
        """Get iterator for all real contacts."""
        it = filter_iterator(
            0, len(self.linView), lambda i: self.IsReal()(self.linView[i])
        )
        it.container = self.linView
        return it

    def toString(self):
        """Return string representation of the container."""
        return f"ContactContainer with {len(self.linView)} contacts ({self.countReal()} real)"

    def __str__(self):
        """String representation for print()."""
        return self.toString()

    def __len__(self):
        """Return number of contacts (for len())."""
        return self.size()
