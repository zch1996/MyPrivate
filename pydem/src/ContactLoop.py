#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Optional, Dict, Set, Tuple, Any
from enum import IntEnum
import threading
import numpy as np

from pydem.src.demmath import Vector3r, Matrix3r, Real, NAN
from pydem.src.Engine import Engine
from pydem.src.ContactHook import ContactHook
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.Dispatcher import (
    CGeomDispatcher,
    CPhysDispatcher,
    LawDispatcher,
    IntraDispatcher,
)


class UpdatePhys(IntEnum):
    """Enum for contact physics update modes."""

    UPDATE_PHYS_NEVER = 0
    UPDATE_PHYS_ALWAYS = 1
    UPDATE_PHYS_ONCE = 2


@DEM_LOGGER
class ContactLoop(Engine):
    """
    Engine that processes all contacts in the simulation.

    This engine handles:
    1. Contact geometry updates
    2. Contact physics updates
    3. Application of contact laws
    4. Force application
    5. Stress calculation
    """

    def __init__(self):
        """Initialize ContactLoop with default values."""
        super().__init__()

        # Dispatchers
        self.geoDisp = CGeomDispatcher()  # CGeomDispatcher
        self.phyDisp = CPhysDispatcher()  # CPhysDispatcher
        self.lawDisp = LawDispatcher()  # LawDispatcher
        self.hook = None  # ContactHook

        # Configuration
        self.applyForces = True
        self.evalStress = False
        self.updatePhys = UpdatePhys.UPDATE_PHYS_NEVER
        self.dist00 = True
        self.reorderEvery = 1000

        # Internal state
        self.alreadyWarnedNoCollider = False
        self.stress = np.zeros((3, 3))
        self.prevVol = NAN
        self.prevStress = np.zeros((3, 3))
        self.gradVIx = -1

        # Thread-safe contact removal
        self.removeAfterLoopRefs = []
        for _ in range(threading.active_count()):
            self.removeAfterLoopRefs.append([])

    def updateScenePtr(self, scene, field):
        """
        Update scene and field pointers and initialize dispatchers.

        Args:
            scene: Scene object
            field: Field object
        """
        super().updateScenePtr(scene, field)

        # Initialize functors in dispatchers
        self.geoDisp.initializeFunctors()
        self.phyDisp.initializeFunctors()
        self.lawDisp.initializeFunctors()

        # Update dispatchers' scene and field
        self.geoDisp.updateScenePtr(scene, field)
        self.phyDisp.updateScenePtr(scene, field)
        self.lawDisp.updateScenePtr(scene, field)

    def reorderContacts(self):
        """Reorder contacts to optimize processing (real contacts first)."""
        dem = self.field
        contacts = dem.getContacts()

        size = len(contacts)
        if size < 2:
            return

        lowestReal = 0

        # Move real contacts to the beginning
        for i in range(size - 1, lowestReal, -1):
            contact = contacts[i]
            if not contact.isReal():
                continue

            while lowestReal < i and contacts[lowestReal].isReal():
                lowestReal += 1

            if lowestReal == i:
                return

            # Swap contacts
            contacts[i], contacts[lowestReal] = contacts[lowestReal], contacts[i]
            contacts[i].linIx = i
            contacts[lowestReal].linIx = lowestReal

    def applyForceUninodal(self, contact, particle):
        """
        Apply force and torque to a uninodal particle.

        Args:
            contact: Contact object
            particle: Particle object
        """
        shape = particle.getShape()
        if not shape or len(shape.getNodes()) != 1:
            return

        # Get force, torque and branch vector
        force, torque, branch = contact.getForceTorqueBranch(particle, 0, self.scene)

        # Apply force and torque to the node
        shape.nodes[0].getData("DEM").addForceTorque(force, torque)

    def removeAfterLoop(self, contact):
        """
        Mark a contact for removal after the contact loop.

        Args:
            contact: Contact to remove
        """
        thread_id = threading.get_ident() % len(self.removeAfterLoopRefs)
        self.removeAfterLoopRefs[thread_id].append(contact)

    def run(self):
        """Run the contact loop."""
        dem = self.field
        contacts = dem.getContacts()

        # Remove pending contacts
        if contacts.removeAllPending() > 0 and not self.alreadyWarnedNoCollider:
            self.warning(
                "Contacts pending removal found (and removed); no collider being used?"
            )
            self.alreadyWarnedNoCollider = True

        if contacts.dirty:
            raise RuntimeError(
                "ContactContainer is dirty; collider should reinitialize"
            )

        # Cache transformed cell size
        cellHsize = None
        if self.scene.isPeriodic:
            cellHsize = self.scene.cell.getHSize()

        self.stress = np.zeros((3, 3))

        # Check for automatic contact removal
        removeUnseen = (
            contacts.stepColliderLastRun >= 0
            and contacts.stepColliderLastRun == self.scene.step
        )

        doStress = self.evalStress and self.scene.isPeriodic
        deterministic = self.scene.deterministic

        # Reorder contacts if needed
        if self.reorderEvery > 0 and (self.scene.step % self.reorderEvery == 0):
            self.reorderContacts()

        size = len(contacts)
        hasHook = self.hook is not None

        # Process all contacts
        for i in range(size):
            contact = contacts[i]

            # Remove non-colliding contacts
            if (
                removeUnseen
                and not contact.isReal()
                and contact.stepLastSeen < self.scene.step
            ):
                self.removeAfterLoop(contact)
                continue

            if not contact.isReal() and not contact.isColliding():
                self.removeAfterLoop(contact)
                continue

            # Handle fresh contacts
            if not contact.isReal() and contact.isFresh(self.scene):
                swap = False
                geomFunctor, swap = self.geoDisp.getFunctor(
                    contact.leakPA().getShape().getShapeType(),
                    contact.leakPB().getShape().getShapeType(),
                )

                if not geomFunctor:
                    continue

                if swap:
                    contact.swapOrder()

                geomFunctor.setMinDist00Sq(
                    contact.getParticleA().getShape(),
                    contact.getParticleB().getShape(),
                    contact,
                )

            # Skip non-colliding contacts
            if not contact.isColliding():
                continue

            # Get particles
            pA = contact.getParticleA()
            pB = contact.getParticleB()

            # Skip if either particle has no shape or material
            if (
                not pA.getShape()
                or not pB.getShape()
                or not pA.getMaterial()
                or not pB.getMaterial()
            ):
                continue

            # Calculate shift for periodic boundaries
            shift2 = Vector3r(0, 0, 0)
            if self.scene.isPeriodic:
                shift2 = self.scene.cell.intrShiftPos(contact.getCellDist())

            # Update geometry
            geomCreated = False
            swap = False

            if contact.isReal():
                # Use existing geometry functor
                geomFunctor = self.geoDisp.getFunctor(
                    pA.getShape().getShapeType(), pB.getShape().getShapeType(), swap
                )

                if geomFunctor:
                    geomCreated = geomFunctor.go(
                        pA.getShape(), pB.getShape(), shift2, False, contact
                    )
            else:
                # Try to create geometry
                geomFunctor, swap = self.geoDisp.getFunctor(
                    pA.getShape().getShapeType(), pB.getShape().getShapeType()
                )

                if geomFunctor:
                    geomCreated = geomFunctor.go(
                        pA.getShape(), pB.getShape(), shift2, False, contact
                    )

            if not geomCreated:
                if contact.isReal():
                    self.error(
                        f"Geometry update failed for contact ##{pA.getId()}+{pB.getId()}"
                    )
                continue

            # Update physics
            if not contact.phys:
                contact.stepCreated = self.scene.step

            if not contact.phys or self.updatePhys > UpdatePhys.UPDATE_PHYS_NEVER:
                swap2 = False
                physFunctor = self.phyDisp.getFunctor(
                    pA.getMaterial().getMaterialType(),
                    pB.getMaterial().getMaterialType(),
                    swap2,
                )

                if physFunctor:
                    physFunctor.go(pA.getMaterial(), pB.getMaterial(), contact)

            if not contact.phys:
                raise RuntimeError(
                    f"No physics created for contact ##{pA.getId()}+{pB.getId()}"
                )

            # Handle new contacts with hook
            if (
                hasHook
                and contact.isFresh(self.scene)
                and self.hook.isMatch(pA.getMask(), pB.getMask())
            ):
                self.hook.hookNew(dem, contact)

            # Apply constitutive law
            keepContact = False
            lawFunctor = self.lawDisp.getFunctor(contact.geom, contact.phys)

            if lawFunctor:
                keepContact = lawFunctor.go(contact.geom, contact.phys, contact)

            if not keepContact:
                if hasHook and self.hook.isMatch(pA.getMask(), pB.getMask()):
                    self.hook.hookDel(dem, contact)
                contacts.requestRemoval(contact)
                continue

            # Apply forces
            if self.applyForces and contact.isReal() and not deterministic:
                self.applyForceUninodal(contact, pA)
                self.applyForceUninodal(contact, pB)

            # Track stress
            if doStress and contact.isReal():
                nodesA = pA.getShape().getNodes()
                nodesB = pB.getShape().getNodes()

                if len(nodesA) != 1 or len(nodesB) != 1:
                    raise RuntimeError(
                        "Stress calculation not supported for multi-node particles"
                    )

                branch = contact.getPositionDifference(self.scene)
                force = contact.geom.getNode().ori * contact.phys.getForce()

                self.stress += np.outer(force, branch)

        # Process removed contacts
        for thread_list in self.removeAfterLoopRefs:
            for contact in thread_list:
                contacts.remove(contact)
            thread_list.clear()

        # Finalize stress calculation
        if doStress:
            self.stress /= self.scene.cell.getVolume()

            if self.scene.trackEnergy:
                midStress = 0.5 * (self.stress + self.prevStress)
                midVol = (
                    0.5 * (self.prevVol + self.scene.cell.getVolume())
                    if not np.isnan(self.prevVol)
                    else self.scene.cell.getVolume()
                )

                dW = (
                    -(np.trace(self.scene.cell.getGradV() @ midStress))
                    * self.scene.dt
                    * midVol
                )

                self.scene.addEnergy("gradV", dW, False)

            self.prevVol = self.scene.cell.getVolume()
            self.prevStress = self.stress.copy()

        # Apply forces deterministically if needed
        if deterministic and self.applyForces:
            for contact in contacts:
                if not contact.isReal():
                    continue

                self.applyForceUninodal(contact, contact.getParticleA())
                self.applyForceUninodal(contact, contact.getParticleB())

        # Reset updatePhys if it was one-time
        if self.updatePhys == UpdatePhys.UPDATE_PHYS_ONCE:
            self.updatePhys = UpdatePhys.UPDATE_PHYS_NEVER

    # Accessor methods
    def setGeoDispatcher(self, dispatcher):
        """Set geometry dispatcher."""
        self.geoDisp = dispatcher

    def setPhysDispatcher(self, dispatcher):
        """Set physics dispatcher."""
        self.phyDisp = dispatcher

    def setLawDispatcher(self, dispatcher):
        """Set law dispatcher."""
        self.lawDisp = dispatcher

    def setContactHook(self, hook):
        """Set contact hook."""
        self.hook = hook

    def setEvalStress(self, eval_stress):
        """Set whether to evaluate stress."""
        self.evalStress = eval_stress

    def setApplyForces(self, apply_forces):
        """Set whether to apply forces."""
        self.applyForces = apply_forces

    def setUpdatePhys(self, update_phys):
        """Set physics update mode."""
        self.updatePhys = update_phys

    def setDist00(self, dist00):
        """Set whether to use dist00."""
        self.dist00 = dist00

    def setReorderEvery(self, reorder_every):
        """Set how often to reorder contacts."""
        self.reorderEvery = reorder_every
