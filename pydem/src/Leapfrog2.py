#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import weakref
import math

from pydem.src.Engine import Engine
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import (
    Vector3r,
    Matrix3r,
    Quaternionr,
    AngleAxisr,
    Real,
    NAN,
    levi_civita,
)
from pydem.src.DEMData import DEMData
from pydem.src.Impose import Impose
from pydem.src.ClumpData import ClumpData
from pydem.src.Cell import DeformationMode


@DEM_LOGGER
class ForceResetter(Engine):
    """Reset forces on nodes in DEM field."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

    def run(self):
        """Reset forces on all nodes."""
        dem = self.field
        hasGravity = (dem.gravity != Vector3r(0, 0, 0)).any()

        for n in dem.nodes:
            dyn = n.getDataTyped(DEMData)

            # Apply gravity if needed
            if hasGravity and not dyn.isGravitySkip():
                dyn.force = dem.gravity * dyn.mass
            else:
                dyn.force = Vector3r(0, 0, 0)

            dyn.torque = Vector3r(0, 0, 0)

            # Apply imposed forces
            if dyn.impose and (dyn.impose.what & Impose.Type.FORCE):
                dyn.impose.force(self.scene, n)

            # Zero gravity on clump members
            if dyn.isClumped():
                ClumpData.resetForceTorque(n)


@DEM_LOGGER
class Leapfrog2(Engine):
    """
    Engine integrating newtonian motion equations using the leap-frog scheme.

    This integrator handles:
    1. Translation and rotation of particles
    2. Damping of motion
    3. Periodic boundary conditions
    4. Energy tracking
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.damping = 0.2  # Damping coefficient for non-viscous damping
        self.reset = False  # Reset forces immediately after applying them
        self._forceResetChecked = False  # Warning flag for force reset
        self.maxVelocitySq = NAN  # Store square of max velocity
        self.dontCollect = False  # Don't collect DEM nodes when none in first step

        # Energy tracking
        self.kinSplit = False  # Whether to separately track translational and rotational kinetic energy
        self.nonviscDampIx = -1  # Index for non-viscous damping energy
        self.gravWorkIx = -1  # Index for gravity work
        self.kinEnergyIx = -1  # Index for kinetic energy
        self.kinEnergyTransIx = -1  # Index for translational kinetic energy
        self.kinEnergyRotIx = -1  # Index for rotational kinetic energy

        # Cached values
        self.IpLL4h = np.zeros((3, 3))
        self.ImLL4hInv = np.zeros((3, 3))
        self.LmL = np.zeros((3, 3))
        self.deltaSpinVec = Vector3r(0, 0, 0)

        # State variables
        self.homoDeform = -1  # Updated from scene at every call
        self.dt = 0.0  # Updated from scene at every call
        self.dGradV = np.zeros((3, 3))  # Updated from scene at every call
        self.midGradV = np.zeros((3, 3))  # Updated from scene at every call

    def nonviscDamp1st(self, force, vel):
        """Apply 1st order numerical damping."""
        for i in range(3):
            force[i] *= 1 - self.damping * np.sign(force[i] * vel[i])
        return force

    def nonviscDamp2nd(self, dt, force, vel, accel):
        """Apply 2nd order numerical damping."""
        for i in range(3):
            accel[i] *= 1 - self.damping * np.sign(
                force[i] * (vel[i] + 0.5 * dt * accel[i])
            )
        return accel

    def computeAccel(self, force, mass, dyn):
        """Compute linear acceleration, respecting blocked DOFs."""
        if dyn.isBlockedNone():
            return force / mass

        ret = Vector3r(0, 0, 0)
        for i in range(3):
            if not dyn.isBlockedAxisDOF(i, False):
                ret[i] = force[i] / mass

        return ret

    def computeAngAccel(self, torque, inertia, dyn):
        """Compute angular acceleration, respecting blocked DOFs."""
        if dyn.isBlockedNone():
            return Vector3r(
                torque[0] / inertia[0], torque[1] / inertia[1], torque[2] / inertia[2]
            )

        ret = Vector3r(0, 0, 0)
        for i in range(3):
            if not dyn.isBlockedAxisDOF(i, True):
                ret[i] = torque[i] / inertia[i]

        return ret

    def doDampingDissipation(self, node):
        """Track energy dissipated by damping."""
        dyn = node.getDataTyped(DEMData)
        if dyn.isEnergySkip():
            return

        # Compute damping dissipation
        transDiss = np.abs(dyn.vel).dot(np.abs(dyn.force)) * self.damping * self.dt
        rotDiss = np.abs(dyn.angVel).dot(np.abs(dyn.torque)) * self.damping * self.dt

        # Add to energy tracker
        self.scene.energy.add(
            transDiss + rotDiss,
            "nonviscDamp",
            self.nonviscDampIx,
            self.scene.energy.IsIncrement | self.scene.energy.ZeroDontCreate,
            node.pos,
        )

    def doGravityWork(self, dyn, dem, pos):
        """Track work done by gravity."""
        if dyn.isGravitySkip():
            return

        gr = 0.0
        if dyn.isBlockedNone():
            gr = -dem.gravity.dot(dyn.vel) * dyn.mass * self.dt
        else:
            for ax in range(3):
                if not dyn.isBlockedAxisDOF(ax, False):
                    gr -= dem.gravity[ax] * dyn.vel[ax] * dyn.mass * self.dt

        self.scene.energy.add(
            gr, "grav", self.gravWorkIx, self.scene.energy.IsIncrement, pos
        )

    def doKineticEnergy(
        self, node, pprevFluctVel, pprevFluctAngVel, linAccel, angAccel
    ):
        """Track kinetic energy."""
        dyn = node.getDataTyped(DEMData)
        if dyn.isEnergySkip():
            return

        # Compute current velocities
        currFluctVel = pprevFluctVel + 0.5 * self.dt * linAccel
        currFluctAngVel = pprevFluctAngVel + 0.5 * self.dt * angAccel

        # Compute translational kinetic energy
        Etrans = 0.5 * dyn.mass * currFluctVel.squaredNorm()

        # Compute rotational kinetic energy
        Erot = 0.0
        if dyn.isAspherical():
            mI = np.diag(dyn.inertia)
            T = node.ori.toRotationMatrix()
            Erot = 0.5 * currFluctAngVel.transpose().dot(
                (T.transpose() * mI * T) * currFluctAngVel
            )
        else:
            Erot = 0.5 * currFluctAngVel.dot((dyn.inertia * currFluctAngVel))

        if math.isnan(Erot) or np.isinf(dyn.inertia.max()):
            Erot = 0

        # Add to energy tracker
        if not self.kinSplit:
            self.scene.energy.add(
                Etrans + Erot,
                "kinetic",
                self.kinEnergyIx,
                self.scene.energy.IsResettable,
            )
        else:
            self.scene.energy.add(
                Etrans,
                "kinTrans",
                self.kinEnergyTransIx,
                self.scene.energy.IsResettable,
            )
            self.scene.energy.add(
                Erot, "kinRot", self.kinEnergyRotIx, self.scene.energy.IsResettable
            )

    def applyPeriodicCorrections(self, node, linAccel):
        """Apply periodic boundary corrections."""
        dyn = node.getDataTyped(DEMData)

        if (
            self.homoDeform == DeformationMode.HOMO_VEL.value
            or self.homoDeform == DeformationMode.HOMO_VEL_2ND.value
        ):
            # Update velocity reflecting changes in macroscopic velocity field
            if self.homoDeform == DeformationMode.HOMO_VEL_2ND.value:
                dyn.vel += (
                    self.midGradV.dot(dyn.vel - (self.dt / 2.0) * linAccel) * self.dt
                )

            # Reflect macroscopic acceleration in velocity
            dyn.vel += self.dGradV.dot(node.pos)

        elif self.homoDeform == DeformationMode.HOMO_POS.value:
            node.pos += self.scene.cell.nextGradV.dot(node.pos) * self.dt

    def leapfrogTranslate(self, node):
        """Update position using current velocity."""
        dyn = node.getDataTyped(DEMData)
        node.pos += dyn.vel * self.dt

    def leapfrogSphericalRotate(self, node):
        """Update orientation for spherical particles."""
        dyn = node.getDataTyped(DEMData)
        axis = dyn.angVel

        if (axis != Vector3r(0, 0, 0)).any():
            angle = axis.norm()
            axis = axis / angle
            q = Quaternionr(AngleAxisr(angle * self.dt, axis))
            node.ori = q * node.ori

        node.ori.normalize()

    def leapfrogAsphericalRotate(self, node, M):
        """Update orientation for aspherical particles."""
        ori = node.ori
        dyn = node.getDataTyped(DEMData)
        angMom = dyn.angMom
        angVel = dyn.angVel
        inertia = dyn.inertia

        # Initialize angular momentum if needed
        if np.isnan(angMom).any():
            angMom = dyn.inertia.asDiagonal() * ori.conjugate() * dyn.angVel
            dyn.angMom = angMom

        # Rotation matrix from global to local reference frame
        A = ori.conjugate().toRotationMatrix()

        # Global angular momentum at time n
        l_n = angMom + self.dt / 2 * M

        # Local angular momentum at time n
        l_b_n = A * l_n

        # Local angular velocity at time n
        angVel_b_n = l_b_n / inertia

        # dQ/dt at time n
        dotQ_n = self.DotQ(angVel_b_n, ori)

        # Q at time n+1/2
        Q_half = ori + self.dt / 2 * dotQ_n

        # Global angular momentum at time n+1/2
        angMom += self.dt * M
        dyn.angMom = angMom

        # Local angular momentum at time n+1/2
        l_b_half = A * angMom

        # Local angular velocity at time n+1/2
        angVel_b_half = l_b_half / inertia

        # dQ/dt at time n+1/2
        dotQ_half = self.DotQ(angVel_b_half, Q_half)

        # Q at time n+1
        ori += self.dt * dotQ_half

        # Global angular velocity at time n+1/2
        angVel = ori * angVel_b_half
        dyn.angVel = angVel

        # Normalize orientation
        ori.normalize()

    def DotQ(self, angVel, Q):
        """Compute quaternion derivative from angular velocity."""
        dotQ = Quaternionr()
        dotQ.w = (-Q.x * angVel[0] - Q.y * angVel[1] - Q.z * angVel[2]) / 2
        dotQ.x = (Q.w * angVel[0] - Q.z * angVel[1] + Q.y * angVel[2]) / 2
        dotQ.y = (Q.z * angVel[0] + Q.w * angVel[1] - Q.x * angVel[2]) / 2
        dotQ.z = (-Q.y * angVel[0] + Q.x * angVel[1] + Q.w * angVel[2]) / 2
        return dotQ

    def run(self):
        """Run the integrator for one time step."""
        if not self.reset and not self._forceResetChecked:
            resetter = None
            for e in self.scene.engines:
                if isinstance(e, ForceResetter):
                    resetter = e
                    break

            if not resetter:
                self.warning(
                    "Leapfrog.reset==False and no ForceResetter in Scene.engines! "
                    "Are you sure this is ok? (proceeding)"
                )
            self._forceResetChecked = True

        # Get scene and field
        scene = self.scene
        dem = self.field

        # Update time step and cell information
        self.homoDeform = -1 if not scene.isPeriodic else scene.cell.homoDeform.value
        self.dGradV = (
            scene.cell.nextGradV - scene.cell.gradV
            if scene.isPeriodic
            else np.zeros((3, 3))
        )
        self.midGradV = (
            0.5 * (scene.cell.gradV + scene.cell.nextGradV)
            if scene.isPeriodic
            else np.zeros((3, 3))
        )
        self.dt = scene.dt

        # Handle HOMO_GRADV2 deformation mode
        if self.homoDeform == DeformationMode.HOMO_GRADV2.value:
            pprevL = scene.cell.gradV
            nnextL = scene.cell.nextGradV
            self.ImLL4hInv = np.linalg.inv(
                np.eye(3) - self.dt * (nnextL + pprevL) / 4.0
            )
            self.IpLL4h = np.eye(3) + self.dt * (nnextL + pprevL) / 4.0
            self.LmL = nnextL - pprevL

            # Difference of spin vectors
            self.deltaSpinVec = -0.5 * levi_civita(
                0.5 * (pprevL - pprevL.transpose())
            ) + 0.5 * levi_civita(0.5 * (nnextL - nnextL.transpose()))

        # Initialize energy trackers if needed
        reallyTrackEnergy = scene.trackEnergy and (
            not scene.isPeriodic or scene.step > 0 or scene.cell.gradV == np.eye(3)
        )

        # Process all nodes
        maxVSq = 0.0
        hasGravity = (dem.gravity != Vector3r(0, 0, 0)).any()

        for node in dem.nodes:
            dyn = node.getDataTyped(DEMData)

            if dyn.isClumped():
                continue

            isClump = dyn.isClump()

            # if isClump, collect force and torque from clump members
            if isClump and not (
                dyn.isBlockedAll()
                or (dyn.impose and (dyn.impose.what & Impose.Type.READ_FORCE))
            ):
                ClumpData.forceTorqueFromMembers(node, dyn.force, dyn.torque)

            # Track maximum velocity
            vSq = np.dot(dyn.vel, dyn.vel)
            maxVSq = max(maxVSq, vSq)

            # Store previous velocity for energy calculations
            pprevFluctVel = Vector3r(0, 0, 0)
            pprevFluctAngVel = Vector3r(0, 0, 0)

            # Determine whether to use aspherical rotation integration
            useAspherical = dyn.useAsphericalLeapfrog()

            # Compute accelerations
            linAccel = Vector3r(0, 0, 0)
            angAccel = Vector3r(0, 0, 0)

            # For particles not totally blocked, compute accelerations
            if not dyn.isBlockedAll():
                linAccel = self.computeAccel(dyn.force, dyn.mass, dyn)

                # Calculate fluctuation velocities
                if scene.isPeriodic:
                    pprevFluctVel = scene.cell.pprevFluctVel(node.pos, dyn.vel, self.dt)
                    pprevFluctAngVel = scene.cell.pprevFluctAngVel(dyn.angVel)
                else:
                    pprevFluctVel = dyn.vel
                    pprevFluctAngVel = dyn.angVel

                # Apply damping if needed
                if self.damping != 0 and not dyn.isDampingSkip():
                    if reallyTrackEnergy:
                        self.doDampingDissipation(node)

                    if useAspherical:
                        self.nonviscDamp1st(dyn.force, pprevFluctVel)
                        self.nonviscDamp1st(dyn.torque, pprevFluctAngVel)
                    else:
                        linAccel = self.nonviscDamp2nd(
                            self.dt, dyn.force, pprevFluctVel, linAccel
                        )
                        angAccel = self.computeAngAccel(dyn.torque, dyn.inertia, dyn)
                        angAccel = self.nonviscDamp2nd(
                            self.dt, dyn.torque, pprevFluctAngVel, angAccel
                        )

                # Compute velocity at t+dt/2
                if self.homoDeform == DeformationMode.HOMO_GRADV2.value:
                    dyn.vel = self.ImLL4hInv.dot(
                        self.LmL.dot(node.pos)
                        + self.IpLL4h.dot(dyn.vel)
                        + linAccel * self.dt
                    )
                else:
                    dyn.vel += self.dt * np.array(
                        linAccel
                    )  # Correction for this case is below

                # Compute angular acceleration if needed
                if not np.allclose(dyn.inertia, Vector3r(0, 0, 0)):
                    if not useAspherical:  # Spherical integrator
                        angAccel = self.computeAngAccel(dyn.torque, dyn.inertia, dyn)
                        if self.damping != 0 and not dyn.isDampingSkip():
                            angAccel = self.nonviscDamp2nd(
                                self.dt, dyn.torque, pprevFluctAngVel, angAccel
                            )
                        dyn.angVel += self.dt * angAccel
                        if self.homoDeform == DeformationMode.HOMO_GRADV2.value:
                            dyn.angVel -= self.deltaSpinVec
                    else:  # Aspherical integrator
                        # Block DOFs if needed
                        for i in range(3):
                            if dyn.isBlockedAxisDOF(i, True):
                                dyn.torque[i] = 0

                        if self.damping != 0 and not dyn.isDampingSkip():
                            self.nonviscDamp1st(dyn.torque, pprevFluctAngVel)

                # Apply velocity impositions if needed
                if dyn.impose and (dyn.impose.what & Impose.Type.VELOCITY):
                    dyn.impose.velocity(scene, node)
            else:
                # Fixed particle with HOMO_GRADV2: velocity correction without acceleration
                if self.homoDeform == DeformationMode.HOMO_GRADV2.value:
                    dyn.vel = self.ImLL4hInv.dot(
                        self.LmL.dot(node.pos) + self.IpLL4h.dot(dyn.vel)
                    )

            # Apply periodic corrections for both fixed and free particles
            if (
                scene.isPeriodic
                and self.homoDeform >= 0
                and self.homoDeform != DeformationMode.HOMO_GRADV2.value
            ):
                self.applyPeriodicCorrections(node, linAccel)

            # Track kinetic energy
            if reallyTrackEnergy:
                self.doKineticEnergy(
                    node, pprevFluctVel, pprevFluctAngVel, linAccel, angAccel
                )

            # Update position
            self.leapfrogTranslate(node)

            # Update orientation
            if not useAspherical:
                self.leapfrogSphericalRotate(node)
            else:
                if np.allclose(dyn.inertia, Vector3r(0, 0, 0)):
                    raise RuntimeError(
                        f"Leapfrog::run: DemField.nodes[{dyn.linIx}].den.inertia==(0,0,0), "
                        f"but the node wants to use aspherical integrator. Aspherical integrator "
                        f"is selected for non-spherical particles which have at least one "
                        f"rotational DOF free."
                    )

                if not scene.isPeriodic:
                    self.leapfrogAsphericalRotate(node, dyn.torque)
                else:
                    # FIXME: add fake torque from rotating space or modify angMom or angVel
                    self.leapfrogAsphericalRotate(node, dyn.torque)

            # Read back forces from the node (before they are reset)
            if dyn.impose and (dyn.impose.what & Impose.Type.READ_FORCE):
                dyn.impose.readForce(scene, node)

            # Reset forces if requested
            if self.reset:
                # Apply gravity only to the clump itself
                if hasGravity and not dyn.isGravitySkip():
                    dyn.force = dyn.mass * dem.gravity
                else:
                    dyn.force = Vector3r(0, 0, 0)

                dyn.torque = Vector3r(0, 0, 0)

                if dyn.impose and (dyn.impose.what & Impose.Type.FORCE):
                    dyn.impose.force(scene, node)

            # Apply velocity impositions
            if dyn.impose and (dyn.impose.what & Impose.Type.VELOCITY):
                dyn.impose.velocity(scene, node)

            # Update clump members
            if isClump:
                ClumpData.applyToMembers(node, self.reset)

        # Store maximum velocity
        self.maxVelocitySq = maxVSq
