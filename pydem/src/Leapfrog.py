#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import weakref

from pydem.src.Engine import Engine
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr, AngleAxisr, Real, NAN
from pydem.src.DEMData import DEMData
from pydem.src.Impose import Impose
from pydem.src.ClumpData import ClumpData


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
class Leapfrog(Engine):
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
            if force[i] * vel[i] > 0:
                force[i] *= 1 - self.damping
        return force

    def nonviscDamp2nd(self, dt, force, vel, accel):
        """Apply 2nd order numerical damping."""
        for i in range(3):
            if force[i] * (vel[i] + 0.5 * dt * accel[i]) > 0:
                accel[i] *= 1 - self.damping
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
        # self.scene.addEnergy(
        #     transDiss + rotDiss,
        #     "nonviscDamp",
        #     self.nonviscDampIx,
        #     isIncrement=True,
        #     pos=node.pos,
        # )

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

        # self.scene.addEnergy(gr, "grav", self.gravWorkIx, isIncrement=True, pos=pos)

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
        Etrans = 0.5 * dyn.mass * currFluctVel.dot(currFluctVel)

        # Compute rotational kinetic energy
        if dyn.isAspherical():
            mI = np.diag(dyn.inertia)
            T = node.ori.toRotationMatrix()
            Erot = 0.5 * currFluctAngVel.dot(
                T.transpose().dot(mI.dot(T.dot(currFluctAngVel)))
            )
        else:
            Erot = 0.5 * currFluctAngVel.dot(dyn.inertia * currFluctAngVel)

        # Add to energy tracker
        # if self.kinSplit:
        #     self.scene.addEnergy(
        #         Etrans, "kinTrans", self.kinEnergyTransIx, pos=node.pos
        #     )
        #     self.scene.addEnergy(Erot, "kinRot", self.kinEnergyRotIx, pos=node.pos)
        # else:
        #     self.scene.addEnergy(
        #         Etrans + Erot, "kinetic", self.kinEnergyIx, pos=node.pos
        #     )

    def applyPeriodicCorrections(self, node, linAccel):
        """Apply periodic boundary corrections."""
        dyn = node.getDataTyped(DEMData)

        if self.homoDeform == 1 or self.homoDeform == 3:  # HOMO_VEL or HOMO_VEL_2ND
            # Update velocity reflecting changes in macroscopic velocity field
            if self.homoDeform == 3:  # HOMO_VEL_2ND
                dyn.vel += (
                    self.midGradV.dot(dyn.vel - (self.dt / 2.0) * linAccel) * self.dt
                )

            # Reflect macroscopic acceleration in velocity
            dyn.vel += self.dGradV.dot(node.pos)

        elif self.homoDeform == 2:  # HOMO_POS
            node.pos += self.scene.cell.nextGradV.dot(node.pos) * self.dt

    def leapfrogTranslate(self, node):
        """Update position using current velocity."""
        dyn = node.getDataTyped(DEMData)
        node.pos += np.array(dyn.vel) * self.dt

    def leapfrogSphericalRotate(self, node):
        """Update orientation for spherical particles."""
        dyn = node.getDataTyped(DEMData)
        axis = dyn.angVel

        if (axis != Vector3r(0, 0, 0)).any():
            angle = np.linalg.norm(axis)
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
            angMom = np.diag(dyn.inertia).dot(ori.conjugate().dot(dyn.angVel))
            dyn.angMom = angMom

        # Rotation matrix from global to local reference frame
        A = ori.conjugate().toRotationMatrix()

        # Global angular momentum at time n
        l_n = angMom + self.dt / 2 * M

        # Local angular momentum at time n
        l_b_n = A.dot(l_n)

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
        l_b_half = A.dot(angMom)

        # Local angular velocity at time n+1/2
        angVel_b_half = l_b_half / inertia

        # dQ/dt at time n+1/2
        dotQ_half = self.DotQ(angVel_b_half, Q_half)

        # Q at time n+1
        ori += self.dt * dotQ_half

        # Global angular velocity at time n+1/2
        angVel = ori.dot(angVel_b_half)
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
        # Get scene and field
        scene = self.scene
        dem = self.field

        # Update time step and cell information
        self.dt = scene.dt
        self.homoDeform = -1 if not scene.isPeriodic else scene.cell.homoDeform

        # Initialize energy trackers if needed
        # if self.nonviscDampIx < 0 and self.damping > 0:
        #     self.nonviscDampIx = scene.energyRegister("nonviscDamp", self.damping > 0)

        # if self.gravWorkIx < 0 and (dem.gravity != Vector3r(0, 0, 0)).any():
        #     self.gravWorkIx = scene.energyRegister("grav", True)

        # if self.kinEnergyIx < 0:
        #     self.kinEnergyIx = scene.energyRegister("kinetic", True)

        # if self.kinSplit and self.kinEnergyTransIx < 0:
        #     self.kinEnergyTransIx = scene.energyRegister("kinTrans", True)
        #     self.kinEnergyRotIx = scene.energyRegister("kinRot", True)

        # Update velocity gradient
        if scene.isPeriodic:
            self.dGradV = scene.cell.nextGradV - scene.cell.gradV
            self.midGradV = 0.5 * (scene.cell.gradV + scene.cell.nextGradV)

        # Process all nodes
        maxVSq = 0.0
        hasGravity = (dem.gravity != Vector3r(0, 0, 0)).any()

        for node in dem.nodes:
            dyn = node.getDataTyped(DEMData)

            if dyn.isClumped():
                continue

            isClump = dyn.isClump()

            # if isClump, collect force andd torque from clump members
            # (!dyn.isBlockedAll() ||
            # (dyn.impose && (dyn.impose->what & Impose::READ_FORCE))))
            if isClump and not (
                dyn.isBlockedAll()
                or (dyn.impose and (dyn.impose.what & Impose.Type.FORCE))
            ):
                ClumpData.forceTorqueFromMembers(node, dyn.force, dyn.torque)

            # Track maximum velocity
            vSq = np.dot(dyn.vel, dyn.vel)
            maxVSq = max(maxVSq, vSq)

            # Store previous velocity for energy calculations
            pprevVel = dyn.vel
            pprevAngVel = dyn.angVel

            # Compute accelerations
            linAccel = self.computeAccel(dyn.force, dyn.mass, dyn)
            angAccel = Vector3r(0, 0, 0)

            print("Lin Accel: ", linAccel)
            print("Ang Accel: ", angAccel)

            # Apply damping
            if self.damping > 0:
                self.doDampingDissipation(node)

                if dyn.isAspherical():
                    self.nonviscDamp1st(dyn.force, dyn.vel)
                    self.nonviscDamp1st(dyn.torque, dyn.angVel)
                else:
                    linAccel = self.nonviscDamp2nd(
                        self.dt, dyn.force, dyn.vel, linAccel
                    )
                    angAccel = self.computeAngAccel(dyn.torque, dyn.inertia, dyn)
                    angAccel = self.nonviscDamp2nd(
                        self.dt, dyn.torque, dyn.angVel, angAccel
                    )
                    print("Lin Accel: ", linAccel)
                    print("Ang Accel: ", angAccel)
            else:
                if not dyn.isAspherical():
                    angAccel = self.computeAngAccel(dyn.torque, dyn.inertia, dyn)

            # Track gravity work
            if hasGravity and self.gravWorkIx >= 0:
                self.doGravityWork(dyn, dem, node.pos)

            # Apply periodic corrections
            if scene.isPeriodic:
                self.applyPeriodicCorrections(node, linAccel)

            # Update position
            print("Node Pos: ", node.pos)
            self.leapfrogTranslate(node)
            print("Node Pos: ", node.pos)

            # Update orientation
            if dyn.isAspherical():
                self.leapfrogAsphericalRotate(node, dyn.torque)
            else:
                dyn.angVel += angAccel * self.dt
                self.leapfrogSphericalRotate(node)

            # Update velocity
            dyn.vel += linAccel * self.dt

            # Track kinetic energy
            if self.kinEnergyIx >= 0 or (self.kinSplit and self.kinEnergyTransIx >= 0):
                self.doKineticEnergy(node, pprevVel, pprevAngVel, linAccel, angAccel)

            # Reset forces if requested
            if self.reset:
                if hasGravity and not dyn.isGravitySkip():
                    dyn.force = dem.gravity * dyn.mass
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
