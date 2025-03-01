#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Functor import LawFunctor
from pydem.src.ContactGeom import ContactGeom
from pydem.src.ContactPhys import ContactPhys, FrictPhys
from pydem.src.L6Geom import L6Geom
from pydem.src.DEMLogging import DEM_LOGGER
import numpy as np
from pydem.src.demmath import Vector3r, Vector2r, Real, NAN
from pydem.src.ContactData import IdealElPlData


@DEM_LOGGER
class Law2_L6Geom_FrictPhys_IdealElPl(LawFunctor):
    """Ideal elastic-plastic law."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        # Parameters
        self.iniEqlb = False  # Initial equilibrium
        self.relRollStiff = 0.0  # Rolling stiffness relative to normal stiffness
        self.relTwistStiff = 0.0  # Twisting stiffness relative to rolling stiffness
        self.rollTanPhi = 0.0  # Rolling friction angle
        self.noSlip = False  # Disable plastic slipping
        self.noBreak = False  # Disable tension breakage
        self.noFrict = False  # Disable friction

        # Energy indices
        self.plastDissipIx = -1  # Plastic dissipation energy index
        self.elastPotIx = -1  # Elastic potential energy index
        self.brokenIx = -1  # Broken contact energy index

    def go(self, geom, phys, contact):
        """
        Apply contact law.

        Args:
            geom: Contact geometry
            phys: Contact physics
            contact: Contact object

        Returns:
            bool: True if contact should be kept, False otherwise
        """
        g = geom.cast(L6Geom)
        ph = phys.cast(FrictPhys)

        # Get normal displacement
        uN = g.uN
        if self.iniEqlb:
            if contact.isFresh(self.scene):
                contact.data = IdealElPlData()
                contact.data.uN0 = uN
            if contact.data:
                uN -= contact.data.uN0

        # Check for tension breakage
        if uN > 0 and not self.noBreak:
            if self.noSlip and self.scene.trackEnergy:
                velT = Vector2r(g.vel[1], g.vel[2])
                prevFt = Vector2r(ph.force[1], ph.force[2])
                Fn = ph.kn * uN
                Ft = prevFt + self.scene.dt * ph.kt * velT

                # Calculate energy
                energy = 0.5 * ((Fn * Fn) / ph.kn + np.dot(Ft, Ft) / ph.kt)
                self.scene.addEnergy("broken", energy)

            return False

        # Reset torque
        ph.torque = Vector3r(0.0, 0.0, 0.0)

        # Normal force
        ph.force[0] = ph.kn * uN

        # Tangential force
        Ft = Vector2r(ph.force[1], ph.force[2])
        if self.noFrict or ph.tanPhi == 0.0:
            Ft = Vector2r(0.0, 0.0)
        else:
            velT = Vector2r(g.vel[1], g.vel[2])
            Ft += self.scene.dt * ph.kt * velT
            maxFt = abs(ph.force[0]) * ph.tanPhi

            if np.dot(Ft, Ft) > maxFt * maxFt and not self.noSlip:
                FtNorm = np.linalg.norm(Ft)
                ratio = maxFt / FtNorm if FtNorm > 0 else 0.0

                if self.scene.trackEnergy:
                    dissip = maxFt * (FtNorm - maxFt) / ph.kt if ph.kt != 0 else 0.0
                    self.scene.addEnergy("plast", dissip)

                Ft *= ratio

        # Update force
        ph.force[1] = Ft[0]
        ph.force[2] = Ft[1]

        # Track elastic energy
        if self.scene.trackEnergy:
            elast = 0.5 * ((ph.force[0] * ph.force[0]) / ph.kn)
            if ph.kt != 0:
                elast += 0.5 * np.dot(Ft, Ft) / ph.kt

            self.scene.addEnergy("elast", elast)

        # Rolling and bending
        if self.relRollStiff > 0.0 and self.rollTanPhi > 0.0:
            charLen = g.lens.sum()
            if charLen <= 0:
                raise RuntimeError("Invalid characteristic length <= 0")

            kr = ph.kn * charLen

            # Twist
            if self.relTwistStiff > 0:
                ph.torque[0] += self.scene.dt * self.relTwistStiff * kr * g.angVel[0]
                maxTt = abs(ph.force[0] * self.rollTanPhi * charLen)
                if abs(ph.torque[0]) > maxTt:
                    ph.torque[0] = np.sign(ph.torque[0]) * maxTt

            # Rolling resistance
            Tr = Vector2r(ph.torque[1], ph.torque[2])
            angVelR = Vector2r(g.angVel[1], g.angVel[2])
            Tr += self.scene.dt * kr * angVelR
            maxTr = max(0.0, abs(ph.force[0]) * self.rollTanPhi * charLen)

            if np.dot(Tr, Tr) > maxTr * maxTr:
                TrNorm = np.linalg.norm(Tr)
                ratio = maxTr / TrNorm if TrNorm > 0 else 0.0
                Tr *= ratio

            # Update torque
            ph.torque[1] = Tr[0]
            ph.torque[2] = Tr[1]

        return True


class Law2_L6Geom_FrictPhys_Linear(LawFunctor):
    """Linear elastic law with 6 DOFs."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.charLen = -1.0  # Characteristic length for stiffness ratios
        self.elastPotIx = -1  # Elastic potential energy index

    def go(self, geom, phys, contact):
        """
        Apply contact law.

        Args:
            geom: Contact geometry
            phys: Contact physics
            contact: Contact object

        Returns:
            bool: True if contact should be kept, False otherwise
        """
        g = geom.cast(L6Geom)
        ph = phys.cast(FrictPhys)
        dt = self.scene.dt

        if self.charLen < 0:
            raise ValueError("Characteristic length must be non-negative")

        # Simple linear increments
        kntt = Vector3r(ph.kn, ph.kt, ph.kt)  # Normal and tangent stiffnesses
        ktbb = kntt / self.charLen  # Twist and bending stiffnesses

        # Element-wise multiplication and addition
        ph.force += dt * g.vel * kntt
        ph.torque += dt * g.angVel * ktbb

        # Compute normal force non-incrementally
        ph.force[0] = ph.kn * g.uN

        # Track energy
        if self.scene.trackEnergy:
            E = 0.5 * (ph.force[0] * ph.force[0]) / ph.kn

            if kntt[1] != 0.0:
                E += (
                    0.5
                    * (ph.force[1] * ph.force[1] + ph.force[2] * ph.force[2])
                    / kntt[1]
                )

            if ktbb[0] != 0.0:
                E += 0.5 * (ph.torque[0] * ph.torque[0]) / ktbb[0]

            if ktbb[1] != 0.0:
                E += (
                    0.5
                    * (ph.torque[1] * ph.torque[1] + ph.torque[2] * ph.torque[2])
                    / ktbb[1]
                )

            self.scene.addEnergy("elast", E)

        return True
