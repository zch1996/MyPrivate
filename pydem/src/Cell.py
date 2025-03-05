import numpy as np
import enum


from datetime import datetime
from pydem.src.utils import CompUtils
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import (
    Vector3r,
    Vector3i,
    levi_civita,
    Real,
    AlignedBox3r,
)


class DeformationMode(enum.Enum):
    """Homogeneous deformation modes"""

    HOMO_NONE = 0  # No homothetic deformation
    HOMO_POS = 1  # Position only
    HOMO_VEL = 2  # Position & velocity, 1st order
    HOMO_VEL_2ND = 3  # Position & velocity, 2nd order
    HOMO_GRADV2 = 4  # Leapfrog-consistent


@DEM_LOGGER
class Cell:
    """
    Periodic cell parameters and routines.
    The Cell represents a parallelepiped with deformation capabilities.
    """

    def __init__(self):
        """Constructor"""
        # self.info(
        #     f"Creating Cell at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Python port"
        # )

        # Core state variables
        self.trsf = np.eye(3, dtype=Real)  # Deformation gradient tensor
        self.refHSize = np.eye(3, dtype=Real)  # Reference cell configuration
        self.hSize = np.eye(3, dtype=Real)  # Current cell configuration
        self.pprevHsize = np.eye(3, dtype=Real)  # Previous half-step configuration
        self.W = np.zeros((3, 3), dtype=Real)  # Spin tensor
        self.spinVec = Vector3r(0, 0, 0)  # Angular velocity vector

        # Internal state
        self._invTrsf = np.eye(3, dtype=Real)
        self._trsfInc = np.zeros((3, 3), dtype=Real)
        self._size = np.ones(3, dtype=Real)
        self._cos = np.ones(3, dtype=Real)
        self._hasShear = False
        self._shearTrsf = np.eye(3, dtype=Real)
        self._unshearTrsf = np.eye(3, dtype=Real)

        # Configuration
        self.homoDeform = DeformationMode.HOMO_GRADV2
        self.trsfUpperTriangular = False

        # Gradient velocity
        self.gradV = np.zeros((3, 3), dtype=Real)
        self.nextGradV = np.zeros((3, 3), dtype=Real)

        # Initialize derived quantities
        self.updateDerivedQuantities()

    def getSize(self):
        """Get current cell size"""
        return self._size

    def setSize(self, s):
        """Set cell size"""
        for k in range(3):
            norm = np.linalg.norm(self.hSize[:, k])
            if norm > 0:
                self.hSize[:, k] *= s[k] / norm

        self.refHSize = self.hSize.copy()
        self.integrateAndUpdate(0)

    def getCos(self):
        """Get cosines between cell axes"""
        return self._cos

    def hasShear(self):
        """Check if cell has shear components"""
        return self._hasShear

    def getHSize(self):
        """Get current cell shape matrix"""
        return self.hSize

    def setHSize(self, m):
        """Set cell shape matrix"""
        self.hSize = m.copy()
        self.refHSize = m.copy()
        self.pprevHsize = m.copy()
        self.integrateAndUpdate(0)

    def getTrsf(self):
        """Get deformation gradient tensor"""
        return self.trsf

    def setTrsf(self, m):
        """Set deformation gradient tensor"""
        self.trsf = m.copy()
        self.integrateAndUpdate(0)

    def getHSize0(self):
        """Get reference cell shape"""
        return self._invTrsf @ self.hSize

    def getSize0(self):
        """Get reference cell size"""
        h0 = self.getHSize0()
        return Vector3r(
            np.linalg.norm(h0[:, 0]), np.linalg.norm(h0[:, 1]), np.linalg.norm(h0[:, 2])
        )

    def setBox(self, size, s1=None, s2=None):
        """Set cell as a box with given dimensions"""
        if s1 is not None and s2 is not None:
            # Called with three scalars
            size = Vector3r(size, s1, s2)

        # Create diagonal matrix from size
        m = np.zeros((3, 3), dtype=Real)
        np.fill_diagonal(m, size)

        self.setHSize(m)
        self.trsf = np.eye(3, dtype=Real)
        self.integrateAndUpdate(0)

    def getVolume(self):
        """Get current cell volume"""
        return np.linalg.det(self.hSize)

    def canonicalizePt(self, pt, period=None, return_period=False):
        """
        Transform point to canonical cell coordinates
        Optionally computes periodicity vector
        """
        unsheared = self.unshearPt(pt)

        if period is None or not return_period:
            wrapped = self.wrapPt(unsheared)
            sheared = self.shearPt(wrapped)

            if return_period:
                new_period = Vector3i(0, 0, 0)
                for i in range(3):
                    x_div_size = unsheared[i] / self._size[i]
                    new_period[i] = int(np.floor(x_div_size))
                return sheared, new_period
            return sheared
        else:
            wrapped, new_period = self.wrapPt(unsheared, period, return_period=True)
            sheared = self.shearPt(wrapped)

            if return_period:
                return sheared, new_period
            else:
                for i in range(3):
                    period[i] = new_period[i]
                return sheared

    def unshearPt(self, pt):
        """Remove shear from point coordinates"""
        return self._unshearTrsf @ pt

    def shearPt(self, pt):
        """Add shear to point coordinates"""
        return self._shearTrsf @ pt

    def wrapPt(self, pt, period=None, return_period=False):
        """
        Wrap point to primary cell
        Optionally computes periodicity vector
        """
        ret = np.zeros(3, dtype=Real)

        if period is None or not return_period:
            for i in range(3):
                ret[i] = CompUtils.wrapNum(pt[i], self._size[i])
            if return_period:
                new_period = Vector3i(0, 0, 0)
                for i in range(3):
                    x_div_size = pt[i] / self._size[i]
                    new_period[i] = int(np.floor(x_div_size))
                return ret, new_period
            return ret
        else:
            new_period = Vector3i(0, 0, 0)
            for i in range(3):
                x_div_size = pt[i] / self._size[i]
                new_period[i] = int(np.floor(x_div_size))
                ret[i] = (x_div_size - np.floor(x_div_size)) * self._size[i]

            if return_period:
                return ret, new_period
            else:
                for i in range(3):
                    period[i] = new_period[i]
                return ret

    def isCanonical(self, pt):
        """Check if point is in canonical cell coordinates"""
        box = AlignedBox3r(Vector3r(0, 0, 0), self.getSize())
        return box.contains(self.unshearPt(pt))

    def shearAlignedExtents(self, perpExtent):
        """Compute extents aligned with shear directions"""
        if not self.hasShear():
            return perpExtent.copy()

        ret = perpExtent.copy()
        cos = self.getCos()

        for ax in range(3):
            ax1 = (ax + 1) % 3
            ax2 = (ax + 2) % 3
            ret[ax1] += 0.5 * perpExtent[ax1] * (1 / cos[ax] - 1)
            ret[ax2] += 0.5 * perpExtent[ax2] * (1 / cos[ax] - 1)

        return ret

    def getGradV(self):
        """Get velocity gradient tensor"""
        return self.gradV

    def setGradV(self, v):
        """Set velocity gradient tensor (not directly allowed)"""
        raise ValueError("gradV is not directly settable, use setCurrGradV() instead.")

    def setCurrGradV(self, v):
        """Set current velocity gradient tensor"""
        self.nextGradV = v.copy()
        self.gradV = v.copy()
        self.updateDerivedQuantities()

    def getShearTrsf(self):
        """Get shear transformation matrix"""
        return self._shearTrsf

    def getUnshearTrsf(self):
        """Get unshear transformation matrix"""
        return self._unshearTrsf

    def intrShiftPos(self, cellDist):
        """Compute position shift due to cell periodicity"""
        return self.hSize @ cellDist.astype(Real)

    def intrShiftVel(self, cellDist):
        """Compute velocity shift due to cell periodicity"""
        if self.homoDeform in [DeformationMode.HOMO_VEL, DeformationMode.HOMO_VEL_2ND]:
            return self.gradV @ self.hSize @ cellDist.astype(Real)
        elif self.homoDeform == DeformationMode.HOMO_GRADV2:
            return self.gradV @ self.pprevHsize @ cellDist.astype(Real)
        else:
            return Vector3r(0, 0, 0)

    def pprevFluctVel(self, currPos, pprevVel, dt):
        """Compute fluctuating velocity"""
        if self.homoDeform in [DeformationMode.HOMO_NONE, DeformationMode.HOMO_POS]:
            return pprevVel
        elif self.homoDeform in [
            DeformationMode.HOMO_VEL,
            DeformationMode.HOMO_VEL_2ND,
        ]:
            return pprevVel - self.gradV @ currPos
        elif self.homoDeform == DeformationMode.HOMO_GRADV2:
            return pprevVel - self.gradV @ (currPos - dt / 2 * pprevVel)
        else:
            self.error("Invalid homoDeform value")
            return Vector3r(0, 0, 0)

    def pprevFluctAngVel(self, pprevAngVel):
        """Compute fluctuating angular velocity"""
        if self.homoDeform == DeformationMode.HOMO_GRADV2:
            return pprevAngVel - self.spinVec
        else:
            return pprevAngVel

    @staticmethod
    def spin2angVel(W):
        """Convert spin tensor to angular velocity vector"""
        return 0.5 * levi_civita(W)

    def setNextGradV(self):
        """Update velocity gradient tensor for next step"""
        self.gradV = self.nextGradV.copy()
        self.W = 0.5 * (self.gradV - self.gradV.T)
        self.spinVec = 0.5 * levi_civita(self.W)

    def checkTrsfUpperTriangular(self):
        """Check if transformation matrix is upper triangular"""
        if self.trsfUpperTriangular and (
            self.trsf[1, 0] != 0.0 or self.trsf[2, 0] != 0.0 or self.trsf[2, 1] != 0.0
        ):
            raise RuntimeError("Cell.trsf must be upper-triangular")

    def integrateAndUpdate(self, dt):
        """Integrate cell deformation over time step dt"""
        # Calculate incremental deformation gradient
        self._trsfInc = dt * self.gradV

        # Update total deformation
        self.trsf += self._trsfInc @ self.trsf
        self._invTrsf = np.linalg.inv(self.trsf)

        if self.trsfUpperTriangular:
            self.checkTrsfUpperTriangular()

        # Update shape
        prevHsize = self.hSize.copy()
        if self.homoDeform == DeformationMode.HOMO_GRADV2:
            temp = np.eye(3, dtype=Real) - self.gradV * dt / 2.0
            self.hSize = (
                np.linalg.inv(temp)
                @ (np.eye(3, dtype=Real) + self.gradV * dt / 2.0)
                @ self.hSize
            )
        else:
            self.hSize += self._trsfInc @ self.hSize

        self.pprevHsize = 0.5 * (prevHsize + self.hSize)

        # Check for degenerate cell
        if abs(np.linalg.det(self.hSize)) < 1e-10:
            raise RuntimeError("Cell is degenerate (zero volume)")

        self.updateDerivedQuantities()

    def updateDerivedQuantities(self):
        """Update derived quantities after cell changes"""
        # Update size and normalized basis vectors
        Hnorm = np.zeros((3, 3), dtype=Real)
        for i in range(3):
            base = self.hSize[:, i].copy()
            self._size[i] = np.linalg.norm(base)
            if self._size[i] > 0:
                base /= self._size[i]
                Hnorm[:, i] = base
            else:
                Hnorm[:, i] = np.zeros(3, dtype=Real)
                Hnorm[i, i] = 1.0

        # Update skew cosines
        for i in range(3):
            i1 = (i + 1) % 3
            i2 = (i + 2) % 3
            cross = np.cross(Hnorm[:, i1], Hnorm[:, i2])
            self._cos[i] = np.dot(cross, cross)

        # Update shear transformation
        self._shearTrsf = Hnorm
        self._unshearTrsf = np.linalg.inv(self._shearTrsf)

        # Update shear flag
        self._hasShear = (
            abs(self.hSize[0, 1]) > 1e-10
            or abs(self.hSize[0, 2]) > 1e-10
            or abs(self.hSize[1, 0]) > 1e-10
            or abs(self.hSize[1, 2]) > 1e-10
            or abs(self.hSize[2, 0]) > 1e-10
            or abs(self.hSize[2, 1]) > 1e-10
        )

        # Update W and spinVec
        self.W = 0.5 * (self.gradV - self.gradV.T)
        self.spinVec = 0.5 * levi_civita(self.W)
