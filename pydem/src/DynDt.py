#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Any

from pydem.src.Engine import PeriodicEngine
from pydem.src.DEMLogging import DEM_LOGGER
from pydem.src.demmath import Vector3r, Matrix3r, Real, INF, NAN
from pydem.src.DEMData import DEMData
from pydem.src.Dispatcher import IntraDispatcher
from pydem.src.Material import FrictMat
from pydem.src.L6Geom import L6Geom
from pydem.src.ContactPhys import FrictPhys


@DEM_LOGGER
class DynDt(PeriodicEngine):
    """
    Adjusts Scene.dt based on current stiffness of particle contacts.

    This engine computes the critical timestep based on the stiffness of
    contacts and adjusts the scene's timestep accordingly.
    """

    def __init__(self):
        """Initialize with default values."""
        super().__init__()

        # Configuration
        self.maxRelInc = 1e-4  # Maximum relative increment of timestep within one step
        self.dryRun = False  # Only compute dt, don't apply it
        self.dt = NAN  # New timestep value (used with dryRun)

        # Internal state
        self.intraForce = None  # Cache for IntraForce dispatcher

    def postLoad(self):
        """Check configuration after loading."""
        if 1.0 + self.maxRelInc == 1.0:
            raise RuntimeError("DynDt: maxRelInc too small (1.0+maxRelInc==1.0)")

    def nodalStiffAdd(self, n, ktrans, krot):
        """
        Add stiffness contributions from contacts to a node.

        Args:
            n: Node to compute stiffness for
            ktrans: Translational stiffness vector (modified in-place)
            krot: Rotational stiffness vector (modified in-place)
        """
        dyn = n.getDataTyped(DEMData)

        # For every particle with this node, traverse its contacts
        for p in dyn.parRef:
            for idC in p.contacts.values():
                C = idC
                if not C.isReal():
                    continue

                assert C.geom is not None and C.phys is not None
                assert isinstance(C.geom, L6Geom)
                assert isinstance(C.phys, FrictPhys)

                ix = C.pIndex(p)
                ph = C.phys
                g = C.geom

                # Contact normal in global coords
                n_vec = C.geom.node.ori.rotate(Vector3r(1, 0, 0))
                n2 = Vector3r(n_vec[0] ** 2, n_vec[1] ** 2, n_vec[2] ** 2)

                # Add translational stiffness
                ktrans += n2 * (ph.kn - ph.kt) + Vector3r(ph.kt, ph.kt, ph.kt)

                # Add rotational stiffness (only due to translation)
                krot += (
                    g.lens[ix] ** 2
                    * ph.kt
                    * Vector3r(n2[1] + n2[2], n2[2] + n2[0], n2[0] + n2[1])
                )

            # Add intra-stiffness for multi-node particles
            if len(p.shape.nodes) > 1 and self.intraForce is not None:
                self.intraForce.addIntraStiffness(p, n, ktrans, krot)

    def nodalCritDtSq(self, n):
        """
        Return square of critical timestep for given node.

        Args:
            n: Node to compute critical timestep for

        Returns:
            Square of critical timestep
        """
        dyn = n.getDataTyped(DEMData)

        # Completely blocked particles are always stable
        if dyn.isBlockedAll():
            return INF

        # Rotational and translational stiffness
        ktrans = Vector3r(0, 0, 0)
        krot = Vector3r(0, 0, 0)

        # Add stiffnesses of contacts of all particles belonging to this node
        self.nodalStiffAdd(n, ktrans, krot)

        # For clump, add stiffnesses of contacts of all particles of all clump nodes
        if dyn.isClumped():
            clump = dyn
            for cn in clump.nodes:
                self.nodalStiffAdd(cn, ktrans, krot)

        ret = INF
        self.trace(
            f"ktrans={ktrans}, krot={krot}, mass={dyn.mass}, inertia={dyn.inertia}"
        )

        # Check translational DOFs
        for i in range(3):
            if ktrans[i] != 0 and dyn.mass > 0 and not dyn.isBlockedAxisDOF(i, False):
                ret = min(ret, dyn.mass / abs(ktrans[i]))

        # Check rotational DOFs
        for i in range(3):
            if (
                krot[i] != 0
                and dyn.inertia[i] > 0
                and not dyn.isBlockedAxisDOF(i, True)
            ):
                ret = min(ret, dyn.inertia[i] / abs(krot[i]))

        return 2 * ret  # (sqrt(2)*sqrt(ret))^2

    def critDt_stiffness(self):
        """
        Compute critical timestep based on stiffness.

        Returns:
            Critical timestep
        """
        # Traverse nodes, find critical timestep for each of them
        ret = INF

        for n in self.field.nodes:
            dt_sq = self.nodalCritDtSq(n)
            ret = min(ret, dt_sq)

            if ret == 0:
                self.error(f"DynDt::nodalCritDtSq returning 0 for node at {n.pos}")

            if np.isnan(ret):
                self.error(f"DynDt::nodalCritDtSq returning nan for node at {n.pos}")

            assert not np.isnan(ret)

        return math.sqrt(ret)

    def critDt_compute(self, scene=None, field=None):
        """
        Compute critical timestep.

        Args:
            scene: Scene object (optional)
            field: Field object (optional)

        Returns:
            Critical timestep
        """
        if scene is not None:
            self.scene = scene
        if field is not None:
            self.field = field

        # Find IntraForce dispatcher if present
        self.intraForce = None
        for e in self.scene.engines:
            if isinstance(e, IntraDispatcher):
                self.intraForce = e
                break

        # Compute timestep from contact stiffnesses
        cdt = self.critDt_stiffness()

        # Clean up
        self.intraForce = None

        return cdt

    def critDt(self):
        """
        Return critical timestep.

        Returns:
            Critical timestep
        """
        return self.critDt_compute()

    def run(self):
        """Run the engine to update timestep."""
        # Apply critical timestep times safety factor
        crDt = self.critDt_compute()

        if np.isinf(crDt):
            if not self.dryRun:
                self.info(
                    f"No timestep computed, keeping the current value {self.scene.dt}"
                )
            return

        # Prevent too fast changes, so cap the value with maxRelInc
        nSteps = self.scene.step - self.stepPrev
        maxNewDt = self.scene.dt * (1.0 + self.maxRelInc) ** nSteps

        self.debug(
            f"dt={self.scene.dt}, crDt={crDt} ({crDt * self.scene.dtSafety} with dtSafety={self.scene.dtSafety}), "
            f"maxNewDt={maxNewDt} with exponent {1.0 + self.maxRelInc}^{nSteps}={(1.0 + self.maxRelInc) ** nSteps}"
        )

        nextDt = min(crDt * self.scene.dtSafety, maxNewDt)

        if not self.dryRun:
            # Don't bother with diagnostics if there is no change
            if nextDt != self.scene.dt:
                self.info(f"Timestep {self.scene.dt} -> {nextDt}")
                self.dt = NAN  # Only meaningful with dryRun
                self.scene.nextDt = nextDt
        else:
            self.dt = nextDt
