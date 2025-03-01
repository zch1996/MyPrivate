#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
from typing import List, Optional, Dict, Any, Set

from .demmath import Vector3r, Real, INF, NAN
from .DEMData import DEMData


class DemFuncs:
    """Utility functions for DEM simulations."""

    @staticmethod
    def pWaveDt(field: "DEMField", noClumps: bool = False) -> Real:
        """
        Calculate critical time step based on P-wave propagation.

        Args:
            field: DEM field
            noClumps: Whether to ignore clumps in the calculation

        Returns:
            Real: Critical time step value
        """
        dt = INF

        for particle in field.getParticles():
            # Skip invalid particles
            if (
                particle is None
                or particle.getMaterial() is None
                or particle.getShape() is None
                or len(particle.getShape().getNodes()) != 1
                or not particle.getShape().getNodes()[0].hasDataTyped(DEMData)
            ):
                continue

            # Get node and its dynamic data
            node = particle.getShape().getNodes()[0]
            dyn = node.getDataTyped(DEMData)

            # Check for elastic material
            elastMat = None
            if hasattr(particle.getMaterial(), "getYoung"):
                elastMat = particle.getMaterial()
            else:
                continue

            # Get radius from shape
            radius = particle.getShape().getEquivalentRadius()
            if radius is None or math.isnan(radius) or radius <= 0:
                continue

            # Handle clumps
            velMult = 1.0
            if dyn.isClumped() and not noClumps:
                raise RuntimeError(
                    "DemFuncs.pWaveDt does not currently work with clumps; pass "
                    "noClumps=True to ignore clumps (and treat them as spheres) "
                    "at your own risk."
                )

            # Calculate critical time step based on P-wave velocity
            p_wave_velocity = math.sqrt(elastMat.getYoung() / elastMat.getDensity())
            particle_dt = radius / (velMult * p_wave_velocity)

            dt = min(dt, particle_dt)

        return dt

    @staticmethod
    def updateClumps(field: "DEMField") -> None:
        """
        Update all clumps in the field.

        Args:
            field: DEM field
        """
        from ClumpData import ClumpData

        # Collect forces and torques on clumps from members
        for node in field.nodes:
            if node is None or not node.hasDataTyped(DEMData):
                continue

            demData = node.getDataTyped(DEMData)

            if demData.isClump() and isinstance(demData, ClumpData):
                demData.force = Vector3r(0.0, 0.0, 0.0)
                demData.torque = Vector3r(0.0, 0.0, 0.0)
                ClumpData.forceTorqueFromMembers(node, demData.force, demData.torque)

        # Apply clump motion to member nodes
        for node in field.nodes:
            if node is None or not node.hasDataTyped(DEMData):
                continue

            demData = node.getDataTyped(DEMData)

            if demData.isClump() and isinstance(demData, ClumpData):
                ClumpData.applyToMembers(node, True)
