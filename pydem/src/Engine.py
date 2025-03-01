#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import datetime
import threading
import inspect
from typing import Dict, List, Optional, Set, Any, Union, Callable
import numpy as np

from .Object import Object
from .DEMLogging import DEM_LOGGER
from .demmath import INF, NAN, Real


class TimingInfo:
    """Stores timing information for an engine."""

    def __init__(self):
        """Initialize timing info."""
        self.nsec = 0  # Nanoseconds
        self.nExec = 0  # Number of executions


class TimingDeltas:
    """Stores timing deltas for engine operations."""

    def __init__(self):
        """Initialize timing deltas."""
        self.deltas = []  # List of (name, delta) pairs


@DEM_LOGGER
class Engine(Object):
    """Base class for simulation engines."""

    def __init__(self):
        """Initialize Engine with default values."""
        super().__init__()
        self.scene = None
        self.timingInfo = TimingInfo()
        self.timingDeltas = TimingDeltas()
        self.field = None
        self.userAssignedField = False
        self.dead = False
        self.label = ""
        self.isNewObject = True

    def selfTest(self):
        pass

    def isActivated(self):
        """Check if engine is active. Default returns True."""
        return True

    def notifyDead(self):
        """Notify that engine is dead. Default does nothing."""
        pass

    def critDt(self):
        """Calculate critical timestep for this engine."""
        return INF

    def run(self):
        """Run the engine. Default implementation raises an error."""
        raise RuntimeError(f"{self.getClassName()} did not override Engine.run()")

    def runHook(self, callerId, hook):
        """Run hook method. Default implementation raises an error."""
        raise RuntimeError("Abstract Engine::runHook called")

    def toString(self):
        """Return string representation of the engine."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = os.environ.get("USER", "unknown")
        return f"<{self.getClassName()} @ {id(self)} [{timestamp} by {user}]>"

    def needsField(self):
        """Check if engine needs a field. Default returns True."""
        return True

    def setField(self):
        """Set field from scene if not already set."""
        if self.userAssignedField or not self.needsField():
            return

        if self.scene is None:
            self.warning(f"No scene assigned to engine {self.getClassName()}")
            return

        field = self.scene.getField()
        if field is None:
            raise RuntimeError("Scene has no field but engine requires one")

        if not self.acceptsField(field):
            raise RuntimeError("Engine does not accept the provided field")

        self.field = field
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.debug(f"Engine {self.getClassName()} assigned field at {timestamp}")
        self.userAssignedField = True

    def acceptsField(self, field):
        """Check if engine accepts the given field. Default returns True."""
        return True

    def getField(self):
        """Get current field."""
        return self.field

    def setField(self, field):
        """Set field manually."""
        if field is None:
            self.setField()
            self.userAssignedField = False
        else:
            self.field = field
            self.userAssignedField = True
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user = os.environ.get("USER", "unknown")
            self.debug(
                f"User assigned field to engine {self.getClassName()} by {user} at {timestamp}"
            )

    def updateScenePtr(self, scene, field):
        """Update scene and field pointers."""
        self.scene = scene
        self.setField(field)

    def runPy(self, callerId, command):
        """Run Python command with scene and engine variables set."""
        if not command:
            return

        try:
            # Create local namespace with useful variables
            local_vars = {
                "scene": self.scene,
                "S": self.scene,
                "engine": self,
                "field": self.field,
                "pydem": __import__("pydem"),
            }

            # Execute the command
            exec(command, globals(), local_vars)
        except Exception as e:
            raise RuntimeError(f"{callerId}: exception in '{command}':\n{str(e)}")


@DEM_LOGGER
class ParallelEngine(Engine):
    """Engine that runs multiple slave engines in parallel."""

    def __init__(self, slaves=None):
        """Initialize with empty slave engines list."""
        super().__init__()
        self.slaves = []  # List of lists of engines

        # Handle initialization with slaves
        if slaves is not None:
            self.setSlaves(slaves)

    def run(self):
        """Run all slave engines in parallel."""
        # openMP warns if the iteration variable is unsigned...
        size = len(self.slaves)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.debug(f"ParallelEngine starting {size} slave groups at {timestamp}")

        # In Python we can't easily parallelize with OpenMP,
        # but we could use multiprocessing or threading
        for i in range(size):
            # run every slave group sequentially
            for engine in self.slaves[i]:
                engine.scene = self.scene
                if engine.getField() is None and engine.needsField():
                    raise RuntimeError(
                        f"Slave engine {engine.getClassName()} has no field but requires one."
                    )

                if not engine.dead and engine.isActivated():
                    self.debug(
                        f"Running slave engine {engine.getClassName()} in group {i}"
                    )
                    engine.run()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.debug(f"ParallelEngine completed at {timestamp}")

    def setField(self):
        """Set field for all slave engines."""
        for group in self.slaves:
            for engine in group:
                engine.scene = self.scene
                engine.setField()

    def setSlaves(self, slaves_list):
        """Set slave engines from a list."""
        self.slaves = []

        for item in slaves_list:
            if isinstance(item, list) or isinstance(item, tuple):
                # Item is a sequence of engines
                self.slaves.append(list(item))
            elif isinstance(item, Engine):
                # Item is a single engine
                self.slaves.append([item])
            else:
                raise TypeError("Slaves must be either engines or sequences of engines")

    def getSlaves(self):
        """Get slave engines as a list."""
        result = []
        for group in self.slaves:
            if len(group) == 1:
                result.append(group[0])
            else:
                result.append(group)
        return result


@DEM_LOGGER
class PeriodicEngine(Engine):
    """
    Engine that runs periodically based on virtual time, real time, or step number.

    This engine will run at specified intervals based on:
    - Virtual time (simulation time)
    - Real time (wall clock time)
    - Step number

    The number of executions can be limited with nDo.
    """

    def __init__(self):
        """Initialize PeriodicEngine with default values."""
        super().__init__()

        # Periodicity criteria
        self.virtPeriod = 0  # Virtual time period (deactivated if <= 0)
        self.realPeriod = 0  # Real time period (deactivated if <= 0)
        self.stepPeriod = 1  # Step period (deactivated if <= 0)
        self.stepModulo = True  # Interpret stepPeriod as modulo

        # Execution control
        self.nDo = -1  # Limit number of executions (deactivated if negative)
        self.nDone = 0  # Number of executions done
        self.initRun = True  # Run the first time we're called

        # Tracking variables
        self.virtLast = NAN  # Last virtual time when run
        self.realLast = time.time()  # Last real time when run
        self.stepLast = -1  # Last step when run
        self.stepPrev = -1  # Previous step when run
        self.virtPrev = -1  # Previous virtual time when run
        self.realPrev = -1  # Previous real time when run

    @staticmethod
    def getClock():
        """Get current wall clock time."""
        return time.time()

    def isActivated(self):
        """
        Check if the engine should be activated based on periodicity criteria.

        Returns:
            bool: True if the engine should run, False otherwise
        """
        # Get current time and step
        virtNow = self.scene.time
        realNow = self.getClock()
        stepNow = self.scene.step

        # First time initialization
        initNow = self.stepLast < 0
        if initNow:
            self.realLast = realNow
            self.virtLast = virtNow
            self.stepLast = stepNow

        # Check if we should run
        should_run = (
            self.nDo < 0 or self.nDone < self.nDo
        ) and (  # Execution limit not reached
            (
                self.virtPeriod > 0 and virtNow - self.virtLast >= self.virtPeriod
            )  # Virtual time period elapsed
            or (
                self.realPeriod > 0 and realNow - self.realLast >= self.realPeriod
            )  # Real time period elapsed
            or (
                self.stepPeriod > 0
                and (  # Step period elapsed
                    (stepNow - self.stepLast >= self.stepPeriod)  # Regular step period
                    or (
                        self.stepModulo and stepNow % self.stepPeriod == 0
                    )  # Step modulo
                )
            )
        )

        # Update tracking variables
        self.realPrev = self.realLast
        self.realLast = realNow
        self.virtPrev = self.virtLast
        self.virtLast = virtNow
        self.stepPrev = self.stepLast
        self.stepLast = stepNow

        # Skip initial run if not desired
        if initNow and not self.initRun:
            return False

        # Increment execution counter if we're going to run
        if should_run:
            self.nDone += 1

        return should_run

    def fakeRun(self):
        """
        Update tracking variables as if the engine ran, without actually running it.
        This doesn't modify nDo/nDone.
        """
        virtNow = self.scene.time
        realNow = self.getClock()
        stepNow = self.scene.step

        self.realPrev = self.realLast
        self.realLast = realNow
        self.virtPrev = self.virtLast
        self.virtLast = virtNow
        self.stepPrev = self.stepLast
        self.stepLast = stepNow


@DEM_LOGGER
class PyRunner(PeriodicEngine):
    """
    Execute a Python command periodically.

    This engine runs a Python command string at intervals specified by
    the periodicity settings inherited from PeriodicEngine.
    """

    def __init__(self, command="", stepPeriod=None):
        """
        Initialize PyRunner.

        Args:
            command: Python command to execute
            stepPeriod: Step period (if provided)
        """
        super().__init__()
        self.command = command

        # Set step period if provided
        if stepPeriod is not None:
            self.stepPeriod = stepPeriod

    def run(self):
        """Run the Python command."""
        if self.command:
            self.runPy("PyRunner", self.command)

    def needsField(self):
        """PyRunner doesn't need a field."""
        return False
