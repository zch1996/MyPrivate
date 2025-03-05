#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import time
import datetime
import math
import sys
from typing import Dict, List, Optional, Set, Any
import traceback
from pydem.src.Object import Object
from pydem.src.demmath import Vector3r, Real, NAN, INF
from pydem.src.Cell import Cell
from pydem.src.DEMField import DEMField
from pydem.src.DEMLogging import DEM_LOGGER


class EnergyTracker:
    """Tracks energy in the simulation."""

    class Type:
        """Enum for energy types."""

        KINETIC = 0
        POTENTIAL = 1
        ELASTIC = 2
        PLASTIC = 3
        FRICTION = 4
        DAMPING = 5
        THERMAL = 6
        CUSTOM = 7

    def __init__(self):
        """Initialize energy tracker."""
        self.energies = {}  # Dictionary of {name: value}
        self.resettable = set()  # Set of resettable energy names

    def add(self, type_or_name, value, resettable=False):
        """Add energy value."""
        if isinstance(type_or_name, int):
            # Type as integer
            name = self._getTypeName(type_or_name)
        else:
            # Type as string
            name = type_or_name

        if name not in self.energies:
            self.energies[name] = 0.0

        self.energies[name] += value

        if resettable:
            self.resettable.add(name)

    def get(self, type_or_name):
        """Get energy value."""
        if isinstance(type_or_name, int):
            # Type as integer
            name = self._getTypeName(type_or_name)
        else:
            # Type as string
            name = type_or_name

        return self.energies.get(name, 0.0)

    def resetResettables(self):
        """Reset resettable energies."""
        for name in self.resettable:
            self.energies[name] = 0.0

    def checkBalance(self):
        """Check energy balance."""
        # This is a placeholder - actual implementation would be more complex
        return True

    def _getTypeName(self, type):
        """Convert type enum to name."""
        if type == self.Type.KINETIC:
            return "kinetic"
        elif type == self.Type.POTENTIAL:
            return "potential"
        elif type == self.Type.ELASTIC:
            return "elastic"
        elif type == self.Type.PLASTIC:
            return "plastic"
        elif type == self.Type.FRICTION:
            return "friction"
        elif type == self.Type.DAMPING:
            return "damping"
        elif type == self.Type.THERMAL:
            return "thermal"
        elif type == self.Type.CUSTOM:
            return "custom"
        else:
            return f"unknown_{type}"


@DEM_LOGGER
class Scene(Object):
    """Represents a simulation scene."""

    # Substep constants
    SUBSTEP_INIT = -1
    SUBSTEP_PROLOGUE = 0

    def __init__(self):
        """Initialize Scene with default values."""
        super().__init__()
        self.fields = []  # Lists of fields
        self.engines = []  # List of engines
        self.nextEngines = []  # List of engines
        self.cell = Cell()
        self.energy = EnergyTracker()

        # Simulation parameters
        self.dt = NAN  # Current timestep
        self.nextDt = NAN  # Next timestep
        self.dtSafety = 0.9  # Safety factor for timestep
        self.time = 0.0  # Current simulation time
        self.step = 0  # Current step number
        self.subStep = -1  # Current sub-step (-1: init, 0: prologue)
        self.subStepping = False  # Single engine stepping mode
        self.isPeriodic = False
        self.stopAtStep = 0
        self.stopAtTime = NAN
        self.deterministic = False
        self.trackEnergy = False

        # Timing
        self.clock0 = datetime.datetime.now()  # simulation start time
        self.clock0adjusted = False
        self.preSaveDuration = 0

        # Labels
        self.labels = {}

        # Control flags
        self._runningFlag = False
        self._stopFlag = False

        # Stopping conditions
        self.throttle = 0.0  # Delay between steps

        # Threading
        self.runMutex = threading.RLock()
        self.engineMutex = threading.RLock()
        self.backgroundThreadId = None
        self.lastException = None

        # Timing
        self.startTime = time.time()

        # Create a DEMField object
        self.fields.append(DEMField())

    def __del__(self):
        """Clean up when Scene is deleted."""
        try:
            import sys

            if not hasattr(sys, "modules") or not sys.modules:
                # Python is shutting down, nothing to clean up
                return

            try:
                from .DEMLogging import DEM_LOGGER

                # self.info(f"Scene {self.toString()} is being deleted")
            except (ImportError, AttributeError):
                pass

            # 执行必要的清理操作，但避免导入新模块
            # 例如，释放资源、关闭文件等

            # Clear engines
            if hasattr(self, "engines"):
                self.engines.clear()

            # Clear fields
            if hasattr(self, "fields"):
                self.fields.clear()

        except Exception:
            # Ignore exceptions while cleaning up
            pass

    @property
    def field(self):
        return self.fields[0]

    @field.setter
    def field(self, value):
        self.fields[0] = value

    def integrateCell(self, dt):
        """Integrate cell state."""
        if not self.isPeriodic or self.cell is None:
            return

        try:
            self.cell.integrateAndUpdate(dt)
        except Exception as e:
            print(f"Cell integration failed: {e}")
            raise

    def updateCellGradV(self):
        """Update cell gradient velocity."""
        if not self.isPeriodic or self.cell is None:
            return
        self.cell.setNextGradV()

    def run(self, steps=-1, wait=False, targetTime=NAN):
        """Run simulation for specified number of steps."""
        self.lastException = None
        if self.isRunning():
            raise RuntimeError("Scene is already running")

        with self.runMutex:
            if steps > 0:
                self.stopAtStep = self.step + steps
            if targetTime > 0:
                self.stopAtTime = self.time + targetTime

            self._runningFlag = True
            self._stopFlag = False

            # Start background thread
            thread = threading.Thread(target=self.backgroundLoop)
            self.backgroundThreadId = thread.ident
            thread.daemon = True
            thread.start()

        if wait:
            self.wait()

    def stop(self):
        """Stop simulation."""
        if not self.isRunning():
            return

        with self.runMutex:
            self._stopFlag = True

    def oneStep(self):
        """Run a single simulation step."""
        self.lastException = None
        if self.isRunning():
            raise RuntimeError("Scene is already running")
        self.doOneStep()

    def wait(self, timeout=0):
        """Wait for simulation to complete."""
        if not self.isRunning():
            return

        start = time.time()
        while self.isRunning() or (
            not self.subStepping and self.subStep != self.SUBSTEP_INIT
        ):
            if timeout > 0:
                elapsed = time.time() - start
                if elapsed >= timeout:
                    raise RuntimeError(f"Timeout {timeout}s exceeded.")
            time.sleep(0.04)  # 40ms

        if self.lastException:
            e = self.lastException
            self.lastException = None
            raise e

    def setIsPeriodic(self, periodic):
        """Set whether simulation is periodic."""
        self.isPeriodic = periodic
        if periodic and self.cell is None:
            self.cell = Cell()
            print("Created new Cell")

    def backgroundLoop(self):
        """Background loop for simulation."""
        try:
            # Print only once at the beginning
            print(f"Background thread started with ID: {threading.get_ident()}")

            while True:
                if self.subStepping:
                    print("Sub-stepping disabled")
                    self.subStepping = False

                self.doOneStep()

                # Remove the print statement that was causing issues
                # Only print status occasionally if needed (e.g., every 10 steps)
                if self._runningFlag and self.step % 10 == 0:
                    pass  # We can add minimal logging here if needed

                if self.checkStopConditions():
                    print("Background thread stopping due to stop conditions")
                    break

                if self.throttle > 0:
                    time.sleep(self.throttle)

            print(f"Background thread exiting normally")
        except Exception as e:
            print(f"Exception in background loop: {e}")
            print(traceback.format_exc())
            self.lastException = e
        finally:
            with self.runMutex:
                self._runningFlag = False
                self.subStep = self.SUBSTEP_INIT
            print("Background thread terminated")

    def doOneStep(self):
        """Perform one simulation step."""
        with self.engineMutex:
            # Update timestep if needed
            if math.isnan(self.dt) or math.isinf(self.dt) or self.dt < 0:
                self.dt = self.computeCriticalTimeStep()
                self.dt *= self.dtSafety

            # Update engines if necessary
            self.updateEngines()

            # Main simulation step
            if not self.subStepping and self.subStep == self.SUBSTEP_INIT:
                self.subStep = self.SUBSTEP_PROLOGUE

                # Run self-tests
                self.runSelfTest()

                # Integrate cell before engine execution
                if self.isPeriodic and self.cell:
                    self.integrateCell(self.dt)

                # Track energy
                if self.trackEnergy:
                    self.energy.resetResettables()

                # Run engines
                for engine in self.engines:
                    if engine is None:
                        continue

                    engine.scene = self
                    if engine.getField() is None and engine.needsField():
                        print(f"{engine.getClassName()} has no field but requires one")
                        raise RuntimeError("Engine has no field but requires one")

                    if not engine.dead and engine.isActivated():
                        engine.run()

                # Update cell after engine execution
                if self.isPeriodic and self.cell:
                    self.updateCellGradV()

                # Update simulation state
                self.step += 1
                self.time += self.dt
                self.subStep = self.SUBSTEP_INIT

                # Check energy balance if tracking is enabled
                if self.trackEnergy and self.energy and not self.energy.checkBalance():
                    print(f"Energy imbalance detected at step {self.step}")

                # Update timestep if next one is specified
                if not math.isnan(self.nextDt):
                    self.dt = self.nextDt
                    self.nextDt = NAN

    def computeCriticalTimeStep(self):
        """Compute critical timestep based on engines and field."""
        criticalDt = INF

        print("Length of engines", len(self.engines))

        # Check engines
        for engine in self.engines:
            if engine is None or engine.dead:
                continue

            if engine.getField() is None and engine.needsField():
                if engine.getField() is None:
                    print(f"{engine.getClassName()} has no field but requires one")
                raise RuntimeError("Engine has no field but requires one")

            criticalDt = min(criticalDt, engine.critDt())

        # Check field
        if self.field:
            criticalDt = min(criticalDt, self.field.critDt())

        if math.isinf(criticalDt):
            raise RuntimeError("Failed to obtain meaningful timestep")

        return criticalDt

    def updateEngines(self):
        """Update engines if new ones are queued."""
        if self.nextEngines and (
            self.subStep == self.SUBSTEP_INIT
            or (self.subStep <= self.SUBSTEP_PROLOGUE and not self.subStepping)
        ):
            self.engines = self.nextEngines.copy()
            self.nextEngines.clear()
            self.subStep = self.SUBSTEP_INIT

    def checkStopConditions(self):
        """Check if stop conditions are met."""
        if (self.stopAtStep > 0 and self.step == self.stopAtStep) or (
            self.stopAtTime > 0
            and self.time >= self.stopAtTime
            and self.time < self.stopAtTime + self.dt
        ):
            with self.runMutex:
                self._stopFlag = True
                self._runningFlag = False
                return True
        return self._stopFlag

    def runSelfTest(self):
        """Run self-tests on field and engines."""
        if self.field:
            self.field.scene = self
            self.field.selfTest()

        for engine in self.engines:
            if engine is None:
                continue
            engine.scene = self
            if engine.getField() is None and engine.needsField():
                print(f"{engine.getClassName()} has no field but requires one")
                raise RuntimeError("Engine has no field but requires one")
            engine.selfTest()

    def getDuration(self):
        """Get simulation duration in seconds."""
        return int(time.time() - self.startTime)

    def addEngine(self, engine):
        """Add an engine to the scene."""
        if engine is None:
            return

        if self.field is None:
            raise RuntimeError("Scene has no field")

        if engine.needsField():
            engine.updateScenePtr(self, self.field)
        else:
            # Only update scene in Engine
            print(f"Engine: {engine.getClassName()}")
            engine.scene = self

        print(
            f"Adding engine: {engine.getClassName()} to scene, with field {engine.getField()}"
        )
        self.engines.append(engine)

    def clearEngines(self):
        """Clear all engines."""
        self.engines.clear()
        self.nextEngines.clear()

    def addEnergy(self, type, value):
        """Add energy value by type."""
        if not self.trackEnergy:
            print("Energy tracking is disabled. Enable it using enableEnergyTracking()")
            return

        if self.energy is None:
            self.enableEnergyTracking()

        self.energy.add(type, value)

    def addEnergy(self, name, value, resettable=False):
        """Add energy value by name."""
        if not self.trackEnergy:
            print("Energy tracking is disabled. Enable it using enableEnergyTracking()")
            return

        if self.energy is None:
            self.enableEnergyTracking()

        self.energy.add(name, value, resettable)

    def getEnergy(self, type):
        """Get energy by type."""
        if not self.trackEnergy or self.energy is None:
            return 0.0
        return self.energy.get(type)

    def getEnergy(self, name):
        """Get energy by name."""
        if not self.trackEnergy or self.energy is None:
            return 0.0
        return self.energy.get(name)

    def postLoad(self, scene, attr):
        """Handle post-loading operations."""
        # Adjust simulation start time if needed
        if not self.clock0adjusted:
            self.clock0 = self.clock0 - datetime.timedelta(seconds=self.preSaveDuration)
            self.clock0adjusted = True

        # Assign fields to engines
        for i, e in enumerate(self.engines):
            if e is None:
                raise RuntimeError(f"Scene.engines[{i}]==None (not allowed).")
            e.scene = self
            e.setField()

    def addParticle(self, p):
        """Add a particle to the field."""
        if self.field is None:
            raise RuntimeError("Scene has no field")

        dem = self.field
        if not isinstance(dem, DEMField):
            raise RuntimeError("Field is not a DEMField")

        return dem.getParticles().add(p)

    def isRunning(self):
        """Check if simulation is running."""
        return self._runningFlag

    def enableEnergyTracking(self, enable=True):
        """Enable or disable energy tracking."""
        self.trackEnergy = enable
        if enable and self.energy is None:
            self.energy = EnergyTracker()

    def isEnergyTrackingEnabled(self):
        """Check if energy tracking is enabled."""
        return self.trackEnergy

    def setField(self, field, index=0):
        """Set the field for this scene."""
        self.fields[index] = field

    def getField(self, index=0):
        """Get the field for this scene."""
        return self.fields[index]

    def setCellBox(self, size):
        """Set cell size for periodic simulation."""
        if self.cell is None:
            self.cell = Cell()
        self.cell.setBox(size)
        self.isPeriodic = True

    def getCell(self):
        """Get simulation cell if periodic."""
        return self.cell if self.isPeriodic else None

    def cleanup(self):
        """Explicitly clean up resources."""
        # self.info(f"Cleaning up Scene {self.toString()}")

        # 执行清理操作
        self.engines.clear()
        self.fields.clear()
        # 其他清理...

        return True

    def isBackgroundThread(self):
        """Check if the current thread is the background simulation thread."""
        import threading

        current_thread_id = threading.get_ident()
        return current_thread_id == self.backgroundThreadId

    def isMainThread(self):
        """Check if the current thread is the main thread."""
        import threading

        return threading.current_thread() is threading.main_thread()

    def isBackgroundRunning(self):
        """Check if the background thread is running while in the main thread."""
        if not self.isRunning():
            return False
        return self.isMainThread() and self.backgroundThreadId is not None

    def finalize(self):
        """Properly finalize the scene, ensuring background threads are stopped."""
        print("Finalizing scene...")

        if self.isRunning():
            print("Stopping running simulation...")
            self.stop()
            try:
                self.wait(timeout=5)  # Wait up to 5 seconds for clean shutdown
            except RuntimeError as e:
                print(f"Warning during wait: {e}")

        # Clear resources
        print("Cleaning up resources...")
        self.cleanup()

        # Ensure thread is marked as not running
        with self.runMutex:
            self._runningFlag = False
            self.backgroundThreadId = None

        print("Scene finalized.")
        return True


# Import at the end to avoid circular imports
from pydem.src.DEMField import DEMField
