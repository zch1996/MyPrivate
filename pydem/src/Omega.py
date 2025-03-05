#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import datetime
import threading
import weakref
import numpy as np
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Type

from pydem.src.Object import Object
from pydem.src.Scene import Scene
from pydem.src.DEMLogging import DEM_LOGGER


@DEM_LOGGER
class Omega(Object):
    """
    Singleton class managing global simulation state.

    This is the Python equivalent of the MoDEM Master class, providing
    centralized access to the simulation scene and other global resources.
    """

    # Singleton instance
    _instance = None

    @classmethod
    def instance(cls):
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = Omega()
        return cls._instance

    def __init__(self):
        """Initialize Omega with default values."""
        super().__init__()

        # Core state
        self.startupLocalTime = datetime.datetime.now()
        self.scene = None  # Current scene
        self.tmpFileCounter = 0
        self.tmpFileDir = self._getTempDir()
        self.tmpFilePrefix = f"{self.tmpFileDir}/~PyDEM-tmp_p{os.getpid()}_"

        # Mutex for thread safety
        self.renderMutex = threading.RLock()

        # Create initial scene
        self.scene = Scene()

        # Configuration
        self.confDir = self._getConfigDir()

        # self.debug(f"Omega initialized at {self.startupLocalTime}")

    def _getTempDir(self) -> str:
        """Get temporary directory path."""
        if "PYDEM_TEMP" in os.environ:
            tmpDir = os.environ["PYDEM_TEMP"]
            self.debug(f"Using PYDEM_TEMP directory: {tmpDir}")
        else:
            import tempfile

            tmpDir = tempfile.gettempdir()
            self.debug(f"Using system temp directory: {tmpDir}")

        if not os.path.exists(tmpDir):
            raise RuntimeError(
                f"Temp directory {tmpDir} does not exist "
                f"({'specified by PYDEM_TEMP' if 'PYDEM_TEMP' in os.environ else 'system temp dir'})"
            )

        return tmpDir

    def _getConfigDir(self) -> str:
        """Get configuration directory path."""
        if "PYDEM_CONF" in os.environ:
            confDir = os.environ["PYDEM_CONF"]
        else:
            confDir = os.path.expanduser("~/pydem_conf")
            if not os.path.exists(confDir):
                try:
                    os.makedirs(confDir)
                except Exception as e:
                    self.warning(f"Could not create config directory {confDir}: {e}")
                    confDir = self.tmpFileDir

        return confDir

    def getRealTime(self) -> float:
        """Return clock time the simulation has been running."""
        return (datetime.datetime.now() - self.startupLocalTime).total_seconds()

    def getRealTime_duration(self) -> datetime.timedelta:
        """Return time duration since startup."""
        return datetime.datetime.now() - self.startupLocalTime

    def getScene(self) -> Scene:
        """Get the current scene."""
        return self.scene

    def setScene(self, scene: Scene) -> None:
        """Set the current scene."""
        if scene is None:
            raise RuntimeError("Scene must not be None.")

        with self.renderMutex:
            self.scene = scene
            self.debug(f"Set new scene: {scene.toString()}")

    def waitForScenes(self) -> None:
        """
        Wait for master scene to finish, including the possibility of it being replaced.

        This is different from Scene.wait() which will return when that particular
        scene object will have stopped.
        """
        while True:
            # Copy the shared_ptr
            current_scene = self.getScene()
            # Wait for that one to finish
            current_scene.wait()
            # If the scene finished and it is still the master scene, we're done
            if current_scene is self.getScene():
                return
            # Otherwise keep waiting for the new master scene

    def tmpFilename(self) -> str:
        """Return unique temporary filename."""
        assert self.tmpFilePrefix
        filename = f"{self.tmpFilePrefix}{self.tmpFileCounter}"
        self.tmpFileCounter += 1
        return filename

    def getTmpFileDir(self) -> str:
        """Get temporary file directory."""
        return self.tmpFileDir

    def setTmpFileDir(self, directory: str) -> None:
        """Set temporary file directory."""
        if not os.path.exists(directory):
            raise RuntimeError(f"Directory {directory} does not exist")

        self.tmpFileDir = directory
        self.tmpFilePrefix = f"{self.tmpFileDir}/~PyDEM-tmp_p{os.getpid()}_"
        self.debug(f"Set temporary file directory to {directory}")

    def deepcopy(self, obj: Object) -> Object:
        """
        Create a deep copy of an object.

        Args:
            obj: Object to copy

        Returns:
            Deep copy of the object
        """
        import copy

        return copy.deepcopy(obj)

    def reset(self) -> None:
        """Reset to empty main scene."""
        self.getScene().stop()
        self.setScene(Scene())
        self.debug("Reset to empty scene")

    def exitNoBacktrace(self, status: int = 0) -> None:
        """
        Exit program without backtrace.

        Args:
            status: Exit status code
        """
        sys.exit(status)

    def initVisualization(self, use_ui=True):
        """
        Initialize visualization system.

        Args:
            use_ui: Whether to start the UI (True) or just initialize the renderer (False)

        Returns:
            The renderer or window instance if successful, None otherwise
        """
        from pydem.visualization import initialize_visualization, start_control_ui

        if use_ui:
            return start_control_ui(self)
        else:
            return initialize_visualization(self)

    ##############################################
