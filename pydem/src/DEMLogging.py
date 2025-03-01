#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import functools
import inspect
import os
import threading

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DEMLogger:
    """
    Logger implementation for DEM system.
    This provides similar functionality to the C++ macros DEM_DECL_LOGGER and DEM_IMPL_LOGGER.
    """

    _loggers = {}
    _lock = threading.RLock()

    @staticmethod
    def getLogger(class_name):
        """Get a logger instance for the given class name."""
        with DEMLogger._lock:
            if class_name not in DEMLogger._loggers:
                DEMLogger._loggers[class_name] = logging.getLogger(f"DEM.{class_name}")
            return DEMLogger._loggers[class_name]

    @staticmethod
    def debug(msg, *args, **kwargs):
        """Log a debug message."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).debug(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        """Log an info message."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        """Log a warning message."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).warning(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        """Log an error message."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).error(msg, *args, **kwargs)

    @staticmethod
    def critical(msg, *args, **kwargs):
        """Log a critical message."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).critical(msg, *args, **kwargs)

    @staticmethod
    def fatal(msg, *args, **kwargs):
        """Log a fatal message and abort."""
        frame = inspect.currentframe().f_back
        class_name = frame.f_locals.get("self", None).__class__.__name__
        DEMLogger.getLogger(class_name).critical(f"FATAL: {msg}", *args, **kwargs)
        # In C++ this would abort(), in Python we'll raise a SystemExit
        raise SystemExit(1)


# Define decorator to add logger to classes (equivalent to DEM_DECL_LOGGER)
def DEM_LOGGER(cls):
    """Decorator to add logger functionality to a class."""
    cls.logger = DEMLogger.getLogger(cls.__name__)

    # Add logging methods to the class
    cls.debug = lambda self, msg, *args, **kwargs: cls.logger.debug(
        msg, *args, **kwargs
    )
    cls.info = lambda self, msg, *args, **kwargs: cls.logger.info(msg, *args, **kwargs)
    cls.warning = lambda self, msg, *args, **kwargs: cls.logger.warning(
        msg, *args, **kwargs
    )
    cls.error = lambda self, msg, *args, **kwargs: cls.logger.error(
        msg, *args, **kwargs
    )
    cls.critical = lambda self, msg, *args, **kwargs: cls.logger.critical(
        msg, *args, **kwargs
    )
    cls.fatal = lambda self, msg, *args, **kwargs: DEMLogger.fatal(msg, *args, **kwargs)
    cls.trace = lambda self, msg, *args, **kwargs: cls.logger.debug(
        f"TRACE: {msg}", *args, **kwargs
    )

    return cls


# Define a function to implement the logger (equivalent to DEM_IMPL_LOGGER)
def DEM_IMPL_LOGGER(cls_name):
    """Implement logger for a class."""
    return DEMLogger.getLogger(cls_name)
