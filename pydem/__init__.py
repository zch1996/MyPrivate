"""
PyDEM - Python Discrete Element Method Framework
"""

__version__ = "0.1.0"

import os
import logging
import importlib
import inspect
import sys
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PyDEM")

# Import core modules
from pydem.src.demmath import Vector3r, Matrix3r, Quaternionr
from pydem.src.Scene import Scene
from pydem.src.DEMField import DEMField
from pydem.src.Omega import Omega

# Import shape classes for convenience
from pydem.src.Sphere import Sphere
from pydem.src.Wall import Wall
from pydem.src.Facet import Facet
from pydem.src.Capsule import Capsule
from pydem.src.Cone import Cone
from pydem.src.Ellipsoid import Ellipsoid
from pydem.src.InfCylinder import InfCylinder

# Import material classes
from pydem.src.Material import Material, FrictMat

# Import engines
from pydem.src.Engine import Engine
from pydem.src.ContactLoop import ContactLoop
from pydem.src.DynDt import DynDt
from pydem.src.Leapfrog import ForceResetter, Leapfrog

# Import utis
from pydem.src import utils

# Initialize OpenMP thread count
try:
    from pydem.src.OpenMPSimulator import omp_set_num_threads

    omp_set_num_threads(os.cpu_count() or 4)
except ImportError:
    logger.warning("OpenMP support not available")

# Create global Omega instance
O = Omega.instance()

# Initialize FunctorFactory
from pydem.src.FunctorFactory import FunctorFactory

_factory = FunctorFactory.instance()


# Function to register all functors
def register_all_functors():
    """Register all available functors."""
    logger.info("Registering functors...")

    # 获取 src 目录
    src_dir = os.path.join(os.path.dirname(__file__), "src")

    # 导入所有模块
    for filename in os.listdir(src_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]  # 移除 .py 扩展名
            try:
                logger.debug(f"Importing module: {module_name}")
                module = importlib.import_module(f"pydem.src.{module_name}")

                # 查找所有函数子类
                for name, obj in inspect.getmembers(module):
                    if (
                        inspect.isclass(obj)
                        and hasattr(obj, "registerClass")
                        and obj.__module__ == module.__name__
                    ):
                        try:
                            # 检查是否是函数子类
                            from pydem.src.Functor import Functor

                            if issubclass(obj, Functor) and obj is not Functor:
                                logger.debug(f"Registering functor: {obj.__name__}")
                                obj.registerClass()
                        except (ImportError, TypeError, AttributeError) as e:
                            logger.debug(f"Error registering {name}: {e}")
            except ImportError as e:
                logger.debug(f"Could not import module {module_name}: {e}")

    logger.info("Functor registration complete")


# Register all functors
register_all_functors()

# Try to print functors registered
if True:
    logger.info("Registered functors:")
    for functor_type, functor_class in _factory.boundFunctors.items():
        logger.info(f"  {functor_type.__name__}: {functor_class.__name__}")
    for functor_type, functor_class in _factory.geomFunctors.items():
        logger.info(
            f"  {functor_type[0].__name__} + {functor_type[1].__name__}: {functor_class.__name__}"
        )
    for functor_type, functor_class in _factory.physFunctors.items():
        logger.info(
            f"  {functor_type[0].__name__} + {functor_type[1].__name__}: {functor_class.__name__}"
        )
    for functor_type, functor_class in _factory.lawFunctors.items():
        logger.info(
            f"  {functor_type[0].__name__} + {functor_type[1].__name__} + {functor_type[2]}: {functor_class.__name__}"
        )
    for functor_type, functor_class in _factory.intraFunctors.items():
        logger.info(f"  {functor_type.__name__}: {functor_class.__name__}")


# Export common symbols
__all__ = [
    "Vector3r",
    "Matrix3r",
    "Quaternionr",
    "Scene",
    "DEMField",
    "Omega",
    "O",
    "Sphere",
    "Wall",
    "Facet",
    "Capsule",
    "Cone",
    "Ellipsoid",
    "InfCylinder",
    "Material",
    "FrictMat",
    "Engine",
    "ContactLoop",
    "DynDt",
    "Leapfrog",
    "ForceResetter",
    "utils",
]


def version():
    """Return PyDEM version."""
    return __version__
