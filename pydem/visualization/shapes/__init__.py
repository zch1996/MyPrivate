"""Shape renderers for PyDEM visualization."""

from .sphere import SphereRenderer
from .facet import FacetRenderer
from .wall import WallRenderer
from .capsule import CapsuleRenderer
from .ellipsoid import EllipsoidRenderer
from .infcylinder import InfCylinderRenderer
from .cone import ConeRenderer


def get_shape_renderer(shape_type):
    """Get renderer for a specific shape type.

    Args:
        shape_type (str): Name of the shape class

    Returns:
        Renderer class for the shape type or None if not available
    """
    renderers = {
        "Sphere": SphereRenderer,
        "Facet": FacetRenderer,
        "Wall": WallRenderer,
        "Capsule": CapsuleRenderer,
        "Ellipsoid": EllipsoidRenderer,
        "InfCylinder": InfCylinderRenderer,
        "Cone": ConeRenderer,
    }

    return renderers.get(shape_type)
