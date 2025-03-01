import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Any
from pydem.src.Node import Node
from pydem.src.demmath import Vector3r, Real, INF
from pydem.src.Material import FrictMat, Material
from pydem.src.Leapfrog import Leapfrog

from pydem.src.DynDt import DynDt
from pydem.src.ContactLoop import ContactLoop
from pydem.src.Particle import Particle

from pydem.src.DEMData import DEMData

from pydem.src.Sphere import Sphere

hasGL = False


class CompUtils:
    """Computational utilities"""

    defaultCmap = 19  # corresponds to 'coolwarm'

    @staticmethod
    def wrapNum(x, size, period=None, return_period=False):
        """Wrap a number to [0, size)"""
        if size <= 0:
            return (x, 0) if return_period else x

        x_div_size = x / size
        floor_x_div_size = np.floor(x_div_size)
        wrapped = (x_div_size - floor_x_div_size) * size

        if return_period:
            return wrapped, int(floor_x_div_size)
        elif period is not None:
            period = int(floor_x_div_size)

        return wrapped

    @staticmethod
    def clamped(x, a, b):
        """Clamp value x between a and b"""
        return a if x < a else (b if x > b else x)

    @staticmethod
    def clamp(x, a, b):
        """Clamp value x between a and b (in-place)"""
        return CompUtils.clamped(x, a, b)

    @staticmethod
    def angleInside(phi, a, b):
        """Test whether phi is inside angle interval <a,b>"""
        if abs(a - b) >= 2 * math.pi:
            return True  # interval covers everything
        if a > b:
            a -= 2 * math.pi
        if a == b:
            return (a % (2 * math.pi)) == (phi % (2 * math.pi))  # corner case
        # wrap phi so that a+pphi is in a..a+2*M_PI, i.e. pphi in 0..2*M_PI
        pphi = CompUtils.wrapNum(phi - a, 2 * math.pi)
        return pphi < (b - a)

    @staticmethod
    def mapColor(normalizedColor, cmap=-1, reversed=False):
        """Map normalized color value to RGB"""
        # This is a simplified version as we don't have the colormaps data
        if cmap == -1:
            cmap = CompUtils.defaultCmap
        normalizedColor = max(0.0, min(normalizedColor, 1.0))
        if reversed:
            normalizedColor = 1 - normalizedColor
        # Simplified implementation - in real code you'd use the colormaps
        return CompUtils.mapColor_map0(normalizedColor)

    @staticmethod
    def mapColor_map0(xnorm):
        """Simple colormap implementation (blue-cyan-green-yellow-red)"""
        if xnorm < 0.25:
            return np.array([0, 4.0 * xnorm, 1])
        if xnorm < 0.5:
            return np.array([0, 1, 1.0 - 4.0 * (xnorm - 0.25)])
        if xnorm < 0.75:
            return np.array([4.0 * (xnorm - 0.5), 1.0, 0])
        return np.array([1, 1 - 4.0 * (xnorm - 0.75), 0])

    @staticmethod
    def scalarOnColorScale(x, xmin, xmax, cmap=-1, reversed=False):
        """Map scalar value to color using a colormap"""
        xnorm = min(1.0, max((x - xmin) / (xmax - xmin), 0.0))
        return CompUtils.mapColor(xnorm, cmap, reversed)

    @staticmethod
    def closestParams_LineLine(P, u, Q, v, parallel=None):
        """Return parameters where lines approach the most"""
        w0 = P - Q
        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        denom = a * c - b * b
        is_parallel = False

        if abs(denom) < 1e-8:
            is_parallel = True
            if parallel is not None:
                parallel = True
            return np.array([0, 0 if b == 0 else (d / b) if b != 0 else (e / c)])

        if parallel is not None:
            parallel = False

        return np.array([(b * e - c * d) / denom, (a * e - b * d) / denom])

    @staticmethod
    def distSq_LineLine(P, u, Q, v, parallel, st):
        """Return squared distance of two lines where they approach the most"""
        st[:] = CompUtils.closestParams_LineLine(P, u, Q, v, parallel)
        return np.sum((P + u * st[0] - (Q + v * st[1])) ** 2)

    @staticmethod
    def closestSegmentPt(P, A, B, normPos=None):
        """Find closest point on segment AB to point P"""
        BA = B - A
        BA_norm_sq = np.sum(BA**2)

        if BA_norm_sq < 1e-10:
            if normPos is not None:
                normPos = 0.0
            return A

        u = (np.dot(P, BA) - np.dot(A, BA)) / BA_norm_sq
        u_clamped = min(1.0, max(0.0, u))

        if normPos is not None:
            normPos = u

        return A + u_clamped * BA

    @staticmethod
    def inscribedCircleCenter(v0, v1, v2):
        """Calculate center of inscribed circle for triangle"""
        norm_v1_v0 = np.linalg.norm(v1 - v0)
        norm_v2_v0 = np.linalg.norm(v2 - v0)
        norm_v0_v2 = np.linalg.norm(v0 - v2)

        return v0 + ((v2 - v0) * norm_v1_v0 + (v1 - v0) * norm_v2_v0) / (
            norm_v1_v0 + np.linalg.norm(v2 - v1) + norm_v0_v2
        )

    @staticmethod
    def circumscribedCircleCenter(A, B, C):
        """Calculate center of circumscribed circle for triangle"""
        a = A - C
        b = B - C

        a_sq_norm = np.sum(a**2)
        b_sq_norm = np.sum(b**2)

        cross_ab = np.cross(a, b)
        cross_ab_sq_norm = np.sum(cross_ab**2)

        if cross_ab_sq_norm < 1e-10:
            # Points are collinear, no unique circumcenter
            return (A + B + C) / 3  # Return centroid as fallback

        return C + np.cross((a_sq_norm * b - b_sq_norm * a), cross_ab) / (
            2 * cross_ab_sq_norm
        )

    @staticmethod
    def segmentPlaneIntersection(A, B, pt, normal):
        """Calculate intersection parameter of segment AB with plane"""
        return np.dot(normal, pt - A) / np.dot(normal, pt - B)

    @staticmethod
    def lineSphereIntersection(A, u, C, r, t0=0, t1=0, relTol=1e-6):
        """Compute intersection of line with sphere"""
        # Move sphere relative to A
        Cr = C - A
        # Calculate discriminant
        disc = np.dot(u, Cr) ** 2 - np.dot(Cr, Cr) + r**2

        if disc < 0:
            return 0  # No intersection

        if 4 * disc < (relTol * r) ** 2:
            t0 = np.dot(u, Cr)
            return 1  # One intersection (tangent)

        disc_sqrt = np.sqrt(disc)
        t0 = np.dot(u, Cr) - disc_sqrt
        t1 = np.dot(u, Cr) + disc_sqrt
        return 2  # Two intersections

    @staticmethod
    def triangleBarycentrics(x, A, B, C):
        """Return barycentric coordinates of point x on triangle ABC"""
        vB = B - A
        vC = C - A
        vX = x - A

        # Create 2x2 matrix for solving the system
        M = np.array(
            [[np.dot(vB, vB), np.dot(vC, vB)], [np.dot(vB, vC), np.dot(vC, vC)]]
        )

        # Solve for u, v
        try:
            uv = np.linalg.solve(M, np.array([np.dot(vX, vB), np.dot(vX, vC)]))
            return np.array([1 - uv[0] - uv[1], uv[0], uv[1]])
        except np.linalg.LinAlgError:
            # Handle degenerate case
            return np.array([1 / 3, 1 / 3, 1 / 3])  # Return centroid coordinates

    @staticmethod
    def cart2cyl(cart):
        """Convert cartesian coordinates to cylindrical"""
        r = np.sqrt(cart[0] ** 2 + cart[1] ** 2)
        theta = np.arctan2(cart[1], cart[0])
        return np.array([r, theta, cart[2]])

    @staticmethod
    def cyl2cart(cyl):
        """Convert cylindrical coordinates to cartesian"""
        return np.array([cyl[0] * np.cos(cyl[1]), cyl[0] * np.sin(cyl[1]), cyl[2]])

    @staticmethod
    def distSq_SegmentSegment(
        center0,
        direction0,
        halfLength0,
        center1,
        direction1,
        halfLength1,
        st=None,
        parallel=None,
    ):
        """Calculate squared distance between two segments"""
        # This is a complex function, implementing a simplified version
        diff = center0 - center1
        extent0 = halfLength0
        extent1 = halfLength1

        a01 = -np.dot(direction0, direction1)
        b0 = np.dot(diff, direction0)
        b1 = -np.dot(diff, direction1)
        c = np.sum(diff**2)
        det = abs(1.0 - a01 * a01)

        if st is None:
            st = np.zeros(2)

        s0, s1 = 0, 0

        if det >= 1e-8:
            # Segments are not parallel
            if parallel is not None:
                parallel = False

            s0 = a01 * b1 - b0
            s1 = a01 * b0 - b1
            extDet0 = extent0 * det
            extDet1 = extent1 * det

            # Simplified implementation - in real code you'd handle all the cases
            # This is just a basic implementation
            if s0 >= -extDet0 and s0 <= extDet0 and s1 >= -extDet1 and s1 <= extDet1:
                invDet = 1.0 / det
                s0 *= invDet
                s1 *= invDet
            else:
                # Clamp to segment boundaries
                s0 = max(-extent0, min(extent0, s0))
                s1 = max(-extent1, min(extent1, s1))

            st[0] = s0
            st[1] = s1

            sqrDist = s0 * (s0 + a01 * s1 + 2 * b0) + s1 * (a01 * s0 + s1 + 2 * b1) + c
        else:
            # Segments are parallel
            if parallel is not None:
                parallel = True

            e0pe1 = extent0 + extent1
            sign = -1.0 if a01 > 0 else 1.0
            b0Avr = 0.5 * (b0 - sign * b1)
            lambda_val = -b0Avr

            lambda_val = max(-e0pe1, min(e0pe1, lambda_val))

            s1 = -sign * lambda_val * extent1 / e0pe1
            s0 = lambda_val + sign * s1

            st[0] = s0
            st[1] = s1

            sqrDist = lambda_val * (lambda_val + 2 * b0Avr) + c

        # Account for numerical round-off errors
        if sqrDist < 0:
            sqrDist = 0.0

        return sqrDist


#######################################################
"""
- ptr_to_string
- find_executable
- spherePWaveDt
- defaultMaterial
- defaultEngines
- _commonBodySetup
- _mkDemNode
- sphere
"""


def ptr_to_string(obj):
    """
    Equivalent to C++ ptr_to_string function.
    Converts a pointer to a string representation.

    Args:
        obj: Object to convert to string

    Returns:
        String representation of object pointer
    """
    return f"0x{id(obj):x}"


def find_executable(executable):
    """
    Equivalent to C++ find_executable function.
    Finds an executable in the PATH environment variable.

    Args:
        executable: Name of the executable to find

    Returns:
        Path to the executable if found, None otherwise
    """
    try:
        import shutil

        return shutil.which(executable)
    except ImportError:
        import distutils.spawn

        return distutils.spawn.find_executable(executable)


def spherePWaveDt(radius: Real, density: Real, young: Real) -> Real:
    """_summary_
    Compute P-wave travel time for a sphere using the analytical solution

    Args:
        radius (Real): _description_
        density (Real): _description_
        young (Real): _description_

    Returns:
        Real: _description_
    """
    return radius / math.sqrt(young / density)


def defaultMaterial():
    """
    Create a default material.
    This functions returns ``FrictMat(density=1e3, young=1e9, ktDivKn=.2, tanPhi=tan(.5))``
    """
    return FrictMat(density=1e3, young=1e9, ktDivKn=0.2, tanPhi=math.tan(0.5))


def defaultEngines(
    damping=0.0,
    verletDist=-0.05,
    collider="sap",
    kinSplit=False,
    dontCollect=False,
    dynDtPeriod=100,
):
    """
    Create default engines.
    Return default set of engines, suitable for basic simulations during testing
    """
    coll = None
    if collider == "sap":
        from pydem.src.InsertionSortCollider import InsertionSortCollider

        coll = InsertionSortCollider()

    coll.verletDist = verletDist

    # leapfrog
    from pydem.src.Leapfrog2 import Leapfrog2

    leapfrog = Leapfrog2()
    leapfrog.damping = damping
    leapfrog.kinSplit = kinSplit
    leapfrog.dontCollect = dontCollect
    leapfrog.reset = True

    # contactloop
    contactloop = ContactLoop()
    contactloop.applyForces = True

    # dynDt
    dynDt = DynDt()
    dynDt.stepPeriod = dynDtPeriod

    return [
        leapfrog,
        coll,
        contactloop,
    ] + ([dynDt] if dynDtPeriod > 0 else [])


def _commonBodySetup(b: Particle, nodes: List[Node], mat, fixed=False):
    """
    Assign common body parameters
    """

    if isinstance(mat, Material):
        b.material = mat
    elif callable(mat):
        b.material = mat()
    else:
        raise TypeError(
            f"Material must be Material or callable, got {type(mat)} instead."
        )

    b.shape.nodes = nodes

    if len(nodes) == 1:
        b.updateMassInertia()
    else:
        for n in b.shape.nodes:
            n.dem.mass = 0
            n.dem.inertia = Vector3r(0.0, 0.0, 0.0)

    for i, n in enumerate(b.shape.nodes):
        n.dem.addParticleRef(b)  # tell the node that it has ref to this body
        if fixed == None:
            pass
        elif fixed == True:
            n.dem.blocked = "xyzXYZ"
        elif fixed == False:
            n.dem.blocked = "".join(
                ["XYZ"[ax] for ax in (0, 1, 2) if n.dem.inertia[ax] == INF]
            )


def _mkDEMNode(**kw):
    if hasGL:
        pass
    else:
        n = Node()
        n.dem = DEMData()
        for k, v in kw.items():
            setattr(n.dem, k, v)
    return n


from pydem.src.DEMField import DEMField


def sphere(
    center,
    radius,
    mat=defaultMaterial,
    fixed=False,
    wire=False,
    color=None,
    hightlight=False,
    mask=DEMField.MASK_MOVABLE,
    vel=None,
):
    b = Particle()

    b.shape = Sphere()
    b.shape.radius = radius
    b.shape.color = color if color else random.random()
    b.shape.setWireframe(wire)

    _commonBodySetup(
        b,
        (
            [center]
            if isinstance(center, Node)
            else [
                _mkDEMNode(pos=center),
            ]
        ),
        mat=mat,
        fixed=fixed,
    )

    if vel:
        b.setVel(np.array(vel, dtype=np.float64))
    b.mask = mask
    return b
