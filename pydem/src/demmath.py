#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from numpy import linalg as la
import math
from scipy.spatial.transform import Rotation
import random

# Define precision
USE_QUAD_PRECISION = False

if USE_QUAD_PRECISION:
    try:
        # Try to use float128 if available
        Real = np.float128
    except AttributeError:
        # Fall back to float64 if float128 is not available
        Real = np.float64
else:
    Real = np.float64

# Mathematical constants
EPSILON = 1e-6
PI = math.pi
HALF_PI = math.pi / 2
TWO_PI = 2.0 * math.pi
NAN = np.nan
INF = np.inf


def Vector2i(x=0, y=0):
    """Create a 2D integer vector."""
    return np.array([x, y], dtype=np.int32)


def Vector2r(x=0.0, y=0.0):
    """Create a 2D real vector."""
    return np.array([x, y], dtype=Real)


def Vector3i(x=0, y=0, z=0):
    """Create a 3D integer vector."""
    return np.array([x, y, z], dtype=np.int32)


def Vector3r(x=0.0, y=0.0, z=0.0):
    """Create a 3D real vector."""
    return np.array([x, y, z], dtype=Real)


def Vector4r(x=0.0, y=0.0, z=0.0, w=0.0):
    """Create a 4D real vector."""
    return np.array([x, y, z, w], dtype=Real)


def Vector6i(*args):
    """Create a 6D integer vector."""
    if len(args) == 0:
        return np.zeros(6, dtype=np.int32)
    elif len(args) == 6:
        return np.array(args, dtype=np.int32)
    else:
        raise ValueError("Vector6i requires 0 or 6 arguments")


def Vector6r(*args):
    """Create a 6D real vector."""
    if len(args) == 0:
        return np.zeros(6, dtype=Real)
    elif len(args) == 6:
        return np.array(args, dtype=Real)
    else:
        raise ValueError("Vector6r requires 0 or 6 arguments")


def Vector9r(*args):
    """Create a 9D real vector."""
    if len(args) == 0:
        return np.zeros(9, dtype=Real)
    elif len(args) == 9:
        return np.array(args, dtype=Real)
    else:
        raise ValueError("Vector9r requires 0 or 9 arguments")


def Vector12r(*args):
    """Create a 12D real vector."""
    if len(args) == 0:
        return np.zeros(12, dtype=Real)
    elif len(args) == 12:
        return np.array(args, dtype=Real)
    else:
        raise ValueError("Vector12r requires 0 or 12 arguments")


def Matrix3r(*args):
    """Create a 3x3 real matrix."""
    if len(args) == 0:
        return np.zeros((3, 3), dtype=Real)
    elif len(args) == 9:
        return np.array(args, dtype=Real).reshape(3, 3)
    else:
        raise ValueError("Matrix3r requires 0 or 9 arguments")


def Matrix3i(*args):
    """Create a 3x3 integer matrix."""
    if len(args) == 0:
        return np.zeros((3, 3), dtype=np.int32)
    elif len(args) == 9:
        return np.array(args, dtype=np.int32).reshape(3, 3)
    else:
        raise ValueError("Matrix3i requires 0 or 9 arguments")


def Matrix4r(*args):
    """Create a 4x4 real matrix."""
    if len(args) == 0:
        return np.zeros((4, 4), dtype=Real)
    elif len(args) == 16:
        return np.array(args, dtype=Real).reshape(4, 4)
    else:
        raise ValueError("Matrix4r requires 0 or 16 arguments")


def Matrix6r(*args):
    """Create a 6x6 real matrix."""
    if len(args) == 0:
        return np.zeros((6, 6), dtype=Real)
    elif len(args) == 36:
        return np.array(args, dtype=Real).reshape(6, 6)
    else:
        raise ValueError("Matrix6r requires 0 or 36 arguments")


def MatrixXr(rows, cols):
    """Create a dynamic-sized real matrix."""
    return np.zeros((rows, cols), dtype=Real)


def VectorXr(size):
    """Create a dynamic-sized real vector."""
    return np.zeros(size, dtype=Real)


# Helper class for quaternions (wrapping scipy.spatial.transform.Rotation)
class Quaternionr:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        """Create a quaternion with w + xi + yj + zk."""
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    @staticmethod
    def fromRotationMatrix(matrix):
        """Create quaternion from rotation matrix."""
        r = Rotation.from_matrix(matrix)
        quat = r.as_quat()  # returns x, y, z, w
        return Quaternionr(quat[3], quat[0], quat[1], quat[2])

    def toRotationMatrix(self):
        """Convert quaternion to rotation matrix."""
        quat = np.array([self.x, self.y, self.z, self.w])
        r = Rotation.from_quat(quat)
        return r.as_matrix()

    def conjugate(self):
        """Return the conjugate of this quaternion."""
        return Quaternionr(self.w, -self.x, -self.y, -self.z)

    def inverse(self):
        """Return the inverse of this quaternion."""
        norm_squared = self.w**2 + self.x**2 + self.y**2 + self.z**2
        if norm_squared < EPSILON:
            return Quaternionr(0, 0, 0, 0)  # Invalid quaternion
        inv_norm_squared = 1.0 / norm_squared
        return Quaternionr(
            self.w * inv_norm_squared,
            -self.x * inv_norm_squared,
            -self.y * inv_norm_squared,
            -self.z * inv_norm_squared,
        )

    def normalize(self):
        """Return a normalized copy of this quaternion."""
        norm = math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)
        if norm < EPSILON:
            return Quaternionr(1, 0, 0, 0)  # Return identity quaternion
        inv_norm = 1.0 / norm
        return Quaternionr(
            self.w * inv_norm, self.x * inv_norm, self.y * inv_norm, self.z * inv_norm
        )

    def dot(self, other):
        """Compute dot product between two quaternions."""
        if not isinstance(other, Quaternionr):
            raise TypeError("Other must be a Quaternionr")
        return self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z

    def slerp(self, other, t):
        """Spherical linear interpolation between quaternions."""
        if not isinstance(other, Quaternionr):
            raise TypeError("Other must be a Quaternionr")

        # Compute cosine of angle between quaternions
        cos_angle = self.dot(other)

        # If quaternions are very close, use linear interpolation
        if abs(cos_angle) > 0.9995:
            result = self * (1 - t) + other * t
            return result.normalize()

        # Ensure shortest path by negating one quaternion if needed
        if cos_angle < 0:
            cos_angle = -cos_angle
            q2 = Quaternionr(-other.w, -other.x, -other.y, -other.z)
        else:
            q2 = other

        # Calculate interpolation parameters
        angle = math.acos(cos_angle)
        sin_angle = math.sin(angle)
        w1 = math.sin((1 - t) * angle) / sin_angle
        w2 = math.sin(t * angle) / sin_angle

        # Compute interpolated quaternion
        result = self * w1 + q2 * w2
        return result.normalize()

    def toAngleAxis(self):
        """Convert quaternion to angle-axis representation."""
        # Normalize quaternion
        q = self.normalize()

        # Calculate angle
        angle = 2 * math.acos(q.w)

        # Calculate axis
        if abs(angle) < EPSILON:
            # If angle is very small, return zero rotation around x-axis
            return 0.0, Vector3r(1.0, 0.0, 0.0)

        s = 1.0 / math.sqrt(1 - q.w**2)
        axis = Vector3r(q.x * s, q.y * s, q.z * s)

        return angle, axis

    def copy(self):
        """Return a copy of this quaternion."""
        return Quaternionr(self.w, self.x, self.y, self.z)

    def coeffs(self):
        """Return quaternion as a vector [x, y, z, w]."""
        return np.array([self.x, self.y, self.z, self.w], dtype=Real)

    def __mul__(self, other):
        """Quaternion multiplication."""
        if isinstance(other, Quaternionr):
            w = (
                self.w * other.w
                - self.x * other.x
                - self.y * other.y
                - self.z * other.z
            )
            x = (
                self.w * other.x
                + self.x * other.w
                + self.y * other.z
                - self.z * other.y
            )
            y = (
                self.w * other.y
                - self.x * other.z
                + self.y * other.w
                + self.z * other.x
            )
            z = (
                self.w * other.z
                + self.x * other.y
                - self.y * other.x
                + self.z * other.w
            )
            return Quaternionr(w, x, y, z)
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            # Apply rotation to a vector
            q = self.normalize()
            v = other

            # Formula: v' = q * v * q^-1
            # Optimized implementation
            t = 2.0 * np.cross(np.array([q.x, q.y, q.z]), v)
            return v + q.w * t + np.cross(np.array([q.x, q.y, q.z]), t)
        elif isinstance(other, (int, float, np.number)):
            return Quaternionr(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        else:
            raise TypeError("Unsupported multiplication type")

    def __rmul__(self, other):
        """Scalar-quaternion multiplication."""
        if isinstance(other, (int, float, np.number)):
            return Quaternionr(
                self.w * other, self.x * other, self.y * other, self.z * other
            )
        else:
            raise TypeError("Unsupported multiplication type")

    def __add__(self, other):
        """Quaternion addition."""
        if isinstance(other, Quaternionr):
            return Quaternionr(
                self.w + other.w, self.x + other.x, self.y + other.y, self.z + other.z
            )
        else:
            raise TypeError("Unsupported addition type")

    def __sub__(self, other):
        """Quaternion subtraction."""
        if isinstance(other, Quaternionr):
            return Quaternionr(
                self.w - other.w, self.x - other.x, self.y - other.y, self.z - other.z
            )
        else:
            raise TypeError("Unsupported subtraction type")

    def __eq__(self, other):
        """Quaternion equality."""
        if not isinstance(other, Quaternionr):
            return False
        return (
            self.w == other.w
            and self.x == other.x
            and self.y == other.y
            and self.z == other.z
        )

    def __ne__(self, other):
        """Quaternion inequality."""
        return not self.__eq__(other)

    def __getitem__(self, index):
        """Allow indexing into quaternion components."""
        if index == 0:
            return self.w
        elif index == 1:
            return self.x
        elif index == 2:
            return self.y
        elif index == 3:
            return self.z
        else:
            raise IndexError("Quaternion index out of range")

    def __setitem__(self, index, value):
        """Allow setting quaternion components by index."""
        if index == 0:
            self.w = value
        elif index == 1:
            self.x = value
        elif index == 2:
            self.y = value
        elif index == 3:
            self.z = value
        else:
            raise IndexError("Quaternion index out of range")

    def __str__(self):
        """String representation of quaternion."""
        return f"Quaternionr({self.w}, {self.x}, {self.y}, {self.z})"

    def __repr__(self):
        """Detailed string representation of quaternion."""
        return f"Quaternionr(w={self.w}, x={self.x}, y={self.y}, z={self.z})"


# Helper class for angle axis representation
class AngleAxisr:
    def __init__(self, angle=0.0, axis=None):
        """Create an angle-axis rotation with angle and axis."""
        self.angle = angle
        if axis is None:
            self.axis = Vector3r(1.0, 0.0, 0.0)
        else:
            self.axis = axis / np.linalg.norm(axis)

    def toRotationMatrix(self):
        """Convert angle-axis to rotation matrix."""
        r = Rotation.from_rotvec(self.angle * self.axis)
        return r.as_matrix()

    def toQuaternion(self):
        """Convert angle-axis to quaternion."""
        r = Rotation.from_rotvec(self.angle * self.axis)
        quat = r.as_quat()  # x, y, z, w format
        return Quaternionr(quat[3], quat[0], quat[1], quat[2])

    def __str__(self):
        """String representation of angle-axis."""
        return f"AngleAxisr({self.angle}, [{self.axis[0]}, {self.axis[1]}, {self.axis[2]}])"

    def __repr__(self):
        """Detailed string representation of angle-axis."""
        return f"AngleAxisr(angle={self.angle}, axis=[{self.axis[0]}, {self.axis[1]}, {self.axis[2]}])"


# Helper class for aligned boxes
class AlignedBox2r:
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            self.min = Vector2r(INF, INF)
        else:
            self.min = min_point

        if max_point is None:
            self.max = Vector2r(-INF, -INF)
        else:
            self.max = max_point

    def extend(self, point):
        """Extend box to include point."""
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def contains(self, point):
        """Check if box contains point."""
        return np.all(point >= self.min) and np.all(point <= self.max)

    def isEmpty(self):
        """Check if box is empty (has no volume)."""
        return np.any(self.min > self.max)

    def center(self):
        """Get center of box."""
        return 0.5 * (self.min + self.max)

    def sizes(self):
        """Get dimensions of box."""
        return self.max - self.min

    def volume(self):
        """Get volume (area) of box."""
        sizes = self.sizes()
        if np.any(sizes < 0):
            return 0.0
        return sizes[0] * sizes[1]

    def merge(self, other):
        """Merge with another box."""
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)

    def intersect(self, other):
        """Get intersection with another box."""
        result = AlignedBox2r()
        result.min = np.maximum(self.min, other.min)
        result.max = np.minimum(self.max, other.max)
        return result

    def __str__(self):
        """String representation of box."""
        return f"AlignedBox2r(min=[{self.min[0]}, {self.min[1]}], max=[{self.max[0]}, {self.max[1]}])"


class AlignedBox3r:
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            self.min = Vector3r(INF, INF, INF)
        else:
            self.min = min_point

        if max_point is None:
            self.max = Vector3r(-INF, -INF, -INF)
        else:
            self.max = max_point

    def extend(self, point):
        """Extend box to include point."""
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def contains(self, point):
        """Check if box contains point."""
        return np.all(point >= self.min) and np.all(point <= self.max)

    def isEmpty(self):
        """Check if box is empty (has no volume)."""
        return np.any(self.min > self.max)

    def center(self):
        """Get center of box."""
        return 0.5 * (self.min + self.max)

    def sizes(self):
        """Get dimensions of box."""
        return self.max - self.min

    def volume(self):
        """Get volume of box."""
        sizes = self.sizes()
        if np.any(sizes < 0):
            return 0.0
        return sizes[0] * sizes[1] * sizes[2]

    def merge(self, other):
        """Merge with another box."""
        self.min = np.minimum(self.min, other.min)
        self.max = np.maximum(self.max, other.max)

    def intersect(self, other):
        """Get intersection with another box."""
        result = AlignedBox3r()
        result.min = np.maximum(self.min, other.min)
        result.max = np.minimum(self.max, other.max)
        return result

    def containsBox(self, other):
        """Check if box contains another box."""
        return np.all(self.min <= other.min) and np.all(self.max >= other.max)

    def distanceTo(self, point):
        """Get distance to point."""
        closest = np.maximum(self.min, np.minimum(point, self.max))
        return np.linalg.norm(closest - point)

    def __str__(self):
        """String representation of box."""
        return f"AlignedBox3r(min=[{self.min[0]}, {self.min[1]}, {self.min[2]}], max=[{self.max[0]}, {self.max[1]}, {self.max[2]}])"


class AlignedBox2i:
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            self.min = Vector2i(np.iinfo(np.int32).max, np.iinfo(np.int32).max)
        else:
            self.min = min_point

        if max_point is None:
            self.max = Vector2i(np.iinfo(np.int32).min, np.iinfo(np.int32).min)
        else:
            self.max = max_point

    def extend(self, point):
        """Extend box to include point."""
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def contains(self, point):
        """Check if box contains point."""
        return np.all(point >= self.min) and np.all(point <= self.max)

    def isEmpty(self):
        """Check if box is empty (has no volume)."""
        return np.any(self.min > self.max)

    def sizes(self):
        """Get dimensions of box."""
        return self.max - self.min


class AlignedBox3i:
    def __init__(self, min_point=None, max_point=None):
        if min_point is None:
            self.min = Vector3i(
                np.iinfo(np.int32).max, np.iinfo(np.int32).max, np.iinfo(np.int32).max
            )
        else:
            self.min = min_point

        if max_point is None:
            self.max = Vector3i(
                np.iinfo(np.int32).min, np.iinfo(np.int32).min, np.iinfo(np.int32).min
            )
        else:
            self.max = max_point

    def extend(self, point):
        """Extend box to include point."""
        self.min = np.minimum(self.min, point)
        self.max = np.maximum(self.max, point)

    def contains(self, point):
        """Check if box contains point."""
        return np.all(point >= self.min) and np.all(point <= self.max)

    def isEmpty(self):
        """Check if box is empty (has no volume)."""
        return np.any(self.min > self.max)

    def sizes(self):
        """Get dimensions of box."""
        return self.max - self.min


# Matrix operations
def matrix_eigen_decomposition(m):
    """Compute eigendecomposition of a 3x3 symmetric matrix."""
    eigenvalues, eigenvectors = np.linalg.eigh(m)
    return eigenvectors, np.diag(eigenvalues)


def matrix_from_euler_angles_xyz(x, y, z):
    """Create rotation matrix from Euler angles XYZ."""
    r = Rotation.from_euler("xyz", [x, y, z])
    return r.as_matrix()


# Math functions
class Math:
    """Math utility class with various helper functions."""

    @staticmethod
    def sign(f):
        """Return sign of a number."""
        if f < 0:
            return -1
        if f > 0:
            return 1
        return 0

    @staticmethod
    def unitRandom():
        """Return random number in [0,1]."""
        return random.random()

    @staticmethod
    def unitRandom3():
        """Return random 3D vector with components in [0,1]."""
        return Vector3r(random.random(), random.random(), random.random())

    @staticmethod
    def intervalRandom(a, b):
        """Return random number in [a,b]."""
        return a + random.random() * (b - a)

    @staticmethod
    def symmetricRandom():
        """Return random number in [-1,1]."""
        return 2.0 * random.random() - 1.0

    @staticmethod
    def fastInvCos0(value):
        """Fast inverse cosine approximation."""
        root = math.sqrt(1.0 - value)
        result = -0.0187293
        result *= value
        result += 0.0742610
        result *= value
        result -= 0.2121144
        result *= value
        result += 1.5707288
        result *= root
        return result

    @staticmethod
    def safeNormalize(v, tolerance=EPSILON):
        """Safely normalize a vector."""
        norm = np.linalg.norm(v)
        if norm < tolerance:
            return np.zeros_like(v)
        return v / norm

    @staticmethod
    def project(v, n):
        """Project vector v onto vector n."""
        return n * (np.dot(v, n) / np.dot(n, n))

    @staticmethod
    def reject(v, n):
        """Reject vector v from vector n (get perpendicular component)."""
        return v - Math.project(v, n)

    @staticmethod
    def createOrthonormalBasis(n):
        """Create orthonormal basis from normal vector n."""
        n = Math.safeNormalize(n)
        if abs(n[2]) > 0.7071067811865475:
            a = n[1] * n[1] + n[2] * n[2]
            k = 1.0 / math.sqrt(a)
            b1 = Vector3r(0, -n[2] * k, n[1] * k)
            b2 = Vector3r(a * k, -n[0] * b1[2], n[0] * b1[1])
        else:
            a = n[0] * n[0] + n[1] * n[1]
            k = 1.0 / math.sqrt(a)
            b1 = Vector3r(-n[1] * k, n[0] * k, 0)
            b2 = Vector3r(-n[2] * b1[1], n[2] * b1[0], a * k)
        return b1, b2

    @staticmethod
    def uniformRandomRotation():
        """Generate uniform random rotation as quaternion."""
        u1, u2, u3 = random.random(), random.random(), random.random()
        return Quaternionr(
            math.sqrt(u1) * math.cos(2 * PI * u3),
            math.sqrt(1 - u1) * math.sin(2 * PI * u2),
            math.sqrt(1 - u1) * math.cos(2 * PI * u2),
            math.sqrt(u1) * math.sin(2 * PI * u3),
        )

    @staticmethod
    def rotationMatrix(axis, angle):
        """Create rotation matrix from axis and angle."""
        axis = Math.safeNormalize(axis)
        s = math.sin(angle)
        c = math.cos(angle)
        t = 1.0 - c

        x, y, z = axis

        return Matrix3r(
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
        )

    @staticmethod
    def clamp(value, min_val, max_val):
        """Clamp value between min and max."""
        return max(min_val, min(value, max_val))

    @staticmethod
    def lerp(a, b, t):
        """Linear interpolation between a and b."""
        return a + t * (b - a)

    @staticmethod
    def distancePointPlane(point, planePoint, planeNormal):
        """Calculate distance from point to plane."""
        planeNormal = Math.safeNormalize(planeNormal)
        return np.dot(point - planePoint, planeNormal)


# Shorthand
Mathr = Math


# Helper functions
def sgn(val):
    """Return sign of value (-1, 0, 1)."""
    return (0 < val) - (val < 0)


def pow2(x):
    """Power 2 function."""
    return x * x


def pow3(x):
    """Power 3 function."""
    return x * x * x


def pow4(x):
    """Power 4 function."""
    return pow2(x * x)


def pow5(x):
    """Power 5 function."""
    return pow4(x) * x


def pown(x, n):
    """Integer power function."""
    if n == 0:
        return type(x)(1)
    d = n >> 1
    r = n & 1
    x_2_d = pown(x * x, d) if d > 0 else 1
    x_r = x if r > 0 else 1
    return x_2_d * x_r


# Voigt notation conversion functions
def voigt_to_symm_tensor(v, strain=False):
    """Convert Voigt vector to symmetric tensor."""
    k = 0.5 if strain else 1.0
    return np.array(
        [
            [v[0], k * v[5], k * v[4]],
            [k * v[5], v[1], k * v[3]],
            [k * v[4], k * v[3], v[2]],
        ],
        dtype=Real,
    )


def tensor_to_voigt(m, strain=False):
    """Convert symmetric tensor to Voigt vector."""
    k = 2 if strain else 1
    return np.array(
        [
            m[0, 0],
            m[1, 1],
            m[2, 2],
            k * 0.5 * (m[1, 2] + m[2, 1]),
            k * 0.5 * (m[2, 0] + m[0, 2]),
            k * 0.5 * (m[0, 1] + m[1, 0]),
        ],
        dtype=Real,
    )


def levi_civita(m):
    """Compute Levi-Civita product with a matrix."""
    return Vector3r(m[1, 2] - m[2, 1], m[2, 0] - m[0, 2], m[0, 1] - m[1, 0])


# Voigt mapping
# This is a complex 3D array, which we'll represent as a function in Python
def voigt_map(i, j, k):
    """Return the Voigt mapping indices."""
    mapping = [
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 2, 2],
            [0, 0, 1, 2],
            [0, 0, 2, 0],
            [0, 0, 0, 1],
        ],
        [
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 2, 2],
            [1, 1, 1, 2],
            [1, 1, 2, 0],
            [1, 1, 0, 1],
        ],
        [
            [2, 2, 0, 0],
            [2, 2, 1, 1],
            [2, 2, 2, 2],
            [2, 2, 1, 2],
            [2, 2, 2, 0],
            [2, 2, 0, 1],
        ],
        [
            [1, 2, 0, 0],
            [1, 2, 1, 1],
            [1, 2, 2, 2],
            [1, 2, 1, 2],
            [1, 2, 2, 0],
            [1, 2, 0, 1],
        ],
        [
            [2, 0, 0, 0],
            [2, 0, 1, 1],
            [2, 0, 2, 2],
            [2, 0, 1, 2],
            [2, 0, 2, 0],
            [2, 0, 0, 1],
        ],
        [
            [0, 1, 0, 0],
            [0, 1, 1, 1],
            [0, 1, 2, 2],
            [0, 1, 1, 2],
            [0, 1, 2, 0],
            [0, 1, 0, 1],
        ],
    ]
    if 0 <= i < 6 and 0 <= j < 6 and 0 <= k < 4:
        return mapping[i][j][k]
    else:
        raise IndexError("Index out of range for voigt_map")
