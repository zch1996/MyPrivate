#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple, TypeVar, Generic, Any
from pydem.src.demmath import Vector3r, Matrix3r, Real

# Define generic type variables
T = TypeVar("T")
TimeT = TypeVar("TimeT")


def linearInterpolateRel(
    t: Real, tt: List[TimeT], values: List[T], _pos: int
) -> Tuple[T, T, Real]:
    """
    Linear interpolation routine for sequential interpolation.
    Returns two endpoints and scalar value (timeT) in range (0,1) for interpolation between those two.

    Args:
        t: "time" at which the interpolated variable should be evaluated
        tt: "time" points at which values are given; must be increasing
        values: values at "time" points specified by tt
        _pos: holds lookup state (will be updated)

    Returns:
        Tuple containing:
            - First endpoint value
            - Second endpoint value
            - Relative position between endpoints (0-1)
    """
    pos = _pos  # use local copy for safe reentrant access, then write back
    assert len(tt) == len(values), "Time and value arrays must have the same length"

    if t <= tt[0]:
        _pos = 0
        return values[0], values[1 if len(values) > 1 else 0], 0.0

    if t >= tt[-1]:
        _pos = len(tt) - 2
        return values[pos] if len(values) > 1 else values[-1], values[-1], 1.0

    pos = min(pos, len(tt) - 2)
    while (tt[pos] > t) or (tt[pos + 1] < t):
        assert tt[pos] < tt[pos + 1], "Time values must be strictly increasing"
        if tt[pos] > t:
            pos -= 1
        else:
            pos += 1

    t0, t1 = tt[pos], tt[pos + 1]
    _pos = pos
    return values[pos], values[pos + 1], (t - t0) / (t1 - t0)


def linearInterpolate(t: Real, tt: List[TimeT], values: List[T], pos: int) -> T:
    """
    Linear interpolation function suitable only for sequential interpolation.

    Template parameter T must support: addition, subtraction, scalar multiplication.

    Args:
        t: "time" at which the interpolated variable should be evaluated
        tt: "time" points at which values are given; must be increasing
        values: values at "time" points specified by tt
        pos: holds lookup state (will be updated)

    Returns:
        Interpolated value at "time" t; out of range: t<t0 → value(t0), t>t_last → value(t_last)
    """
    A, B, tRel = linearInterpolateRel(t, tt, values, pos)

    if tRel == 0.0:
        return A
    if tRel == 1.0:
        return B

    return A + (B - A) * tRel


def linearInterpolateRel2D(
    x: Real, xxyy: List[np.ndarray], pos: int
) -> Tuple[Real, Real, Real]:
    """
    Linear interpolation for 2D points (x,y) based on x value.

    Args:
        x: x-value at which to interpolate
        xxyy: List of 2D points as numpy arrays [x, y]
        pos: holds lookup state (will be updated)

    Returns:
        Tuple containing:
            - First y endpoint value
            - Second y endpoint value
            - Relative position between endpoints (0-1)
    """
    if x <= xxyy[0][0]:
        pos = 0
        return xxyy[0][1], xxyy[1 if len(xxyy) > 1 else 0][1], 0.0

    if x >= xxyy[-1][0]:
        pos = len(xxyy) - 2
        return xxyy[pos][1] if len(xxyy) > 1 else xxyy[-1][1], xxyy[-1][1], 1.0

    pos = min(pos, len(xxyy) - 2)
    while (xxyy[pos][0] > x) or (xxyy[pos + 1][0] < x):
        assert xxyy[pos][0] < xxyy[pos + 1][0], "X values must be strictly increasing"
        if xxyy[pos][0] > x:
            pos -= 1
        else:
            pos += 1

    x0, x1 = xxyy[pos][0], xxyy[pos + 1][0]
    y0, y1 = xxyy[pos][1], xxyy[pos + 1][1]

    return y0, y1, (x - x0) / (x1 - x0)


def linearInterpolate2D(x: Real, xxyy: List[np.ndarray], pos: int) -> Real:
    """
    Linear interpolation for 2D points (x,y) based on x value.

    Args:
        x: x-value at which to interpolate
        xxyy: List of 2D points as numpy arrays [x, y]
        pos: holds lookup state (will be updated)

    Returns:
        Interpolated y-value at x
    """
    A, B, tRel = linearInterpolateRel2D(x, xxyy, pos)

    if tRel == 0.0:
        return A
    if tRel == 1.0:
        return B

    return A + (B - A) * tRel
