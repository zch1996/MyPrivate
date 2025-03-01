#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydem.src.Object import Object


class ContactData(Object):
    """Base class for contact-specific data."""

    def __init__(self):
        """Initialize ContactData with default values."""
        super().__init__()

    def toString(self) -> str:
        """Return string representation."""
        return f"{self.getClassName()}"


class IdealElPlData(ContactData):
    """Contact data for ideal elastic-plastic law."""

    def __init__(self):
        """Initialize with default values."""
        super().__init__()
        self.uN0 = 0.0  # Reference normal displacement
