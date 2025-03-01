#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Optional
from pydem.src.Object import Object


class ContactHook(Object):
    """Hook for contact events (creation and deletion)."""

    def __init__(self):
        """Initialize ContactHook with default values."""
        super().__init__()
        self.mask = 0  # Mask which must be matched by *both* particles in the contact

    def isMatch(self, maskA: int, maskB: int) -> bool:
        """
        Check if the masks match the hook's mask.

        Args:
            maskA: Mask of first particle
            maskB: Mask of second particle

        Returns:
            True if masks match, False otherwise
        """
        return self.mask == 0 or ((maskA & self.mask) and (maskB & self.mask))

    def hookNew(self, dem, contact):
        """
        Hook called when a new contact is created.

        Args:
            dem: DEM field
            contact: The new contact
        """
        pass  # Default implementation does nothing

    def hookDel(self, dem, contact):
        """
        Hook called when a contact is deleted.

        Args:
            dem: DEM field
            contact: The contact being deleted
        """
        pass  # Default implementation does nothing
