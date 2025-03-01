#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class filter_iterator:
    """
    Python implementation of the C++ filter_iterator template.
    Provides a forward iterator that skips elements not satisfying a predicate.

    This class mimics the behavior of:
    template <typename Iterator, typename Predicate> class filter_iterator
    """

    def __init__(self, begin, end, pred):
        """
        Initialize with iterator range and predicate.

        Args:
            begin: Starting index or iterator
            end: Ending index or iterator
            pred: Predicate function that returns True for elements to include
        """
        self.container = None  # Will be set by the container class
        self.current = begin
        self.end = end
        self.pred = pred

        # Find first matching element
        self._advance_to_valid()

    def _advance_to_valid(self):
        """Advance to the next valid element that satisfies the predicate."""
        while self.current != self.end and not self.pred(self.current):
            self.current += 1

    def __iter__(self):
        """Return self as iterator."""
        return self

    def __next__(self):
        """Get next element satisfying the predicate."""
        if self.current == self.end:
            raise StopIteration

        # Get the current item
        item = self.container[self.current] if self.container else self.current

        # Move to next matching element
        self.current += 1
        self._advance_to_valid()

        return item

    # Dereference operators
    def __deref__(self):
        """Mimic C++ dereference operator (*)."""
        if self.container:
            return self.container[self.current]
        return self.current

    # Increment operators
    def increment(self):
        """Mimic C++ pre-increment operator (++it)."""
        self.current += 1
        self._advance_to_valid()
        return self

    def post_increment(self):
        """Mimic C++ post-increment operator (it++)."""
        tmp = filter_iterator(self.current, self.end, self.pred)
        tmp.container = self.container
        self.increment()
        return tmp

    # Comparison operators
    def __eq__(self, other):
        """Check if iterators point to same position."""
        return self.current == other.current

    def __ne__(self, other):
        """Check if iterators point to different positions."""
        return self.current != other.current
