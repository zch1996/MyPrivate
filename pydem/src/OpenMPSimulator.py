#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import threading


class OpenMPSimulator:
    """
    Simulates OpenMP-like functionality in Python.
    Provides equivalent functions to omp_get_thread_num() and omp_get_max_threads().
    """

    _thread_local = threading.local()
    _thread_ids = {}
    _lock = threading.RLock()
    _max_threads = multiprocessing.cpu_count()

    @staticmethod
    def initialize():
        """Initialize the OpenMP simulator."""
        OpenMPSimulator._thread_ids.clear()

    @staticmethod
    def get_thread_num():
        """
        Equivalent to omp_get_thread_num().
        Returns the current thread's ID within the OpenMP thread pool.
        """
        thread_id = threading.get_ident()

        with OpenMPSimulator._lock:
            if thread_id not in OpenMPSimulator._thread_ids:
                # Assign a new thread ID
                OpenMPSimulator._thread_ids[thread_id] = (
                    len(OpenMPSimulator._thread_ids) % OpenMPSimulator._max_threads
                )

            return OpenMPSimulator._thread_ids[thread_id]

    @staticmethod
    def get_max_threads():
        """
        Equivalent to omp_get_max_threads().
        Returns the maximum number of threads available.
        """
        return OpenMPSimulator._max_threads

    @staticmethod
    def set_max_threads(num_threads):
        """
        Equivalent to omp_set_num_threads().
        Sets the maximum number of threads to use.
        """
        if num_threads > 0:
            OpenMPSimulator._max_threads = num_threads


# Create convenience functions that mimic OpenMP API
def omp_get_thread_num():
    """Get the thread number in the current team."""
    return OpenMPSimulator.get_thread_num()


def omp_get_max_threads():
    """Get the maximum number of threads that can be used."""
    return OpenMPSimulator.get_max_threads()


def omp_set_num_threads(num_threads):
    """Set the number of threads to use."""
    OpenMPSimulator.set_max_threads(num_threads)
