#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name="PyDEM",
    version="0.1.0",
    description="Python Discrete Element Method Simulation Framework",
    author="PyDEM Team",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "coloredlogs",
        "PyOpenGL",
        "PyOpenGL_accelerate",
        "pygame",
        "PyQt5",
        "ipython",
    ],
    entry_points={
        "console_scripts": [
            "pydem=pydem.__main__:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
