from setuptools import setup, find_packages

setup(
    name="pydem",
    version="0.1.0",
    description="PyDEM - 离散元模拟框架",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "ipython>=7.0.0",
        # 其他依赖...
    ],
    entry_points={
        "console_scripts": [
            "pydem=pydem.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
)
