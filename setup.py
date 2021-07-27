#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
from hotelling import __version__ as version

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = ["click", "scipy", "matplotlib", "pandas", "numpy"]

setup_requirements = ["pytest-runner"]

test_requirements = ["pytest>=3.5"]

setup(
    author="Francois Dion",
    author_email="fdion@dionresearch.com",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    description="Hotelling implements one and two sample Hotelling T2 tests and control charts",
    entry_points={"console_scripts": ["hotelling=hotelling.cli:main"]},
    install_requires=requirements,
    license="MIT license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="hotelling",
    name="hotelling",
    packages=find_packages(include=["hotelling", "hotelling.*"]),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/dionresearch/hotelling",
    version=version,
    zip_safe=False,
)
