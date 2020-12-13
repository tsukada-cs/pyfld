#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
# from numpy.distutils.core import setup, Extension
import os

from setuptools import find_packages, setup

try:
    with open("README.md", "r", encoding="utf-8") as f:
        readme = f.read()
except IOError:
    readme = ""


def _requires_from_file(filename):
    return open(filename, "r", encoding="utf-8").read().splitlines()

extensions = []

# version
here = os.path.dirname(os.path.abspath(__file__))
version = next((line.split('=')[1].strip().replace('"', '').replace("'", '')
                for line in open(os.path.join(here, 'pyfld', '__init__.py'))
                if line.startswith('__version__ = ')), '0.0.dev0')
version


setup(
    name="pyfld",
    version=version,
    url="https://github.com/tsukada-cs/pyfld",
    author="Taiga Tsukada",
    author_email="tsukada.cs@gmail.com",
    maintainer="Taiga Tsukada",
    maintainer_email="tsukada.cs@gmail.com",
    description="pyfld",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    install_requires=_requires_from_file("requirements.txt"),
    license="GPLv3+",
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "License :: OSI Approved :: GNU General Public License v3 " +
        "or later (GPLv3+)",
        "Topic :: Scientific/Engineering"
    ],
    entry_points="""
      # -*- Entry points: -*-
      [console_scripts]
      pyfld = pyfld.script.command:main
    """,
)
