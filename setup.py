#!/usr/bin/env python
# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
# from numpy.distutils.core import setup, Extension
import os

from setuptools import find_packages, setup

try:
    with open('README.md') as f:
        readme = f.read()
except IOError:
    readme = ''


def _requires_from_file(filename):
    return open(filename).read().splitlines()

extensions = []

setup(
    name="pyfld",
    version="0.2.4",
    url='https://github.com/tsukada-cs/pyfld',
    author='Taiga Tsukada',
    author_email='tsukada.cs@gmail.com',
    maintainer='Taiga Tsukada',
    maintainer_email='tsukada.cs@gmail.com',
    description='pyfld',
    long_description=readme,
    packages=find_packages(),
    install_requires=_requires_from_file('requirements.txt'),
    license="GPLv3+",
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
    ],
    entry_points="""
      # -*- Entry points: -*-
      [console_scripts]
      pyfld = pyfld.script.command:main
    """,
)
