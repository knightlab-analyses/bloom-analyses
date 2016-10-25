import os
import platform
import re
import ast
import sys

from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy as np

classes = """
    Development Status :: 1 - Pre-Alpha
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

setup(name='bloom_analyses',
      version='1.0',
      license='BSD-3-Clause',
      description='The Bloom Filter',
      long_description='The Bloom Filter',
      author="bloom development team",
      author_email="jamietmorton@gmail.com",
      maintainer="bloom development team",
      maintainer_email="jamietmorton@gmail.com",
      packages=find_packages(),
      include_dirs=[np.get_include()],
      install_requires=[
          'scikit-bio', 'numpy', 'biom-format'
      ],
      classifiers=classifiers,
      package_data={
          }
      )
