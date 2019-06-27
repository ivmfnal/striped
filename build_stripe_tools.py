#!/usr/bin/env python
import numpy as np
from setuptools import setup, Extension, find_packages

setup(name = "striped_c_tools",
      version = "1.0.0",
      description = "Extension module for Striped infrastructure",
      author = "Igor Mandrichenko (Fermilab)",
      author_email = "ivm@fnal.gov",
      maintainer = "Igor Mandrichenko (Fermilab)",
      maintainer_email = "ivm@fnal.gov",
      classifiers = ["Development Status :: 2 - Pre-Alpha",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                     ],
      ext_modules = [Extension('striped_c_tools',['stripe_tools/stripe_tools.c'])],
      include_dirs = [np.get_include()],
      zip_safe = False
      )
