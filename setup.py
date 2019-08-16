#!/usr/bin/env python
import numpy as np
from setuptools import setup, Extension, find_packages

def get_version():
    globs = {}
    exec(open("striped/version.py", "r").read(), globs)
    version = globs["Version"]
    return version
    
packages = ['striped', 'striped.job', 'striped.common', 'striped.client', 'striped.ingestion', 'striped.pythreader']

setup(name = "striped",
      version = get_version(),
      packages = packages,
      scripts = ["ingest/ingestion/striped_ingest"],
      description = "",
      long_description = """""",
      author = "Igor Mandrichenko (Fermilab)",
      author_email = "ivm@fnal.gov",
      maintainer = "Igor Mandrichenko (Fermilab)",
      maintainer_email = "ivm@fnal.gov",
      #license = "Apache Software License v2",    # ???
      test_suite = "tests",
      install_requires = [],
      tests_require = [],
      classifiers = ["Development Status :: 2 - Pre-Alpha",
                     "Environment :: Console",
                     "Intended Audience :: Science/Research",
                     "License :: OSI Approved :: Apache Software License",
                     "Topic :: Scientific/Engineering :: Information Analysis",
                     "Topic :: Scientific/Engineering :: Mathematics",
                     "Topic :: Scientific/Engineering :: Physics",
                     ],
      platforms = "Any",
      ext_modules = [Extension('striped_c_tools',['stripe_tools/stripe_tools.c'])],
        include_dirs = [np.get_include()],
      zip_safe = False
      )
