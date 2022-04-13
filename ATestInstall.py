# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: JaxTutos
#     language: python
#     name: jaxtutos
# ---

# +
from pathinit import *

import numpy
import scipy

import jax
import jaxopt
import numpyro

from functools import partial
import matplotlib
import corner
import arviz

import GPy
import sklearn
# -

JAXTutos_default_version = {
    "matplotlib":"3.5.1",
    "numpy":"1.21.5",
    "scipy":"1.8.0",
    "scikit-learn":"1.0.2",
    "corner":"2.2.1",
    "GPy":"1.10.0",
    "arviz":"0.11.4",
    "numpyro":"0.9.1",
    "jax":"0.3.5",
    "jaxopt":"0.3.1"
}

# +
import pkg_resources
import types
def get_imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            # Split ensures you get root package, 
            # not just imported function
            name = val.__name__.split(".")[0]

        elif isinstance(val, type):
            name = val.__module__.split(".")[0]

        # Some packages are weird and have different
        # imported names vs. system/pip names. Unfortunately,
        # there is no systematic way to get pip names from
        # a package's imported name. You'll have to add
        # exceptions to this list manually!
        poorly_named_packages = {
            "PIL": "Pillow",
            "sklearn": "scikit-learn"
        }
        if name in poorly_named_packages.keys():
            name = poorly_named_packages[name]

        yield name
imports = list(set(get_imports()))

# The only way I found to get the version of the root package
# from only the name of the package is to cross-check the names 
# of installed packages vs. imported packages
requirements = []
for m in pkg_resources.working_set:
    if m.project_name in imports and m.project_name!="pip":
        requirements.append((m.project_name, m.version))

        
for r in requirements:
    assert JAXTutos_default_version[r[0]], f"missing {r[0]}"
    assert JAXTutos_default_version[r[0]]==r[1], f"wrong version {r[0]}:{r[1]}"
# -

for r in requirements:
    print(r)


