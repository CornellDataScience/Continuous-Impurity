import setuptools
from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize(["model/impurity/global_impurity/global_impurity_model_tree2.py"])
)
