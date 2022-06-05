#! /usr/bin/python3

from distutils.core import setup
from Cython.Build import cythonize
import numpy as np

setup(
	ext_modules = cythonize(
		['nq_funcs.pyx', 'sga_funcs.pyx', '_grid_search.pyx'],
		compiler_directives={'language_level': "3"},
		#language='c++',
		#extra_compile_args=["-O3"]
	),
	include_dirs = [np.get_include()],
)
