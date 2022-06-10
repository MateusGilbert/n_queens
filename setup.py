#! /usr/bin/python3

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
	ext_modules = cythonize(#[
		Extension(
			'*',
			['*.pyx'],#['nq_funcs.pyx', 'sga_funcs.pyx', '_grid_search.pyx', '_exp.pyx'],
			language='c++',
			extra_compile_args=["-O3"]
		),
	compiler_directives={'language_level': "3"}),
	include_dirs = [np.get_include()],
)
