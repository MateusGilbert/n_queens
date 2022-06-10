# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython
from sga_funcs import sga_2 as sga
from libcpp.vector cimport vector

ctypedef struct tracker:
	vector[int] n_epochs
	vector[int] pop_size
	vector[int] n_par
	vector[ vector[int] ] J_min
	vector[ vector[float] ] J_med
	vector[ vector[ vector[int] ] ] s_tables
	vector[ int ] meme

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef tracker exp(int N_queens, int pop_size, float prop_par, int epochs, int meme=0, int rounds=10):
	cdef float p
	cdef tracker res

	cdef int l_epoch
	cdef vector[ int ] J_min
	cdef vector[ float ] J_med
	cdef vector[ vector [int] ] s_tables

	cdef int i
	cdef int n_par = int(np.ceil(prop_par*pop_size))

	for i in range(rounds):
		print(f'#{i+1} execution')
		l_epoch, J_min, J_med, s_tables = sga(N_queens, pop_size, n_par, epochs, meme)
		res.n_epochs.push_back(epochs)
		res.pop_size.push_back(pop_size)
		res.n_par.push_back(n_par)
		res.J_min.push_back(J_min)
		res.J_med.push_back(J_med)
		res.s_tables.push_back(s_tables)
		res.meme.push_back(meme)
		if 0 == J_min[-1]:
			print('Successful execution. A solution was found!')
		else:
			print('Failed execution!')

	return res
