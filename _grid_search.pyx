# distutils: language=c++

import numpy as np
cimport numpy as np
from sga_funcs import sga
from libcpp.vector cimport vector

ctypedef struct tracker:
	vector[int] n_epochs
	vector[int] pop_size
	vector[int] n_par
	vector[ vector[int] ] J_min
	vector[ vector[float] ] J_med
	vector[ vector[ vector[int] ] ] s_table

cpdef tracker grid_search(int N_queens, vector[int] pop_sizes, vector[float] prop_par, vector[int] epochs):
	cdef int pop_size
	cdef float p
	cdef tracker res

	s_table = np.empty(N_queens*N_queens, dtype=int)

	cdef int n_par

	for pop_size in pop_sizes:
		for p in prop_par:
			n_par = int(np.ceil(p*pop_size))
			for n_epochs in epochs:
				for i in range(5):				#repeat each combination 5 times
					print(f'{i+1} execution -> pop_size = {pop_size}; n_par = {n_par}; n_epochs = {n_epochs}')
					l_epoch, J_min, J_med, s_tables = sga(N_queens, pop_size, n_par, n_epochs)
					res.n_epochs.push_back(n_epochs)
					res.pop_size.push_back(pop_size)
					res.n_par.push_back(n_par)
					res.J_min.push_back(J_min)
					res.J_med.push_back(J_med)
					res.s_table.push_back(s_tables)

	return res
