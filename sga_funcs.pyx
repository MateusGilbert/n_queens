# distutils: language=c++

from nq_funcs import *
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

cpdef np.ndarray one_point_xover(np.ndarray par, int m):
	'''
		entries:
			par == parents
			m == num of parents
		exit:
			chil == childs

		comements:
			function outputs m childs. Each
			parent are paired, using permutation.
			While m > #par, pairing process is
			repeated.

	'''
	cdef int n = par.shape[0]
	cdef int s = par.shape[1]
	#print(n,m); input()
	cdef np.ndarray idxs = floyd_sampler(n,min(n,m))
	#print(idxs); input()

	while (m > idxs.shape[0]):
		#print(idxs); input()
		idxs = np.concatenate([idxs, floyd_sampler(n, min(m-idxs.shape[0],n))])
	cdef int i,j
	pairs = [[i,j] for i,j in zip(idxs[::2],idxs[1::2])]

	cdef pivot
	cdef np.ndarray ch_1, ch_2
	pop = list()
	for i,j in pairs:
		pivot = np.random.randint(s-1)
		ch_1 = np.concatenate([par[i][:pivot], par[j][pivot:]])
		ch_2 = np.concatenate([par[j][:pivot], par[i][pivot:]])
		pop.extend([ch_1,ch_2])

	return np.array(pop)

cpdef np.ndarray bit_flip(np.ndarray pop, float prop=.05):
	'''
		entry:
			pop == population
			prop == portion that will suffer mutation
		exits:
			pop w/ mutated individuos

		comments:
			as default, approximately 5% of the
			population will suffer mutation. At
			least one will suffer mutationl.
	'''
	cdef int n = pop.shape[0]
	cdef int s = pop.shape[1]
	cdef np.ndarray idxs = floyd_sampler(n,int(np.ceil(n*prop)))

	cdef int i, pos
	for i in idxs:
		pos = np.random.randint(s)
		pop[i][pos] = 0 if pop[i][pos] == 1 else 1

	return np.array(pop)

cpdef np.ndarray rank_sel(np.ndarray pop, int k):
	'''
		entries:
			pop == populacao
			k == numb. of selected ind.
		exit:
			par == parents
	'''
	cdef np.ndarray i
	fitness = [eval_table(conv_tab(i)) for i in pop]
	sort_idx = np.argsort(fitness).tolist()			#order by fitness
	cdef np.ndarray par = pop[sort_idx[:k]]			#select parents
	return par

cpdef np.ndarray meme(np.ndarray pop):
	cdef np.ndarray aux
	cdef np.ndarray o_pop = np.zeros((pop.shape[0],pop.shape[1]), dtype=np.int16)
	for i,aux in enumerate(pop):
		aux = conv_tab(aux)
		aux[:] = mov_comp(aux)
		aux[:] = rem_queens(aux)
		o_pop[i] = conv_tab(aux)
	return o_pop

cpdef (int, vector[int], vector[float], vector[ vector[int] ]) sga(int N_queens, int pop_size, int n_par, int n_epochs=10000, int u_meme=0):
	cdef int epoch=0,i
	cdef vector[int] J_best = np.zeros(n_epochs, dtype=int)
	cdef vector[float] J_med = np.zeros(n_epochs)
	cdef vector[ vector[int] ] sol
	cdef np.ndarray pop = np.array(
		[conv_tab(init_table(N_queens)) for i in range(pop_size)]
	)

	cdef np.ndarray par, res = np.zeros(pop_size, dtype=np.int16)
	cdef np.ndarray ind, b_res = np.zeros(N_queens*N_queens, dtype=np.int16)
	cdef b_fit=N_queens*100
	cdef float mean_fit
	while(epoch<n_epochs):
		par = rank_sel(pop, n_par)
		pop[:] = one_point_xover(par, pop_size)
		pop[:] = bit_flip(pop)
		if u_meme > 0:
			pop[:] = meme(pop)

		#evaluate epoch results
		for i,ind in enumerate(pop):
			res[i] = eval_table(conv_tab(ind))
			if res[i] < b_fit:
				b_res[:] = pop[i].copy()
				b_fit = res[i]
		J_best[epoch] = b_fit
		J_med[epoch] = np.mean(res)
		if any(res == 0):
			break
		epoch += 1

	cdef np.ndarray cand_res = np.where(res == b_fit)[0]
	for i in cand_res:
		sol.push_back(pop[i].tolist())

	return epoch, J_best[:epoch], J_med[:epoch], sol
