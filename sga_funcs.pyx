# distutils: language=c++

from nq_funcs import *
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray one_point_xover(np.ndarray par, int m):
	'''
		entries:
			par == parents
			m == num of parents
		exit:
			chil == childs

		comments:
			function outputs m childs. Each
			parent are paired, using permutation.
			While m > #par, pairing process is
			repeated.

	'''
	cdef int n = par.shape[0]
	cdef int s = par.shape[1]
	cdef np.ndarray idxs = floyd_sampler(n,min(n,m))

	while (m > idxs.shape[0]):
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] order_xover(int[:,:] par, int m):
	'''
		entries:
			par == parents
			m == num of parents
		exit:
			chil == childs

		comments:
			function outputs m childs. Each
			parent are paired, using permutation.
			While m > #par, pairing process is
			repeated.

	'''
	cdef int n = par.shape[0]
	cdef int s = par.shape[1]
	cdef int[:] idxs = floyd_sampler(n,min(n,m))

	while (m > idxs.shape[0]):
		idxs = np.concatenate([idxs, floyd_sampler(n, min(m-idxs.shape[0],n))])
	cdef int i,j
	pairs = [[i,j] for i,j in zip(idxs[::2],idxs[1::2])]

	cdef int pivot_1, pivot_2, choice, ii, jj_1, jj_2, p_i=0
	cdef int[:] ch_1 = np.empty(s,dtype=np.intc), ch_2 = np.empty(s,dtype=np.intc)
	cdef int[:] empty = np.ones(s,dtype=np.intc)*-1
	cdef int[:,:] pop = np.empty((m,s), dtype=np.intc)
	for i,j in pairs:
		ch_1[:],ch_2[:] = empty, empty
		pivot_1,pivot_2 = floyd_sampler(s,2)
		if pivot_1 > pivot_2:
			pivot_1,pivot_2 = pivot_2,pivot_1
		choice = np.random.randint(2)
		if choice:
			ch_1[pivot_1:pivot_2+1] = par[i][pivot_1:pivot_2+1]
			ch_2[pivot_1:pivot_2+1] = par[j][pivot_1:pivot_2+1]
			pivot_2 += 1
			jj_1 = jj_2 = pivot_2
			for ii in range(pivot_2,s+pivot_2+1):
				if par[j][ii % s] not in ch_1:
					ch_1[jj_1 % s] = par[j][ii % s]
					jj_1 += 1
				if par[i][ii % s] not in ch_2:
					ch_2[jj_2 % s] = par[i][ii % s]
					jj_2 += 1
		else:
			ch_1[pivot_1:pivot_2+1] = par[j][pivot_1:pivot_2+1]
			ch_2[pivot_1:pivot_2+1] = par[i][pivot_1:pivot_2+1]
			pivot_2 += 1
			jj_1 = jj_2 = pivot_2
			for ii in range(pivot_2,s+pivot_2+1):
				if par[i][ii % s] not in ch_1:
					ch_1[jj_1 % s] = par[i][ii % s]
					jj_1 += 1
				if par[j][ii % s] not in ch_2:
					ch_2[jj_2 % s] = par[j][ii % s]
					jj_2 += 1

		pop[p_i][:], pop[p_i+1][:] = ch_1,ch_2
		p_i += 2

	return pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray bit_flip(np.ndarray pop, float prop=.1):
	'''
		entry:
			pop == population
			prop == portion that will suffer mutation
		exits:
			pop w/ mutated individuos

		comments:
			as default, approximately 10% of the
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] swap(int[:,:] pop, float prop=.1):#confeir!!!!
	'''
		entry:
			pop == population
			prop == portion that will suffer mutation
		exits:
			pop w/ mutated individuos

		comments:
			as default, approximately 10% of the
			population will suffer mutation. At
			least one will suffer mutationl.
	'''
	cdef int n = pop.shape[0]
	cdef int s = pop.shape[1]
	cdef int[:] idxs = floyd_sampler(n,int(np.ceil(n*prop)))

	cdef int i, x, y
	for i in idxs:
		x,y = floyd_sampler(2, s)
		pop[i][x], pop[i][y] = pop[i][y], pop[i][x]

	return pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] rank_sel(int[:,:] pop, int k, int q_type=0):##trocar rank por torneio
	'''
		entries:
			pop == populacao
			k == numb. of selected ind.
		exit:
			par == parents
	'''
	cdef int[:] i
	if q_type == 1:
		fitness = [eval_diags(decode_tab(i)) for i in pop]
	else:
		fitness = [eval_table(conv_tab(i)) for i in pop]

	sort_idx = np.argsort(fitness).tolist()			#order by fitness
	cdef int[:,:] par = np.empty((k,s), dtype=np.intc)
	cdef int ii,j
	for ii,j in enumerate(sort_idx[:k]):	#select parents
		par[ii][:] = pop[j]

	return par

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray meme(np.ndarray pop):
	cdef np.ndarray aux
	cdef np.ndarray o_pop = np.zeros((pop.shape[0],pop.shape[1]), dtype=np.intc)
	for i,aux in enumerate(pop):
		aux = conv_tab(aux)
		aux[:] = mov_comp(aux)
		aux[:] = rem_queens(aux)
		o_pop[i] = conv_tab(aux)
	return o_pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] meme_2(int[:,:] pop, int n_max):
	cdef int[:] aux
	cdef int[:,:] o_pop = np.empty((pop.shape[0],pop.shape[1]), dtype=np.intc)
	cdef int i,j = pop.shape[1]
	cdef int[:] n_blk = [i for i in range(2,n_max+2) if j % i == 0]
	for i,aux in enumerate(pop):
		for j in n_blk:
			aux[:] = swp_diags(aux, j)
		o_pop[i][:] = aux
	return o_pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef (int, vector[int], vector[float], vector[ vector[int] ]) sga(int N_queens, int pop_size, int n_par, int n_epochs=10000, int u_meme=0):
	cdef int epoch=0,i
	cdef vector[int] J_best = np.zeros(n_epochs, dtype=int)
	cdef vector[float] J_med = np.zeros(n_epochs)
	cdef vector[ vector[int] ] sol
	cdef np.ndarray pop = np.array(
		[conv_tab(init_table(N_queens)) for i in range(pop_size)]
	)

	cdef np.ndarray par, res = np.zeros(pop_size, dtype=np.intc)
	cdef np.ndarray ind, b_res = np.zeros(N_queens*N_queens, dtype=np.intc)
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

	return epoch, J_best[:epoch+1], J_med[:epoch+1], sol
