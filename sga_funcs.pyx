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
cpdef int[:,:] one_point_xover(int[:,:] par, int m):
#cpdef np.ndarray one_point_xover(np.ndarray par, int m):
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
	cdef int[:] idxs = floyd_sampler(n,min(n,m)).astype(np.intc)

	while (m > idxs.shape[0]):
		idxs = np.concatenate([idxs, floyd_sampler(n, min(m-idxs.shape[0],n).astype(np.intc))])
	cdef int i,j
	cdef int[:] pairs = np.array([[i,j] for i,j in zip(idxs[::2],idxs[1::2])], dtype=np.intc)

	cdef pivot
	cdef int[:] ch_1, ch_2
	cdef int[:,:] pop = np.empty((m,s), dtype=np.intc)
	#pop = list()
	for i,j in pairs:
		pivot = np.random.randint(s-1)
		ch_1 = np.concatenate([par[i][:pivot], par[j][pivot:]])
		ch_2 = np.concatenate([par[j][:pivot], par[i][pivot:]])
		pop[i] = ch_1
		pop[j] = ch_2
		#pop.extend([ch_1,ch_2])

	#return np.array(pop)
	return pop

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
#cpdef np.ndarray bit_flip(np.ndarray pop, float prop=.1):
cpdef int[:,:] bit_flip(int[:,:] pop, float prop=.1):
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
	cdef int[:] idxs = floyd_sampler(n,int(np.ceil(n*prop))).astype(np.intc)

	cdef int i, pos
	for i in idxs:
		pos = np.random.randint(s)
		pop[i][pos] = 0 if pop[i][pos] == 1 else 1

	return pop


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
		x,y = floyd_sampler(s, 2)
		pop[i][x], pop[i][y] = pop[i][y], pop[i][x]

	return pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] tor_sel(int[:,:] pop, int mu, int k, int q_type=0):
	'''
		entries:
			pop == populacao
			k == numb. of selected ind.
		exit:
			par == parents
	'''
	cdef int[:] i, idxs, qualities
	if q_type == 1:
		qualities = np.array([eval_diags(decode_tab(i)) for i in pop], dtype=np.intc)
	else:
		qualities = np.array([eval_table(conv_tab(i)) for i in pop], dtype=np.intc)

	cdef int cur_ind = 0, size=pop.shape[0], N_queens=pop.shape[1]
	cdef int q_cur, j, add_idx
	cdef int[:,:] par = np.empty((mu,N_queens), dtype=np.intc)
	while (cur_ind < mu):
		q_cur = 10*N_queens
		add_idx=-1
		idxs = floyd_sampler(size, k)
		for j in idxs:
			if qualities[j] < q_cur:
				q_cur = qualities[j]
				add_idx = j
		par[cur_ind] = pop[add_idx]
		cur_ind += 1

	return par

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] meme(int[:,:] pop):
#cpdef np.ndarray meme(np.ndarray pop):
	cdef int[:] aux
	cdef int[:,:] o_pop = np.zeros((pop.shape[0],pop.shape[1]), dtype=np.intc)
	for i,aux in enumerate(pop):
		aux = conv_tab(aux)
		aux = mov_comp(aux)
		aux = rem_queens(aux)
		o_pop[i] = conv_tab(aux)
	return o_pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] meme_2(int[:,:] pop):
	cdef int[:] aux
	cdef int[:,:] o_pop = np.empty((pop.shape[0],pop.shape[1]), dtype=np.intc)
	cdef int i,j = pop.shape[1]
	for i,aux in enumerate(pop):
		aux = cocktail_swp(aux)										#swap rows
		aux = change_enc(cocktail_swp(change_enc(aux)))		#swap columns
		o_pop[i] = aux
	return o_pop

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef (int, vector[int], vector[float], vector[ vector[int] ]) sga(int N_queens, int pop_size, int n_par, int n_epochs=10000, int u_meme=0):
	cdef int epoch=0,i
	cdef vector[ int ] J_best = np.zeros(n_epochs, dtype=np.intc)
	cdef vector [float] J_med = np.zeros(n_epochs, dtype=np.single)
	cdef int[:,:] pop = np.array(
		[conv_tab(init_table(N_queens)) for i in range(pop_size)], dtype=np.intc
	)

	cdef int[:] ind, res = np.zeros(pop_size, dtype=np.intc), b_res = np.zeros(N_queens*N_queens, dtype=np.intc)
	cdef int[:,:] par
	cdef int b_fit=N_queens*100, k = 5 if pop_size < 1000 else 10
	cdef float mean_fit
	cdef int solved=0
	while(epoch<n_epochs):
		par = tor_sel(pop, n_par, k)
		pop = one_point_xover(par, pop_size)
		pop = bit_flip(pop)
		if u_meme > 0:
			pop = meme(pop)

		#evaluate epoch results
		for i,ind in enumerate(pop):
			res[i] = eval_table(conv_tab(ind))
			if res[i] < b_fit:
				b_res = pop[i].copy()
				b_fit = res[i]
		J_best[epoch] = b_fit
		J_med[epoch] = np.mean(res)

		for i in res:
			if i == 0:
				solved=1
				break

		epoch += 1
		if solved==1:
			break

	cdef int[:] cand_res = np.where(np.array(res) == b_fit)[0].astype(np.intc)
	cdef vector[ vector [int] ] sol# = np.empty((cand_res.shape[0], N_queens*N_queens), dtype=np.intc) #conferir
	for i in cand_res:
		sol.push_back(np.array(pop[i]))

	return epoch, J_best[:epoch+1], J_med[:epoch+1], sol


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef (int, vector[int], vector[float], vector[ vector[int] ]) sga_2(int N_queens, int pop_size, int n_par, int n_epochs=10000, int u_meme=0):
	cdef int epoch=0,i
	cdef vector[ int ] J_best = np.zeros(n_epochs, dtype=np.intc)
	cdef vector [float] J_med = np.zeros(n_epochs, dtype=np.single)
	cdef int[:,:] pop = np.array(
		[np.random.permutation(N_queens).astype(np.intc) for i in range(pop_size)]#, dtype=np.intc
	)

	cdef int[:] ind, res = np.zeros(pop_size, dtype=np.intc), b_res = np.zeros(N_queens*N_queens, dtype=np.intc)
	cdef int[:,:] par
	cdef int b_fit=N_queens*100, k = 5 if pop_size < 1000 else 10
	cdef float mean_fit
	while(epoch<n_epochs):
		par = tor_sel(pop, n_par, k, q_type=1)
		pop = order_xover(par, pop_size)
		pop = swap(pop,prop=.2)
		if u_meme > 0:
			pop = meme_2(pop)

		#evaluate epoch results
		for i,ind in enumerate(pop):
			res[i] = eval_diags(decode_tab(ind))
			if res[i] < b_fit:
				b_res = pop[i].copy()
				b_fit = res[i]
		J_best[epoch] = b_fit
		J_med[epoch] = np.mean(res)

		for i in res:
			if i == 0:
				break

		epoch += 1
		if solved==1:
			break

	cdef int[:] cand_res = np.where(np.array(res) == b_fit)[0].astype(np.intc)
	cdef vector[ vector [int] ] sol
	for i in cand_res:
		sol.push_back(np.array(pop[i]))

	return epoch, J_best[:epoch+1], J_med[:epoch+1], sol
