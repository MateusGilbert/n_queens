# distutils: language=c++

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport cython

np.import_array()

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:] floyd_sampler(int N, int M):
	'''
		Robert Floyd's sampler
		--Reference--
			Title: Programing Pearls: A Sample of Brilliance
			Author: Bantley, J. and Floyd, R.
			Jornal: Communications of the ACM
			Year: 1987
	'''
	#cdef np.ndarray samples = np.zeros(M, dtype=np.intc)
	cdef int[:] samples = np.zeros(M, dtype=np.intc)
	cdef int i, sample
	if N-M == 0:
		return np.random.permutation(N, dtype=np.intc)
	for i in range(N-M,N):
		sample = np.random.randint(i, dtype=np.intc)
		samples[i - N + M] = i if sample in samples else sample
	return samples

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray init_table(int N):
	cdef int N_table = N
	cdef np.ndarray table = np.zeros([N_table,N_table], dtype=np.intc)
	cdef int i,j
	cdef np.ndarray pos = floyd_sampler(N_table*N_table, N)
	cdef int col, row
	for i in pos:
		col = i % N_table
		row = i // N_table
		table[row,col] = 1
	return table


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int eval_diags(int[:,:] table):
	#search diagonals
	cdef int N_table = table.shape[0]
	cdef int i,j,aux,aux_2,res=0
	for i in range(N_table-1):
		aux,aux_2 = 0,0
		for j in range(N_table-i):
			aux += table[j,j+i]
			aux_2 += table[N_table-1-j,i+j]
		if (aux > 1):
			res += aux - 1
		if (aux_2 > 1):
			res += aux_2 - 1
	for i in range(1,N_table-1):
		aux,aux_2 = 0,0
		for j in range(i+1):
			aux += table[i-j,j]
			aux_2 += table[N_table-1-i+j,j]
		if (aux > 1):
			res += aux - 1
		if (aux_2 > 1):
			res += aux_2 - 1
	return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int eval_table(np.ndarray table):
	cdef int conv=0
	cdef int N_table = table.shape[0];
	cdef np.ndarray v1 = np.ones(N_table, dtype=np.intc)
	cdef int res = 0, aux
	cdef np.ndarray vec

	#search rows
	for vec in table:
		aux = vec @ v1
		if (aux > 1):
			res += aux - 1
	#search columns
	for vec in table.T:
		aux = vec @ v1
		if (aux > 1):
			res += aux - 1

	#search diagonals
	cdef int i,j,aux_2
	for i in range(N_table-1):
		aux,aux_2 = 0,0
		for j in range(N_table-i):
			aux += table[j,j+i]
			aux_2 += table[N_table-1-j,i+j]
		if (aux > 1):
			res += aux - 1
		if (aux_2 > 1):
			res += aux_2 - 1
	for i in range(1,N_table-1):
		aux,aux_2 = 0,0
		for j in range(i+1):
			aux += table[i-j,j]
			aux_2 += table[N_table-1-i+j,j]
		if (aux > 1):
			res += aux - 1
		if (aux_2 > 1):
			res += aux_2 - 1

	#count the numb. of missing queens
	cdef int count = np.sum(table)
	res += 10*abs(N_table - count)  #constant 10 strongly penalizes solutions with less then
											#the required number of queens

	return res

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray conv_tab(np.ndarray table):
	if table.ndim > 1:
		return table.reshape(-1,1).squeeze()
	cdef int N_table = int(np.sqrt(table.shape[0]))
	return table.reshape(N_table,N_table)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:,:] decode_tab(int[:] code):										#code = each position indicates a row, store value is the column
	cdef int N_table = code.shape[0], i,j
	cdef int[:,:] table = np.zeros((N_table,N_table), dtype=np.intc)
	for i,j in enumerate(code):
		table[i,j] = 1
	return table

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:] encode_tab(int[:,:] table):
	cdef int N_table = table.shape[0], i,j
	cdef int[:] code = np.zeros(N_table, dtype=np.intc)
	cdef int[:] v = np.arange(N_table, dtype=np.intc)
	for i,row in enumerate(table):
		code[i] = np.matmul(v,row)
	return code

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray mov_comp(np.ndarray table):
	cdef int N_table = table.shape[0]
	cdef np.ndarray x,y,aux_table = table.copy(), res_table = table.copy()
	cdef int fitness = eval_table(res_table)
	cdef np.ndarray mov = np.array(
		[
			[0,1],		#right
			[1,1],		#down-right
			[1,0],		#down
			[1,-1],		#down-left
			[0,-1],		#left
			[-1,-1],		#up-left
			[-1,0],		#up
			[-1,1]		#up-right
		],
		dtype=np.intc
	)

	cdef int i,j,r,c,aux_fit,n_mov = mov.shape[0]
	x,y = np.where(table==1)
	cdef np.ndarray b_mov = np.zeros_like(mov)
	for i,j in zip(x,y):
		aux_table = res_table.copy()
		aux_table[i,j] = 0
		b_mov[:] = np.tile([i,j], (n_mov,1)) + mov
		b_mov[b_mov == N_table] = 0
		b_mov[b_mov == -1] = N_table-1
		for pos in b_mov:
			r,c = pos
			if aux_table[r,c] != 1:
				aux_table[r,c] = 1
				aux_fit = eval_table(aux_table)
				if aux_fit < fitness:
					res_table = aux_table.copy()
					fitness = aux_fit
				aux_table[r,c] = 0
	return res_table


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef int[:] swp_diags(int[:] og_ar, int n_sbar):
	cdef int i, size=og_ar.shape[0], q = eval_diags(decode_tab(og_ar)), q_aux
	cdef int[:] aux = og_ar.copy(), res_ar = og_ar.copy()
	
	for i in range(0,size,n_sbar):
		aux[i:i+n_sbar],aux[i+n_sbar:i+2*n_sbar] = aux[i+n_sbar:i+2*n_sbar],aux[i:i+n_sbar]
		q_aux = eval_diags(decode_tab(aux))
		if q > q_aux:
			res_ar[:] = aux.copy()
			q = q_aux
		else:
			aux[:] = res_ar.copy()

	return res_ar

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef np.ndarray rem_queens(np.ndarray table):
	cdef int count, N_queens = table.shape[0]
	count = np.sum(table)
	if count == N_queens:
		return table

	cdef np.ndarray x,y
	(x,y) = np.where(table == 1)
	cdef int i, j
	#cdef vector[int] x=q_pos[0], y=q_pos[1]
	cdef int old_fit
	if count > N_queens:
		for i,j in zip(x,y):
			old_fit = eval_table(table)
			table[i,j] = 0
			if old_fit - eval_table(table) < 2:
				table[i,j] = 1
			else:
				count -= 1
			if count == N_queens:
				break

	return table
