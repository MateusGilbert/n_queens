import numpy as np
cimport numpy as np

np.import_array()

cpdef np.ndarray floyd_sampler(int N, int M):
	'''
		Robert Floyd's sampler
		--Reference--
			Title: Programing Pearls: A Sample of Brilliance
			Author: Bantley, J. and Floyd, R.
			Jornal: Communications of the ACM
			Year: 1987
	'''
	cdef np.ndarray samples = np.zeros(M, dtype=np.int16)
	cdef int i, sample
	if N-M == 0:
		return np.random.permutation(N)
	for i in range(N-M,N):
		sample = np.random.randint(i)
		samples[i - N + M] = i if sample in samples else sample
	return samples

cpdef np.ndarray init_table(int N):
	cdef int N_table = N
	cdef np.ndarray table = np.zeros([N_table,N_table], dtype=np.int16)
	cdef int i,j
	cdef np.ndarray pos = floyd_sampler(N_table*N_table, N)
	cdef int col, row
	for i in pos:
		col = i % N_table
		row = i // N_table
		table[row,col] = 1
	return table

cpdef int eval_table(np.ndarray table):
	cdef int conv=0
#	if table.ndim < 2:
#		table = conv_tab(table)
#		conv = 1
	cdef int N_table = table.shape[0];
	cdef np.ndarray v1 = np.ones(N_table, dtype=np.int16)
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

#	if conv == 1:
#		table = conv_tab(table)

	return res

cpdef np.ndarray conv_tab(np.ndarray table):
	if table.ndim > 1:
		return table.reshape(-1,1).squeeze()
	cdef int N_table = int(np.sqrt(table.shape[0]))
	return table.reshape(N_table,N_table)

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
		dtype=np.int16
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

cpdef np.ndarray rem_queens(np.ndarray table):
	cdef int count, N_queens = table.shape[0]
	count = np.sum(table)
	if count == N_queens:
		return table

	cdef np.ndarray q_pos = np.where(table == 1)
	cdef int i, j, x=q_pos[0], y=q_pos[1]
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
