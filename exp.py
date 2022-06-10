#! /usr/bin/python3

import pandas as pd
from _exp import exp
from os.path import isfile


N_queens_list = [8, 16, 32, 64, 128]
pop_sizes = 100
prop_par = .3
epochs = 100
meme= 0
filename='results.csv'

for N_queens in N_queens_list:
	res = exp(N_queens, pop_sizes, prop_par, epochs, meme, rounds=100)
	df = pd.DataFrame(res)

	df.to_csv(filename, mode='a', index=False, header=not isfile(filename))
