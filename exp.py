#! /usr/bin/python3

import pandas as pd
from _exp import exp
from os.path import isfile


N_queens_list = [64,128]#[8, 16, 32, 64]#, 128, 256, 512]#, 1024, 2048, 4096]
pop_sizes = 200
prop_par = .3
epochs = 400
memes = [1, 25, 50, 100, 200, 300]
filename='results_4.csv'

for N_queens in N_queens_list:
	for meme in memes:
		res = exp(N_queens, pop_sizes, prop_par, epochs, meme, rounds=10)
		df = pd.DataFrame(res)

		df.to_csv(filename, mode='a', index=False, header=not isfile(filename))
