#! /usr/bin/python3

import pandas as pd
from _grid_search import grid_search


N_queens = 8
pop_sizes = [10**i for i in range(1,6)]
prop_par = [i*.1 for i in range(2,10)]
epochs = [50*i for i in [1, 2, 10, 20, 100, 200]]

res = grid_search(N_queens, pop_sizes, prop_par, epochs)

df = pd.DataFrame(res)
df.to_csv(f'gs_0.csv', index=False)
