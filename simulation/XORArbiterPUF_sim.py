import os

import numpy as np
from pypuf.io import random_inputs
from pypuf.simulation import XORArbiterPUF

n_bits = 32
n_challenges = 100000
k = 4

data_folder_name = f'{n_bits}bit_{k}XOR_{n_challenges // 1000}k'

c_folder = f'../data/challenges/{data_folder_name}'
if not os.path.exists(c_folder):
    os.mkdir(c_folder)

r_folder = f'../data/responses/{data_folder_name}'
if not os.path.exists(r_folder):
    os.mkdir(r_folder)

for seed in range(10):
    puf = XORArbiterPUF(n=n_bits, k=k, seed=seed, noisiness=0)
    challenge = random_inputs(n=n_bits, N=n_challenges, seed=seed)
    response = puf.eval(challenge)

    np.save(f'../data/challenges/{data_folder_name}/s{seed}', challenge)
    np.save(f'../data/responses/{data_folder_name}/s{seed}', response)
