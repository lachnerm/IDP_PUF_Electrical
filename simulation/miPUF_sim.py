import os
from functools import partial

import joblib
import numpy as np
from pypuf.io import random_inputs
from pypuf.simulation import XORArbiterPUF
from simulation.miPUF_sim_original import miPUF_eval

n_bits = 8
m = 2
k = 2
noise = 0
n_challenges = 100000
start = 1
bl = 2

data_folder_name = f'{n_bits}bit_{k}XOR_{n_challenges // 1000}k'
data_folder_name = f'{k}XOR_{n_bits}bit_basics0bl2_{n_challenges}'

c_folder = f'../data_miPUF/challenges/{data_folder_name}'
if not os.path.exists(c_folder):
    os.mkdir(c_folder)

r_folder = f'../data_miPUF/responses/{data_folder_name}'
if not os.path.exists(r_folder):
    os.mkdir(r_folder)

for seed in range(10):
    """ Prepare functions to create the PUF instances """
    gen_ArbPUF = partial(XORArbiterPUF, n=n_bits, k=1, noisiness=noise)
    gen_t_XOR_ArbPUF = partial(XORArbiterPUF, n=n_bits, k=k, noisiness=noise)

    """ Use seeds for reproducibility, etc. """
    l_of_Arb_PUFs = [gen_ArbPUF(seed=seed + i) for i in range(m)]
    t_XORArbPuf = gen_t_XOR_ArbPUF(seed=5)

    miPUF_instance = (l_of_Arb_PUFs, t_XORArbPuf)

    """ Create challenges to be used with seed """
    Challenges_C = random_inputs(n=n_bits, N=n_challenges, seed=seed)

    """ If more than CPU cores shall be used to evaluate the m Arbiter PUFs
        select desired number.
    """
    # Get no. of cores available
    no_cpu = joblib.cpu_count()
    # Leave 3 cores for other tasks free:
    parallel_jobs = max(1, no_cpu - 3)

    """ Now evaluate this miPUF as desired
    """

    """ Basic: First block starting at bit 1, then m=4 blocks of length 16 (all challenge bits affected by XORing)

    result_basic_s0_bl16 = miPUF_eval(miPUF_instance, Challenges_C, 'basic',
                                      start=1, block_length=16,
                                      n_jobs=parallel_jobs)
    """
    """ Basic: First block starting at bit 1, then m=4 blocks of length 8 (32 last challenge bits not affected by XORing)
    """
    '''result_basic_s0_bl8 = miPUF_eval(miPUF_instance, Challenges_C, 'basic', start=1,
                                     block_length=8, n_jobs=parallel_jobs)'''

    result_basic_s0_bl2 = miPUF_eval(miPUF_instance, Challenges_C, 'basic',
                                     start=start,
                                     block_length=bl, n_jobs=parallel_jobs)

    """ Basic: First block starting at bit 5, then m=4 blocks of length 12 (first 4 and 12 last challenge bits not affected by XORing)

    result_basic_s5_bl12 = miPUF_eval(miPUF_instance, Challenges_C, 'basic',
                                      start=5, block_length=12,
                                      n_jobs=parallel_jobs)
    """
    """ Rainbow: Since (n_bits=64) % (m=4) = 0 all bits will be equally affected by cyclic XORing
    """
    """result_rainbow = miPUF_eval(miPUF_instance, Challenges_C, 'rainbow',
                                n_jobs=parallel_jobs)"""

    """ Pyramid: Make sure (m=4) * seg_length <= (n_bits/2=32). Otherwise the Code will not produce valid results!

        Note that it is not possible with even challenge lengths to produce a top that is equally large to a pyramid step size, i.e.
        the seg_length.
        However if here for instance seg_length = 8 is chosen the 16 middle bits will all be affected by XORing with R_1 to R_4;
        this is the same amount as when considering both the left and right steps of the pyramid
        (i.e. bits 1-8 and 57-64 for XORing with R_1).

        Always consider this when using the pyramid scheme.
    """

    """ Pyramid: step size seg_length = 8

    result_pyramid_seg_8 = miPUF_eval(miPUF_instance, Challenges_C, 'pyramid',
                                      seg_length=8, n_jobs=parallel_jobs)
    """
    """ Indi: Use individual XOR choices
    """

    """ To reproduce result_basic_s5_bl12 build the following Ordering 

    before_1st_block = np.zeros((m, 4))
    block_1 = np.zeros((m, 12))
    block_1[0, :] = 1
    block_2 = np.zeros((m, 12))
    block_2[1, :] = 1
    block_3 = np.zeros((m, 12))
    block_3[2, :] = 1
    block_4 = np.zeros((m, 12))
    block_4[3, :] = 1
    after_last_block = np.zeros((m, 12))

    Ordering = np.concatenate(
        (before_1st_block, block_1, block_2, block_3, block_4, after_last_block),
        axis=1)
    """
    # result_indi_order = miPUF_eval(miPUF_instance, Challenges_C, 'indi', Ordering=Ordering, n_jobs=parallel_jobs)

    """ Check the results are the same """
    # print(all(result_indi_order == result_basic_s5_bl12))

    response = (result_basic_s0_bl2 + 1) / 2
    challenge = Challenges_C

    np.save(f'../data_miPUF/challenges/{data_folder_name}/s{seed}', challenge)
    np.save(f'../data_miPUF/responses/{data_folder_name}/s{seed}', response)
