import joblib
from joblib import Parallel, delayed
from functools import partial

import numpy as np

from pypuf.simulation import XORArbiterPUF
from pypuf.io import random_inputs

""" These function implement the (m,t)-miPUF in a rudimentary manner
    from pypuf using m ArbiterPUFs and one t-XORArbiterPUF, providing
    XOR-schemes Basic, Rainbo and Pyramid.
"""

""" The following functions provide the core features required for the 
    miPUF simulation
"""

""" Evaluate a PUF instance from pypuf on challenges provided """


def eval_puf(instance, challenges):
    return instance.eval(challenges)


""" If a,b \in {0,1} then 
    a XOR b = a + b mod 2. 
    NOTE:
    Pypuf uses a,b \in {-1,1} with the transformation
    1->-1 AND 0->1 hence the operation to use is:
    a XOR b = a * b
"""

""" Use the Basic scheme to modify the Global PUF-Challenge.

    Challenges:     numpy.ndarray   - Shape (n, nbits); n Global Challenges of length nbits
    Resps_R_m:      numpy.ndarray   - Shape (n, m); the n Responses to the Global Challenges for each of the m ArbiterPUFs
    start:          int             - Starting position of the first block (0 if starting at the first bit)
    block_length:   int             - Length of the individual blocks where R_m is XORed with the Global Challenges

    mod_Challenges: numpy.ndarray   - Shape (n, nbits); modified Global Challenges obtained from blockwise XORing
"""


def basic_variant(Challenges, Resps_R_m, start, block_length):
    m = Resps_R_m.shape[1]
    challenge_length = Challenges.shape[1]

    """ Produce a modification matrix for XORing by subsequent elementwise
        multiplication. Hence it will contain the bits mod_b.

        First produce "start" number of columns consisting of (1). These will
        be unaltered when XORing since for challenge bit c:
        c XOR mod_b = c XOR (1) = c * (1) = c

        Then produce m blocks of length "block_length" consisting of each of the 
        columns of Resps_R_m. These are the blocks of responses R_1 ... R_m.

        Finally produce a block of columns consisting of (1) to fill the trailing
        bits and retaing the initial challenge unaltered there.

        Example: m = 5, nbits = 64, start = 4, block_length = 10,
            start block of 4 columns (1) 
            + 5 blocks of 10 columns R_m each
            + end block of 10 columns (1)
            = Total of 64 columns
    """
    mod_m_start = np.ones((Challenges.shape[0], start))
    mod_m_end = np.ones(
        (Challenges.shape[0], challenge_length - (start + m * block_length)))

    mod_m = mod_m_start

    for i in range(m):
        block = Resps_R_m[:, i].reshape((Resps_R_m.shape[0], 1)) * np.ones(
            block_length)
        mod_m = np.concatenate((mod_m, block), axis=1)

    mod_m = np.concatenate((mod_m, mod_m_end), axis=1)

    """ Elementwise multiplication of matrices """
    mod_Challenges = np.multiply(Challenges, mod_m)

    return mod_Challenges


""" Use the Rainbow scheme to modify the Global PUF-Challenge.

    Challenges:     numpy.ndarray   - Shape (n, nbits); n Global Challenges of length nbits
    Resps_R_m:      numpy.ndarray   - Shape (n, m); the n Responses to the Global Challenges for each of the m ArbiterPUFs

    mod_Challenges: numpy.ndarray   - Shape (n, nbits); modified Global Challenges obtained from rainbow XORing
"""


def rainbow_variant(Challenges, Resps_R_m):
    m = Resps_R_m.shape[1]
    challenge_length = Challenges.shape[1]

    """ Produce a modification matrix for XORing by subsequent elementwise
        multiplication. Hence it will contain the bits mod_b.

        The columns of the modification matrix are simply the columns of
        Resps_R_m in a cycle until "challenge_length" is exhausted.
        [Depending on m and "challenge_lenth" the individual R_i may be not used
         equally often.]

        Example: m = 5, nbits = 64,
            12 blocks, each equal to Resp_R_m
            + 1 block of 4 columns consisting of R_1...R_4
    """
    mod_m_start = np.ones((Challenges.shape[0], 0))

    mod_m = mod_m_start

    no_blocks, remaining_cols = divmod(challenge_length, m)
    for _ in range(no_blocks):
        mod_m = np.concatenate((mod_m, Resps_R_m), axis=1)
    for i in range(remaining_cols):
        col = Resps_R_m[:, i % m].reshape((Resps_R_m.shape[0], 1))
        mod_m = np.concatenate((mod_m, col), axis=1)

    """ Elementwise multiplication of matrices """
    mod_Challenges = np.multiply(Challenges, mod_m)

    return mod_Challenges


""" Use the Pyramid scheme to modify the Global PUF-Challenge.

    Challenges:     numpy.ndarray   - Shape (n, nbits); n Global Challenges of length nbits
    Resps_R_m:      numpy.ndarray   - Shape (n, m); the n Responses to the Global Challenges for each of the m ArbiterPUFs
    seg_length:     int             - Length after which an additional R_{i+1} is XORed on top of the existing R_1...R_i

    mod_Challenges: numpy.ndarray   - Shape (n, nbits); modified Global Challenges obtained from pyramid XORing
"""


def pyramid_variant(Challenges, Resps_R_m, seg_length):
    m = Resps_R_m.shape[1]
    challenge_length = Challenges.shape[1]

    """ Produce a modification matrix for XORing by subsequent elementwise
        multiplication. Hence it will contain the bits mod_b.

        The construction is done iteratively as follows:
        1st: A matrix M of shape (n, challenge_length) consisting of columns of R_1 is produced.
        2nd: The submatrix from column "seg_length" to column "challenge_length - seq_length"
                is then XORed columnwise with R_2
        3rd: The same is applied to the columns from "2 * seg_length" to "challenge_length - 2 * seq_length"
        ...
        mth: The same is applied to the columns from "(m-1) * seg_length" to "challenge_length - (m-1) * seq_length"

        NOTE1: YOU MUST MAKE SURE "m * seg_length <= challenge_length / 2 " SINCE OTHERWISE THE RESULT WILL BE FLAWED

        NOTE2: This procedure MAY NOT PRODUCE AN EQUIDISTANT PYRAMID if "(2m-1) * seg_length != challenge_length".
                All steps of the pyramid will be the same size, however the top plateau may be larger
                and since these are the middle bits it may have a greater impact depending on the architecture.
                The top plateau may very well have the same size as the steps left and right of it combined!

        Example: m = 4, nbits = 64, seg_length = 7 [pythonic counting 0-63]
            Col 00-06, 57-63:   R_1
            Col 07-13, 50-56:   R_1 XOR R_2
            Col 14-20, 43-49:   R_1 XOR R_2 XOR R_3
            Col 21   -    42:   R_1 XOR R_2 XOR R_3  XOR R_4

            HENCE R_4 IMPACTS MORE BITS IN THE MIDDLE THAN THE OTHERS

            IF seg_length = 8 were chosen still double the bits in the middle would be impacted.

        Also note that since 2m-1 will be uneven and hence no 2^x will suffice for an equidistant pyramid in the 
        middle. This must always be considered.
    """
    mod_m_start = np.ones((Challenges.shape[0], challenge_length))

    mod_m = mod_m_start
    for i in range(m):
        R_i = Resps_R_m[:, i].reshape((Resps_R_m.shape[0], 1))
        R_i_mat = R_i * np.ones(challenge_length - i * 2 * seg_length)

        """ Elementwise multiplication of matrices """
        mod_m[:,
        i * seg_length:challenge_length - i * seg_length] = np.multiply(
            mod_m[:, i * seg_length:challenge_length - i * seg_length], R_i_mat)

    """ Elementwise multiplication of matrices """
    mod_Challenges = np.multiply(Challenges, mod_m)

    return mod_Challenges


""" Use the Rainbow scheme to modify the Global PUF-Challenge.

    Challenges:     numpy.ndarray   - Shape (n, nbits); n Global Challenges of length nbits
    Resps_R_m:      numpy.ndarray   - Shape (n, m); the n Responses to the Global Challenges for each of the m ArbiterPUFs
    Ordering:       numpy.ndarray   - Shape (m, nbits); Selection if ArbiterPUF i's response is XORed with bit j or not. [0 or 1]

    mod_Challenges: numpy.ndarray   - Shape (n, nbits); modified Global Challenges obtained from rainbow XORing
"""


def indi_order_variant(Challenges, Resps_R_m, Ordering):
    m = Resps_R_m.shape[1]
    challenge_length = Challenges.shape[1]

    """ Produce a modification matrix for XORing by subsequent elementwise
        multiplication. Hence it will contain the bits mod_b.

        First initialise the modification matrix with (1) values.

        If a column is not affected these (1)s will retain their values when creating the modified challenges.

        If a column of (1)s is XORed with R_i for the first time it will produce R_i.

        Further XORing of columns will produce the expected XOR result.

        To this end now go through all m Arbiter PUFs. For each Arbiter PUF i
        go through the respectively provided Ordering of values j 
        and XOR if the value a_{ij} == 1.

        Finally produce the altered Global Challenges.
    """
    mod_m_start = np.ones((Challenges.shape[0], challenge_length))

    mod_m = mod_m_start
    for i in range(m):
        R_i = Resps_R_m[:, i]

        for j, val in enumerate(Ordering[i, :]):
            if val == 1:
                """ Elementwise multiplication of matrices """
                mod_m[:, j] = np.multiply(mod_m[:, j], R_i)

    """ Elementwise multiplication of matrices """
    mod_Challenges = np.multiply(Challenges, mod_m)

    return mod_Challenges


""" Function to evaluate a constructed miPUF.

    miPUF:          tuple (list of pypuf objects, pypuf object)
                                    - (list of the m Arbiter PUFs, the t-XORArbiter PUF) used for the miPUF
    Challenges:     numpy.ndarray   - Shape (n, nbits); the n challenges to evaluate the miPUF
    variant:        string          - A string to select the variant for XORing [Basic, Rainbow, Pyramid or inidi]
    start:          int             - For Basic variant: start position of the blocks [natural counting]
    block_length:   int             - For Basic variant: length of the blocks
    seg_length:     int             - For Pyramid variant: length of the segments
    Ordering:       numpy.ndarray   - For indi_order_variant: Shape (m, nbits) Choose when to XOR for each Arbiter PUF and bit individually.
    n_jobs:         int             - Number of CPU cores used simultaneously to evaluate the m Arbiter PUFs

    response:       numpy.array     - Array of the n responses produced by the miPUF
"""


def miPUF_eval(miPUF, Challenges, variant, start=1, block_length=1,
               seg_length=1, Ordering=np.empty((0, 0)), n_jobs=1):
    l_Arbiter_PUFs = miPUF[0]
    t_XOR_Arbiter_PUF = miPUF[1]

    """ Evaluate all m ArbPUFs on the challenges provided """
    m_Arb_PUFs_evaled = Parallel(n_jobs=n_jobs)(
        delayed(eval_puf)(instance, Challenges) for instance in l_Arbiter_PUFs)
    m_Arb_PUFs_evaled = np.array(m_Arb_PUFs_evaled).transpose()

    if variant == 'basic':
        mod_Challenges = basic_variant(Challenges, m_Arb_PUFs_evaled,
                                       start=(start - 1),
                                       block_length=block_length)  # Adapt for pythonic counting

    elif variant == 'rainbow':
        mod_Challenges = rainbow_variant(Challenges, m_Arb_PUFs_evaled)

    elif variant == 'pyramid':
        mod_Challenges = pyramid_variant(Challenges, m_Arb_PUFs_evaled,
                                         seg_length=seg_length)

    elif variant == 'indi':
        mod_Challenges = indi_order_variant(Challenges, m_Arb_PUFs_evaled,
                                            Ordering=Ordering)

    else:
        print('Select suitable variant and provide required parameters!')

    response = t_XOR_Arbiter_PUF.eval(mod_Challenges)

    return response



def run_all():
    """ Prepare functions to create the PUF instances """
    gen_ArbPUF = partial(XORArbiterPUF, n=n_bits, k=1, noisiness=noise)
    gen_t_XOR_ArbPUF = partial(XORArbiterPUF, n=n_bits, k=t, noisiness=noise)

    """ Use seeds for reproducibility, etc. """
    l_of_Arb_PUFs = [gen_ArbPUF(seed=i) for i in range(m)]
    t_XORArbPuf = gen_t_XOR_ArbPUF(seed=5)

    miPUF_instance = (l_of_Arb_PUFs, t_XORArbPuf)

    """ Create challenges to be used with seed """
    Challenges_C = random_inputs(n=n_bits, N=no_challenges, seed=1)

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

    result_basic_s0_bl2 = miPUF_eval(miPUF_instance, Challenges_C, 'basic', start=1,
                                     block_length=2, n_jobs=parallel_jobs)

    """ Basic: First block starting at bit 5, then m=4 blocks of length 12 (first 4 and 12 last challenge bits not affected by XORing)
    
    result_basic_s5_bl12 = miPUF_eval(miPUF_instance, Challenges_C, 'basic',
                                      start=5, block_length=12,
                                      n_jobs=parallel_jobs)
    """
    """ Rainbow: Since (n_bits=64) % (m=4) = 0 all bits will be equally affected by cyclic XORing
    """
    result_rainbow = miPUF_eval(miPUF_instance, Challenges_C, 'rainbow',
                                n_jobs=parallel_jobs)

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
    #print(all(result_indi_order == result_basic_s5_bl12))

    response = (result_basic_s0_bl2 + 1) / 2
    challenge = Challenges_C

    np.save(f'../data/challenges_{t}XOR_{n_bits}bit_basics0bl2_{no_challenges}', challenge)
    np.save(f'../data/respnses_{t}XOR_{n_bits}bit_basics0bl2_{no_challenges}', response)



""" Example Usage
"""

""" Create a 64bit (4,7)-miPUF instance and 10^5 challenges to evaluate 
"""
n_bits = 8
# number of APUFs
m = 2
# final t-XOR APUF
t = 2
noise = 0
no_challenges = 1000000
#run_all()

