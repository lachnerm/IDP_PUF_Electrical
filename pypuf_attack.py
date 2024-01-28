import argparse
import json
import warnings
from functools import partial

import joblib
import numpy as np
import pandas as pd
import pypuf.attack
import pypuf.io
import pypuf.simulation

from simulation.miPUF_sim_original import miPUF_eval

succ_threshold = 0.90
pd.set_option('display.max_rows', None)
pd.options.display.float_format = "{:,.2f}".format

mursi_results = {
    '4': 150000,
    '5': 200000,
    '6': 2000000,
    '7': 4000000,
    '8': 6000000,
    '9': 45000000,
    '10': 119000000,
    '11': 325000000,
}

def get_mipuf_nets(k, m=0, **kwargs):
    nets = {
        'Wisiol': {
            'net': [n, n / 2, n / 2, n],
            'act': 'relu'
        },
        'Custom5': {
            'net': [n + 2 ** (k - 1), n + 2 ** k, n + 2 ** k, n + 2 ** (k - 1)],
            'act': 'tanh'
        }
    }
    return nets


'''
        'Mursi': {
            'net': [2 ** (k - 1), 2 ** k, 2 ** (k - 1)],
            'act': 'tanh'
        },
        'Custom11': {
            'net': [n + m, n / 2, n / 2, n + m],
            'act': 'tanh'
        },
        'Custom13': {
            'net': [n + 2 ** m, n + 2 ** m / 2, n + 2 ** m / 2, n + 2 ** m],
            'act': 'tanh'
        },
        'Custom6': {
            'net': [n + 2 ** (k - 1), n + 2 ** k, n + 2 ** (k - 1)],
            'act': 'tanh'
        },
        'Custom7': {
            'net': [2 * n + 2 ** (k - 1), 2 * n + 2 ** k, 2 * n + 2 ** (k - 1)],
            'act': 'tanh'
        },
        'Custom8': {
            'net': [n + 2 ** (k - 2), n + 2 ** (k - 1), n + 2 ** (k - 1),
                    n + 2 ** (k - 2)],
            'act': 'tanh'
        },
        'Custom9': {
            'net': [n + 2 ** (k - 1), 2 ** k, 2 ** k, n + 2 ** (k - 1)],
            'act': 'tanh'
        },
        'Custom10': {
            'net': [2 ** (k - 1), n + 2 ** k, n + 2 ** k, 2 ** (k - 1)],
            'act': 'tanh'
        },
        'Custom12': {
            'net': [n+2**m, n / 2, n / 2, n+2**m],
            'act': 'tanh'
        },
        'Aseeri': {
            'net': [2 ** k, 2 ** k, 2 ** k],
            'act': 'relu'
        },
'Custom1': {
    'net': [2 ** (k - 1), 2 ** k, 2 ** k, 2 ** (k - 1)],
    'act': 'tanh'
},
'Custom2': {
    'net': [2 ** (k - 1), 2 ** k, 2 ** k, 2 ** k, 2 ** (k - 1)],
    'act': 'tanh'
},
'Custom3': {
    'net': [2 ** k, 2 ** (k - 1), 2 ** (k - 2), 2 ** (k - 3)],
    'act': 'tanh'
},
'Custom4': {
    'net': [2 ** k, 2 ** (k - 1), 2 ** (k - 2), 2 ** (k - 3),
            2 ** (k - 4)],
    'act': 'tanh'
},'''


def check_if_run_exists(architecture, seed, n, k, N):
    with open('results_miPUF.json', 'r+') as f:
        results = json.load(f)

        seed = str(seed)
        n = str(n)
        k = str(k)
        N = str(N)

        if n not in results:
            return False
        elif k not in results[n]:
            return False
        elif N not in results[n][k]:
            return False
        elif architecture not in results[n][k][N]:
            return False
        elif seed not in results[n][k][N][architecture]:
            return False

        return True


def check_if_miPUF_run_exists(architecture, seed, n, k, N, m, start, bl):
    with open('results_miPUF.json', 'r+') as f:
        results = json.load(f)

        seed = str(seed)
        n = str(n)
        k = str(k)
        N = str(N)
        m = str(m)
        start = str(start)
        bl = str(bl)

        if n not in results:
            return False
        elif m not in results[n]:
            return False
        elif start not in results[n][m]:
            return False
        elif bl not in results[n][m][start]:
            return False
        elif k not in results[n][m][start][bl]:
            return False
        elif N not in results[n][m][start][bl][k]:
            return False
        elif architecture not in results[n][m][start][bl][k][N]:
            return False
        elif seed not in results[n][m][start][bl][k][N][architecture]:
            return False

        return True


def store_run_on_seeds(acc, architecture, seed, n, k, N):
    with open('results.json', 'r+') as f:
        results = json.load(f)

        seed = str(seed)
        n = str(n)
        k = str(k)
        N = str(N)

        if n not in results:
            results[n] = {k: {N: {architecture: {seed: acc}}}}
        elif k not in results[n]:
            results[n][k] = {N: {architecture: {seed: acc}}}
        elif N not in results[n][k]:
            results[n][k][N] = {architecture: {seed: acc}}
        elif architecture not in results[n][k][N]:
            results[n][k][N][architecture] = {seed: acc}
        else:
            results[n][k][N][architecture][seed] = acc

        f.seek(0)
        json.dump(results, f)
        f.truncate()


def store_run_on_miPUF_seeds(acc, architecture, seed, n, k, N, m, start, bl):
    with open('results_miPUF.json', 'r+') as f:
        results = json.load(f)

        seed = str(seed)
        n = str(n)
        k = str(k)
        N = str(N)
        m = str(m)
        start = str(start)
        bl = str(bl)

        if n not in results:
            results[n] = {
                m: {start: {bl: {k: {N: {architecture: {seed: acc}}}}}}}
        elif m not in results[n]:
            results[n][m] = {start: {bl: {k: {N: {architecture: {seed: acc}}}}}}
        elif start not in results[n][m]:
            results[n][m][start] = {bl: {k: {N: {architecture: {seed: acc}}}}}
        elif bl not in results[n][m][start]:
            results[n][m][start][bl] = {k: {N: {architecture: {seed: acc}}}}
        elif k not in results[n][m][start][bl]:
            results[n][m][start][bl][k] = {N: {architecture: {seed: acc}}}
        elif N not in results[n][m][start][bl][k]:
            results[n][m][start][bl][k][N] = {architecture: {seed: acc}}
        elif architecture not in results[n][m][start][bl][k][N]:
            results[n][m][start][bl][k][N][architecture] = {seed: acc}
        elif seed not in results[n][m][start][bl][k][N][architecture]:
            results[n][m][start][bl][k][N][architecture][seed] = acc

        f.seek(0)
        json.dump(results, f)
        f.truncate()


def run_all_ns_and_ms_and_ks_and_architectures_and_CRPs(archs, puf_type,
                                                        **kwargs):
    # for n in [32, 64, 128]:
    for n in [64]:
        if n == 32:
            start = 1
            bl = 3
        else:
            start = 10
            bl = 5
        for m in [3, 4, 5, 6, 7]:
            for k in [3, 4, 5, 6]:
                run_all_architectures_and_CRPs(n, k, archs, puf_type, m=m,
                                               bl=bl, start=start, **kwargs)


def run_all_ks_and_architectures_and_CRPs(n, archs, puf_type, **kwargs):
    for k in [3, 4, 5, 6, 7]:
        run_all_architectures_and_CRPs(n, k, archs, puf_type, **kwargs)

def get_Ns_from_XOR_APUF_comparison(puf_type, n, k, m):
    if puf_type == 'XORAPUF':
        raise RuntimeError(f'Undefined for XOR-APUFs.')
    elif puf_type == 'miPUF':
        if n == 64:
            XORAPUF_equiv = int(np.ceil(m // 2 + k))
            Ns = mursi_results[str(XORAPUF_equiv)]
            return [Ns, Ns*2, Ns*5, Ns*10]
        else:
            raise RuntimeError(f'Undefined for n={n}')
    else:
        raise RuntimeError(f'Undefined puf type {puf_type}.')
    return None

def get_Ns_from_run_definition(puf_type, n, k, m):
    if puf_type == 'XORAPUF':
        if k == 2:
            return [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000,
                  8000000, 9000000, 10000000, 20000000, 30000000]
        elif k == 3:
            return [5000000, 6000000, 7000000,
                  8000000, 9000000, 10000000, 20000000]
        elif k == 4:
            return [1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 7000000,
                  8000000, 9000000, 10000000, 20000000, 30000000]
        elif k == 5:
            return [4000000, 5000000, 6000000, 7000000, 8000000]
        elif k == 6:
            return [5000000, 8000000, 12000000, 15000000, 20000000]
        elif k == 7:
            return [5000000, 10000000, 15000000, 20000000, 25000000, 30000000]
        else:
            raise RuntimeError(f'Undefined k={k} for run.')
    elif puf_type == 'miPUF':
        if n == 32:
            if m == 5:
                if k == 3:
                    return [170000]
                elif k == 4:
                    return [400000]
                elif k == 5:
                    return [500000]
                elif k == 6:
                    return [5000000, 6000000, 7000000]
                elif k == 7:
                    return [20000000, 30000000, 40000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 6:
                if k == 3:
                    return [300000, 500000, 700000]
                elif k == 4:
                    return [1000000, 1200000, 150000]
                elif k == 5:
                    return [3000000, 4000000, 5000000]
                elif k == 6:
                    return [10000000, 20000000, 30000000]
                elif k == 7:
                    return [100000000, 200000000, 300000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 7:
                if k == 3:
                    return [500000, 700000, 900000]
                elif k == 4:
                    return [2000000, 3000000, 4000000]
                elif k == 5:
                    return [8000000, 12000000, 15000000]
                elif k == 6:
                    return [30000000, 40000000, 50000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 8:
                if k == 3:
                    return [900000, 1200000, 1500000]
                elif k == 4:
                    return [5000000, 7000000, 9000000]
                elif k == 5:
                    return [10000000, 15000000, 20000000]
                elif k == 6:
                    return [70000000, 90000000, 100000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            else:
                raise RuntimeError(f'Undefined m={m} for run.')
        elif n == 64:
            if m == 5:
                if k == 3:
                    return [400000]
                elif k == 4:
                    return [900000]
                elif k == 5:
                    return [2000000]
                elif k == 6:
                    return [20000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 6:
                if k == 3:
                    return [600000]
                elif k == 4:
                    return [1500000]
                elif k == 5:
                    return [5000000]
                elif k == 6:
                    return [30000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 7:
                if k == 3:
                    return [2000000]
                elif k == 4:
                    return [5000000]
                elif k == 5:
                    return [10000000]
                elif k == 6:
                    return [50000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 8:
                if k == 3:
                    return [900000, 1200000, 1500000]
                elif k == 4:
                    return [5000000, 7000000, 9000000]
                elif k == 5:
                    return [10000000, 15000000, 20000000]
                elif k == 6:
                    return [100000000, 150000000, 200000000]
                elif k == 7:
                    return [700000000, 800000000, 900000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            else:
                raise RuntimeError(f'Undefined m={m} for run.')
        elif n == 128:
            if m == 5:
                if k == 3:
                    return [50000, 100000, 150000]
                elif k == 4:
                    return [500000]
                elif k == 5:
                    return [900000]
                elif k == 6:
                    return [100000000, 150000000, 200000000]
                elif k == 7:
                    return [700000000, 800000000, 900000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 6:
                if k == 3:
                    return [300000, 500000, 700000]
                elif k == 4:
                    return [1000000, 1200000, 150000]
                elif k == 5:
                    return [3000000, 4000000, 5000000]
                elif k == 6:
                    return [10000000, 20000000, 30000000]
                elif k == 7:
                    return [100000000, 200000000, 300000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 7:
                if k == 3:
                    return [500000, 700000, 900000]
                elif k == 4:
                    return [2000000, 3000000, 4000000]
                elif k == 5:
                    return [8000000, 12000000, 15000000]
                elif k == 6:
                    return [100000000, 150000000, 200000000]
                elif k == 7:
                    return [700000000, 800000000, 900000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            elif m == 8:
                if k == 3:
                    return [900000, 1200000, 1500000]
                elif k == 4:
                    return [5000000, 7000000, 9000000]
                elif k == 5:
                    return [10000000, 15000000, 20000000]
                elif k == 6:
                    return [100000000, 150000000, 200000000]
                elif k == 7:
                    return [700000000, 800000000, 900000000]
                else:
                    warnings.warn(f'Undefined k={k} for run.')
                    return
            else:
                raise RuntimeError(f'Undefined m={m} for run.')
    else:
        raise RuntimeError(f'Undefined puf type {puf_type}.')
    return None

def run_all_architectures_and_CRPs(n, k, archs, puf_type, m=0, **kwargs):
    #Ns = get_Ns_from_run_definition(puf_type, n, k, m)
    Ns = get_Ns_from_XOR_APUF_comparison(puf_type, n, k, m)
    for N in Ns:
        run_all_architectures(N, n, k, m, archs, puf_type, **kwargs)


def run_all_architectures(N, n, k, m, archs, puf_type, **kwargs):
    for architecture in archs:
        if puf_type == 'XORAPUF':
            run_seeds(N, n, k, architecture)
        elif puf_type == 'miPUF':
            run_miPUF_seeds(N, n, k, architecture,  m=m, **kwargs)
        else:
            raise RuntimeError(f'Undefined puf type {puf_type}.')


def run_seeds(N, n, k, architecture):
    accs = []
    n_seeds = 10
    for s in range(n_seeds):
        if not check_if_run_exists(architecture, s, n, k, N):
            puf = pypuf.simulation.XORArbiterPUF(n=n, k=k, seed=s, noisiness=0)
            crps = pypuf.io.ChallengeResponseSet.from_simulation(puf, N=N,
                                                                 seed=s)
            print(f"Starting run (N={N}, n={n}, k={k}, s={s}, {architecture})")
            acc = run_on_seed(N, n, k, s, architecture, crps,
                              puf_type='XORAPUF')
            accs.append(acc)
        else:
            print(f"Run exists (N={N}, n={n}, k={k}, s={s}, {architecture})")
    print_succ_runs(accs, n_seeds)


def run_miPUF_seeds(N, n, k, architecture, m=5, start=10, bl=5):
    accs = []
    n_seeds = 10
    for s in range(n_seeds):
        if not check_if_miPUF_run_exists(architecture, s, n, k, N, m, start,
                                         bl):
            gen_ArbPUF = partial(pypuf.simulation.XORArbiterPUF, n=n, k=1,
                                 noisiness=0)
            gen_t_XOR_ArbPUF = partial(pypuf.simulation.XORArbiterPUF, n=n, k=k,
                                       noisiness=0)

            """ Use seeds for reproducibility, etc. """
            l_of_Arb_PUFs = [gen_ArbPUF(seed=s + i) for i in range(m)]
            t_XORArbPuf = gen_t_XOR_ArbPUF(seed=s)

            miPUF_instance = (l_of_Arb_PUFs, t_XORArbPuf)

            """ Create challenges to be used with seed """
            challenge = pypuf.io.random_inputs(n=n, N=N, seed=s)

            """ If more than CPU cores shall be used to evaluate the m Arbiter PUFs
                select desired number.
            """
            # Get no. of cores available
            # Get no. of cores available
            no_cpu = joblib.cpu_count()
            # Leave 3 cores for other tasks free:
            parallel_jobs = max(1, no_cpu - 3)
            response = miPUF_eval(miPUF_instance, challenge,
                                  'basic',
                                  start=start,
                                  block_length=bl,
                                  n_jobs=parallel_jobs)
            # response = (result_basic_s0_bl2 + 1) / 2
            crps = pypuf.io.ChallengeResponseSet(challenge, response)
            print(f"Starting run (N={N}, n={n}, k={k}, m={m}, s={s}, {architecture})")
            acc = run_on_seed(N, n, k, s, architecture, crps, m=m, start=start,
                              bl=bl, puf_type='miPUF')
            accs.append(acc)
        else:
            print(f"Run exists (N={N}, n={n}, k={k}, s={s}, {architecture})")
    print_succ_runs(accs, n_seeds)


def run_on_seed(N, n, k, s, architecture, crps, puf_type=None, **kwargs):
    if puf_type == 'XORAPUF':
        nets = get_xorapuf_nets(k)
        early_stop_acc = 0.95
    elif puf_type == 'miPUF':
        nets = get_mipuf_nets(k, **kwargs)
        early_stop_acc = 0.9
    attack = pypuf.attack.MLPAttack2021(
        crps, seed=s, net=nets[architecture]['net'],
        activation_hl=nets[architecture]['act'],
        epochs=300, lr=.001, bs=1000, early_stop=early_stop_acc, patience=30
    )

    attack.fit()
    acc = attack.history['val_accuracy'][-1]

    if puf_type == 'XORAPUF':
        store_run_on_seeds(acc, architecture, s, n, k, N)
    elif puf_type == 'miPUF':
        store_run_on_miPUF_seeds(acc, architecture, s, n, k, N, **kwargs)
    else:
        raise RuntimeError(f'Undefined puf type {puf_type}.')
    return acc


def print_succ_runs(accs, n_seeds):
    succ_runs = np.count_nonzero([acc > succ_threshold for acc in accs])
    print('Successful runs: ', succ_runs, '/', n_seeds + 1)


def print_stored_results():
    with open('results.json', 'r+') as f:
        results = json.load(f)

    results_pandas = {
        f"{bits}{k}{N}{architecture}{seed}": (
            bits, k, N, architecture, seed,
            float(results[bits][k][N][architecture][seed]))
        for bits in results.keys()
        for k in results[bits].keys()
        for N in results[bits][k].keys()
        for architecture in results[bits][k][N]
        for seed in results[bits][k][N][architecture]
    }
    df = pd.DataFrame.from_dict(
        results_pandas, orient="index",
        columns=("Bits", "k", "N", "Architecture", 'Seed', 'Acc')
    )
    results_succ = df[df['Acc'] > 0.95].groupby(
        ['Bits', 'k', 'N', 'Architecture']).count()
    results_all = df.groupby(['Bits', 'k', 'N', 'Architecture']).count()
    results_pretty = results_succ / results_all
    print(results_pretty.fillna(0))


def print_stored_miPUF_results():
    with open('results_miPUF.json', 'r+') as f:
        results = json.load(f)

    results_pandas = {
        f"{bits}{m}{start}{bl}{k}{N}{architecture}{seed}": (
            bits, m, start, bl, k, N, architecture, seed,
            float(results[bits][m][start][bl][k][N][architecture][seed]))
        for bits in results.keys()
        for m in results[bits].keys()
        for start in results[bits][m].keys()
        for bl in results[bits][m][start].keys()
        for k in results[bits][m][start][bl].keys()
        for N in results[bits][m][start][bl][k].keys()
        for architecture in results[bits][m][start][bl][k][N]
        for seed in results[bits][m][start][bl][k][N][architecture]
    }
    df = pd.DataFrame.from_dict(
        results_pandas, orient="index",
        columns=(
            "Bits", 'm', 'start', 'bl', "k", "N", "Architecture", 'Seed', 'Acc')
    )
    results_all = df.groupby(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Architecture']).mean()
    results_all = results_all.fillna(0)['Acc'].reset_index().sort_values(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Acc'])
    print(results_all.set_index(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Architecture']))

    results_succ = df[df['Acc'] > 0.90].groupby(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Architecture'],
        group_keys=False).count()
    results_all = df.groupby(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Architecture'],
        group_keys=False).count()
    results_pretty = results_succ / results_all
    results_pretty = results_pretty.fillna(0)['Acc'].reset_index().sort_values(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Acc'])
    '''print(results_pretty.set_index(
        ['Bits', 'm', 'start', 'bl', 'k', 'N', 'Architecture']))'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', default=2000000, type=int)
    parser.add_argument('--n', default=64, type=int)
    parser.add_argument('--k', default=6, type=int)
    parser.add_argument('--a', '--architecture', default='Mursi')
    parser.add_argument('--puf', default='miPUF')

    args = parser.parse_args()
    n = args.n
    k = args.k
    N = args.N
    architecture = args.a

    print(f'Running {architecture} attack (n={n}, k={k}, N={N})')

    archs = list(get_mipuf_nets(0).keys())
    # archs = [*[f'Custom{x}' for x in range(1, 5)]]

    if args.puf == 'XORAPUF':
        print_stored_results()
    elif args.puf == 'miPUF':
        print_stored_miPUF_results()
    else:
        raise RuntimeError(f'Undefined puf type {args.puf}.')
    print_stored_results()

    run_all_ns_and_ms_and_ks_and_architectures_and_CRPs(archs, args.puf)
    # run_all_ks_and_architectures_and_CRPs(n, archs, args.puf)

    # run_seeds(N, n, k, architecture)
    # run_miPUF_seeds(N, n, k, architecture)
    # run_all_architectures(N, n, k, archs, args.puf)
    # run_all_architectures_and_CRPs(n, k, archs)
