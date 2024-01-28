import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor as Pool

import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    EarlyStopping, ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger

from ElectricalPUFAttack import ElectricalPUFModule
from data.DataModule import PUFDataModule

ARCHITECTURES = ['Mursi', 'Custom']


def get_model_name(f, s, a, sp):
    # {filename}_s{seed}_a{architecture}_{start_param_model_name if exists}
    return f'{f}_s{s}_{a}{"_sp_" + sp if sp else ""}'


def store_run_on_seeds(architecture, seeds, results_by_data_file):
    with open('results.json', 'r+') as f:
        results = json.load(f)
        # Add empty result dict if field for data does not exist in result file
        if args.f not in results:
            results[args.f] = {a: {} for a in ARCHITECTURES}
        # Add empty result dict if architecture does not exist for data field
        elif architecture not in results[args.f]:
            results[args.f][architecture] = {}
        for seed in seeds:
            acc = results_by_data_file[architecture][seed]
            # Only store actually computed values (-1 is a placeholder)
            if acc > -1:
                results[args.f][architecture][str(seed)] = acc

        f.seek(0)
        json.dump(results, f)
        f.truncate()


def run_model(cfile, rfile, args, ids, start_param_model=None):
    data_module = PUFDataModule(
        args,
        cfile,
        rfile,
        args.hparams['bs'],
        ids
    )
    data_module.setup()

    monitor_params = {
        'monitor': 'Val Accuracy',
        'mode': "max"
    }
    early_stop_callback = EarlyStopping(
        min_delta=0.0,
        patience=100,
        verbose=True,
        stopping_threshold=0.95,
        **monitor_params
    )
    callbacks = [early_stop_callback]
    enable_checkpointing = False

    # Prepare checkpointing if best model should be used as final model
    if args.use_best_model:
        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            **monitor_params
        )
        callbacks.append(checkpoint_callback)
        enable_checkpointing = True

    model_name = get_model_name(args.f, args.s, args.a, args.sp)
    logger = TensorBoardLogger('runs', name=model_name)
    trainer = Trainer(
        accelerator=args.accelerator,
        devices=1,
        max_epochs=args.epochs,
        logger=logger,
        callbacks=callbacks,
        enable_checkpointing=enable_checkpointing
    )

    model = ElectricalPUFModule(args.hparams, args)
    # Load parameters from previously trained model
    if start_param_model:
        model.load_state_dict(torch.load(f'stored_models/{model_name}'))

    trainer.fit(model, datamodule=data_module)

    # Load best model from checkpoint
    if args.use_best_model:
        model = model.load_from_checkpoint(checkpoint_callback.best_model_path)

    train_accs = model.train_accs[-1]
    val_accs = model.val_accs[-1]
    test_accs = []

    # Only run test if dedicated test set is defined
    if ids[2] is not None:
        trainer.test(model, datamodule=data_module)
        test_accs = model.test_accs

    return model, (train_accs, val_accs, test_accs)


def run_multiple_models(args, all_architectures=False):
    """Run for each available data generation seed using multiprocessing. Also
    runs all predefined architectures if desired. Checks the stored results and
    only executes the runs for which no results are available yet.
    """
    architectures = ARCHITECTURES if all_architectures else [args.a]
    seeds = [int(s.split('.')[0][1:]) for s in
             next(os.walk(f'data/challenges/{args.f}/'))[2]]
    args_list = []
    # Fill results with -1 as placeholder (will be ignored later on)
    results_by_data_file = {a: {s: -1 for s in seeds} for a in architectures}

    for a in architectures:
        for s in seeds:
            model_name = get_model_name(args.f, s, a, args.sp)
            # Only execute run if no results for this model have been stored
            if not os.path.exists(f'stored_models/{model_name}'):
                args.a = a
                args.s = s
                args_list.append(argparse.Namespace(**vars(args)))
            else:
                print(f'Run for {model_name} already exists.')

        pool = Pool(args.n_pc)
        results = pool.map(main, args_list)

        val_accs = [r[1] for r in results]
        for acc, args in zip(val_accs, args_list):
            results_by_data_file[args.a][args.s] = acc
        args_list = []
        store_run_on_seeds(a, seeds, results_by_data_file)


def main(args):
    stages = int(args.f.split('bit_')[1].split('XOR')[0])
    # Round stages down to nearest even integer for Mursi attack
    if args.a == 'Mursi' and stages % 2 == 1:
        stages -= 1
    args.stages = stages

    with open('hparams.json', 'r') as hparam_f:
        hparams_all = json.load(hparam_f)
        hparams = hparams_all[args.a]
    args.hparams = hparams

    cfile = f'data/challenges/{args.f}/s{args.s}.npy'
    rfile = f'data/responses/{args.f}/s{args.s}.npy'

    length = args.challenges

    # Ids of CRPs to be used for attack. Shuffle and use 99% for training, 1%
    # for validation and no explicit test set.
    ids = list(range(length))
    np.random.shuffle(ids)

    train_ids = ids[:int(length * 0.99)]
    val_ids = ids[len(train_ids):]
    test_ids = None
    ids = (train_ids, val_ids, test_ids)

    best_model, accs = run_model(cfile, rfile, args, ids,
                                 start_param_model=args.sp)
    print('Model results:', accs)

    model_name = get_model_name(args.f, args.s, args.a, args.sp)
    torch.save(best_model.state_dict(), f'stored_models/{model_name}')
    return accs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--f', '--file', default='64bit_4XOR_50k')
    parser.add_argument('--s', '--seed', type=int, default=0)
    parser.add_argument('--a', '--architecture', default='Mursi')
    parser.add_argument('--epochs', default=1000)
    parser.add_argument('--sp', '--start-params', default=None)
    parser.add_argument('--do-log', default=True)
    parser.add_argument('--use-best-model', default=False)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--n_pc', '--n_processes', default=1)
    args = parser.parse_args()

    args.bits = int(args.f.split('bit')[0])
    args.challenges = int(args.f.split('_')[-1][:-1]) * 1000

    np.random.seed(345)
    # main(args)
    run_multiple_models(args, all_architectures=True)
