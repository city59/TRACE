"""Train and evaluate TRACE with the settings of Sec. 5.1 of the paper.

Example:
    python run_TRACE.py --dataset retail_beh --gpu_id 0 --seed 2026
"""

import argparse
import json
import os
import time
from datetime import datetime
from logging import getLogger

import torch
from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import create_samplers, get_dataloader
from recbole.utils import get_model, get_trainer, init_logger, init_seed, set_color


# Best values reported in Sec. 5.4 (tau, J, Omega) per dataset.
# MBHT dataset names: tmall_beh = Taobao, retail_beh = Retailrocket, ijcai_beh = IJCAI.
DATASET_PRESETS = {
    'retail_beh': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 2},
    'retail_time_beh': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 2},
    'retail_full_time_beh': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 2},
    'tmall_beh': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 6},
    'ijcai_beh': {'path_temperature': 0.6, 'n_path_blocks': 3, 'max_alternatives': 8},
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='retail_beh')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2026)
    # Sec. 5.1 parameter settings.
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--reg_weight', type=float, default=1e-4)
    # tau / J / Omega; a negative value means "use the dataset preset".
    parser.add_argument('--tau', type=float, default=-1)
    parser.add_argument('--blocks', type=int, default=-1)
    parser.add_argument('--omega', type=int, default=-1)
    parser.add_argument('--ablation', type=str, default='none')
    # Settings the paper does not fix.
    parser.add_argument('--train_neg_num', type=int, default=1)
    parser.add_argument('--eval_batch_size', type=int, default=101 * 32)
    parser.add_argument('--micro_batch', type=int, default=256,
                        help='Rows per checkpointed micro-batch (memory only).')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--stopping_step', type=int, default=10)
    parser.add_argument('--max_seq_length', type=int, default=200)
    parser.add_argument('--valid_portion', type=float, default=0.1,
                        help='Hold-out ratio of the training file used for validation '
                             '(datasets without an explicit validation file).')
    parser.add_argument('--eval_mode', type=str, default='pop100',
                        help='100 popularity-sampled negatives per positive (Sec. 5.1).')
    parser.add_argument('--artifact_root', type=str, default='run_artifacts/trace')
    return parser.parse_args()


def main():
    args = get_args()
    timestamped = os.path.isfile(
        os.path.join('dataset', args.dataset, f'{args.dataset}.valid.inter')
    )
    preset = DATASET_PRESETS.get(args.dataset, DATASET_PRESETS['tmall_beh'])
    hyper = {
        'path_temperature': args.tau if args.tau > 0 else preset['path_temperature'],
        'n_path_blocks': args.blocks if args.blocks > 0 else preset['n_path_blocks'],
        'max_alternatives': args.omega if args.omega >= 0 else preset['max_alternatives'],
    }

    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        'neg_sampling': None,
        'benchmark_filename': (
            ['train', 'valid', 'development', 'confirmation']
            if timestamped else ['train', 'test']
        ),
        'alias_of_item_id': ['item_id_list'],
        'MAX_ITEM_LIST_LENGTH': args.max_seq_length,
        'eval_args': {'mode': args.eval_mode, 'order': 'TO'},
        'metrics': ['Hit', 'NDCG'],
        'topk': [10],
        'valid_metric': 'NDCG@10',
        'customized_eval': 0,
        'gpu_id': args.gpu_id,
        'seed': args.seed,
        'reproducibility': True,
        'show_progress': False,
        'epochs': args.epochs,
        'stopping_step': args.stopping_step,
        'min_delta': 0.0,
        'learner': 'adamw',
        'learning_rate': args.learning_rate,
        # The L2 term lambda_reg * ||Theta||^2 is part of the loss (Eq. 16),
        # so the optimizer's decoupled weight decay is disabled.
        'weight_decay': 0.0,
        'train_batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'hidden_size': args.hidden_size,
        'reg_weight': args.reg_weight,
        'train_neg_num': args.train_neg_num,
        'train_micro_batch': args.micro_batch,
        'trace_ablation': args.ablation,
        'checkpoint_dir': os.path.join(args.artifact_root, 'saved'),
        **hyper,
    }
    config = Config(model='TRACE', dataset=args.dataset, config_dict=config_dict)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config, log_root=os.path.join(args.artifact_root, 'log'))
    logger = getLogger()
    logger.info(args)
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    built = dataset.build()
    if timestamped:
        train_dataset, valid_dataset, test_dataset, _ = built
    else:
        full_train, test_dataset = built
        full_train.shuffle()
        train_dataset, valid_dataset = full_train.split_by_ratio(
            [1 - args.valid_portion, args.valid_portion]
        )
    train_sampler, valid_sampler = create_samplers(
        config, dataset, [train_dataset, valid_dataset]
    )
    _, test_sampler = create_samplers(config, dataset, [train_dataset, test_dataset])
    train_data = get_dataloader(config, 'train')(config, train_dataset, train_sampler, shuffle=True)
    valid_data = get_dataloader(config, 'test')(config, valid_dataset, valid_sampler, shuffle=False)
    test_data = get_dataloader(config, 'test')(config, test_dataset, test_sampler, shuffle=False)

    model = get_model('TRACE')(config, train_data.dataset).to(config['device'])
    logger.info(model)
    trainer = get_trainer(config['MODEL_TYPE'], 'TRACE')(config, model)

    started = time.perf_counter()
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )
    test_result = trainer.evaluate(test_data, load_best_model=True,
                                   show_progress=config['show_progress'])
    elapsed = time.perf_counter() - started

    logger.info(set_color('best valid', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    result_dir = os.path.join(args.artifact_root, 'results')
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(
        result_dir, f'{args.dataset}-TRACE-{args.ablation}-seed{args.seed}.json'
    )
    with open(result_path, 'w', encoding='utf-8') as result_file:
        json.dump({
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'dataset': args.dataset,
            'seed': args.seed,
            'ablation': args.ablation,
            'hyperparameters': {**hyper, 'learning_rate': args.learning_rate,
                                'batch_size': args.batch_size,
                                'hidden_size': args.hidden_size,
                                'reg_weight': args.reg_weight,
                                'train_neg_num': args.train_neg_num},
            'best_valid_score': float(best_valid_score),
            'best_valid_result': dict(best_valid_result),
            'test_result': dict(test_result),
            'elapsed_seconds': elapsed,
            'checkpoint': trainer.saved_model_file,
        }, result_file, indent=2)
    logger.info(f'result artifact: {result_path}')


if __name__ == '__main__':
    main()
