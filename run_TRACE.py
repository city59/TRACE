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


DATASET_PRESETS = {
    'taobao': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 6},
    'retailrocket': {'path_temperature': 0.4, 'n_path_blocks': 2, 'max_alternatives': 2},
    'ijcai': {'path_temperature': 0.6, 'n_path_blocks': 3, 'max_alternatives': 8},
}
SEED = 2026


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='retailrocket',
                        choices=sorted(DATASET_PRESETS))
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=256,
                        help='Evaluation instances per batch (each has 1 + 100 candidates).')
    parser.add_argument('--micro_batch', type=int, default=256,
                        help='Rows per checkpointed micro-batch (memory only).')
    parser.add_argument('--artifact_root', type=str, default='run_artifacts/trace')
    return parser.parse_args()


def main():
    args = get_args()
    hyper = DATASET_PRESETS[args.dataset]

    config_dict = {
        'USER_ID_FIELD': 'session_id',
        'load_col': None,
        'neg_sampling': None,
        'benchmark_filename': ['train', 'valid', 'test'],
        'alias_of_item_id': ['item_id_list', 'neg_item_list'],
        'MAX_ITEM_LIST_LENGTH': 200,
        'eval_args': {'mode': 'full', 'order': 'TO'},
        'metrics': ['Hit', 'NDCG'],
        'topk': [10],
        'valid_metric': 'NDCG@10',
        'customized_eval': 1,
        'gpu_id': args.gpu_id,
        'seed': SEED,
        'reproducibility': True,
        'show_progress': False,
        'epochs': 200,
        'stopping_step': 10,
        'min_delta': 0.0,
        'learner': 'adamw',
        'learning_rate': 1e-3,
        'weight_decay': 0.0,
        'train_batch_size': 2048,
        'eval_batch_size': args.eval_batch_size,
        'train_micro_batch': args.micro_batch,
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
    train_dataset, valid_dataset, test_dataset = dataset.build()
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
        result_dir, f'{args.dataset}-TRACE.json'
    )
    with open(result_path, 'w', encoding='utf-8') as result_file:
        json.dump({
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'dataset': args.dataset,
            'seed': SEED,
            'hyperparameters': {key: config[key] for key in (
                'path_temperature', 'n_path_blocks', 'max_alternatives', 'learning_rate',
                'train_batch_size', 'hidden_size', 'reg_weight', 'train_neg_num')},
            'best_valid_score': float(best_valid_score),
            'best_valid_result': dict(best_valid_result),
            'test_result': dict(test_result),
            'elapsed_seconds': elapsed,
            'checkpoint': trainer.saved_model_file,
        }, result_file, indent=2)
    logger.info(f'result artifact: {result_path}')


if __name__ == '__main__':
    main()
