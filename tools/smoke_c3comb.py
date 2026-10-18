"""One-batch runtime witness for the exact combinatorial model."""

import argparse
import time

import torch

from recbole.config import Config
from recbole.data import create_dataset
from recbole.data.utils import create_samplers, get_dataloader
from recbole.utils import get_model, init_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='retail_time_beh')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--max-seq-length', type=int, default=200)
    parser.add_argument('--hidden-size', type=int, default=32)
    parser.add_argument('--negatives', type=int, default=8)
    parser.add_argument('--linear-kernel', type=int, choices=[0, 1], default=1)
    parser.add_argument('--kernel-channels', type=int, default=8)
    parser.add_argument('--closed-form-kernel', type=int, choices=[0, 1], default=1)
    parser.add_argument('--candidate-chunk-size', type=int, default=64)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument(
        '--target-length', type=int, default=0,
        help='When positive, benchmark the training row nearest this prefix length.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config(
        model='C3Comb',
        dataset=args.dataset,
        config_dict={
            'USER_ID_FIELD': 'session_id',
            'load_col': None,
            'benchmark_filename': [
                'train', 'valid', 'development', 'confirmation'
            ],
            'alias_of_item_id': ['item_id_list'],
            'neg_sampling': None,
            'MAX_ITEM_LIST_LENGTH': args.max_seq_length,
            'train_batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'customized_eval': 0,
            'eval_args': {'mode': 'pop100', 'order': 'TO'},
            'topk': [5, 10, 101],
            'metrics': ['Recall', 'NDCG', 'MRR'],
            'gpu_id': args.gpu_id,
            'seed': 2026,
            'reproducibility': True,
            'hidden_size': args.hidden_size,
            'target_behavior_token': '0',
            'combo_negatives': args.negatives,
            'combo_uniform_weight': 0.4,
            'combo_pop_weight': 0.3,
            'combo_hard_weight': 0.3,
            'combo_pop_alpha': 0.75,
            'combo_linear_kernel': args.linear_kernel,
            'combo_kernel_channels': args.kernel_channels,
            'combo_closed_form_kernel': args.closed_form_kernel,
            'combo_temperature': 0.7,
            'combo_min_temperature': 0.05,
            'combo_logit_scale': 8.0,
            'combo_learnable_scale': 1,
            'combo_candidate_chunk_size': args.candidate_chunk_size,
        },
    )
    init_seed(config['seed'], config['reproducibility'])
    dataset = create_dataset(config)
    train_dataset, valid_dataset, _, _ = dataset.build()
    train_sampler, valid_sampler = create_samplers(
        config, dataset, [train_dataset, valid_dataset]
    )
    train_data = get_dataloader(config, 'train')(
        config, train_dataset, train_sampler, shuffle=True
    )
    valid_data = get_dataloader(config, 'test')(
        config, valid_dataset, valid_sampler, shuffle=False
    )
    model = get_model(config['model'])(config, train_dataset).to(config['device'])

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    if args.target_length > 0:
        lengths = train_dataset.inter_feat[config['ITEM_LIST_LENGTH_FIELD']]
        sample_indices = torch.topk(
            -(lengths - args.target_length).abs(),
            min(args.batch_size, len(lengths)),
        ).indices
        train_batch = train_dataset.inter_feat[sample_indices].to(
            config['device']
        )
    else:
        train_batch = next(iter(train_data)).to(config['device'])
    start = time.perf_counter()
    loss = model.calculate_loss(train_batch)
    loss.backward()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_seconds = time.perf_counter() - start

    model.eval()
    valid_tuple = next(iter(valid_data))
    valid_batch_cpu, valid_row_idx, valid_positive_u, valid_positive_i = valid_tuple
    valid_data.pr = 0
    repeated_valid_batch = next(iter(valid_data))[0]
    if not torch.equal(
        valid_batch_cpu['item_id'], repeated_valid_batch['item_id']
    ):
        raise AssertionError('Dataset evaluation candidates are not stable.')
    valid_batch = valid_batch_cpu.to(config['device'])
    start = time.perf_counter()
    with torch.no_grad():
        scores = model.predict(valid_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - start
    peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2)
        if torch.cuda.is_available() else 0.0
    )
    print(
        'C3COMB_WITNESS',
        f'loss={loss.item():.6f}',
        f'train_batch={tuple(train_batch["item_id_list"].shape)}',
        f'train_valid_lengths={train_batch["item_id_list"].ne(0).sum(dim=1).tolist()}',
        f'train_seconds={train_seconds:.4f}',
        f'eval_scores={tuple(scores.shape)}',
        f'eval_row_dtype={getattr(valid_row_idx, "dtype", None)}',
        f'eval_item_dtype={valid_batch_cpu["item_id"].dtype}',
        f'eval_positive_u_dtype={getattr(valid_positive_u, "dtype", None)}',
        f'eval_protocol={config["eval_args"]["mode"]}',
        f'eval_seconds={eval_seconds:.4f}',
        f'peak_gpu_mb={peak_mb:.2f}',
        f'device={config["device"]}',
    )


if __name__ == '__main__':
    main()
