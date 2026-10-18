"""Evaluate a non-learned recency-of-repeat sanity baseline on C3Comb candidates."""

import argparse
import json
from pathlib import Path
import re

import torch

from recbole.data import create_dataset
from recbole.data.utils import create_samplers, get_dataloader


def metrics_from_ranks(ranks, cutoffs=(5, 10, 101)):
    output = {}
    ranks = ranks.float()
    for cutoff in cutoffs:
        hits = ranks.le(cutoff)
        output[f'recall@{cutoff}'] = hits.float().mean().item()
        output[f'ndcg@{cutoff}'] = torch.where(
            hits, 1.0 / torch.log2(ranks + 1.0), torch.zeros_like(ranks)
        ).mean().item()
        output[f'mrr@{cutoff}'] = torch.where(
            hits, 1.0 / ranks, torch.zeros_like(ranks)
        ).mean().item()
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument(
        '--eval-mode', default='pop100',
        help='Shared sampled dataset protocol, e.g. pop100 or uni100.',
    )
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding='utf-8'))
    checkpoint_path = Path(payload['checkpoint'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    mode_match = re.fullmatch(r'(pop|uni)([1-9][0-9]*)', args.eval_mode)
    if mode_match is None:
        raise ValueError(
            'The repeat sanity diagnostic requires sampled popN or uniN evaluation.'
        )
    distribution = (
        'popularity' if mode_match.group(1) == 'pop' else 'uniform'
    )
    sample_count = int(mode_match.group(2))
    config['customized_eval'] = 0
    config['eval_args'] = {'mode': args.eval_mode, 'order': 'TO'}
    config['eval_neg_sample_args'] = {
        'strategy': 'by',
        'by': sample_count,
        'distribution': distribution,
    }
    dataset = create_dataset(config)
    train_dataset, _, development_dataset, _ = dataset.build()
    _, development_sampler = create_samplers(
        config, dataset, [train_dataset, development_dataset]
    )
    development_data = get_dataloader(config, 'test')(
        config, development_dataset, development_sampler, shuffle=False
    )
    all_ranks = []
    for interaction, row_idx, positive_u, positive_i in development_data:
        if row_idx is None:
            raise ValueError(
                'This diagnostic requires a sampled dataset evaluation protocol.'
            )
        items = interaction['item_id_list']
        candidates = interaction['item_id']
        positions = torch.arange(items.size(1)).view(1, -1)
        occurrences = candidates.unsqueeze(1).eq(items)
        last_occurrence = torch.where(
            occurrences,
            positions,
            torch.full_like(positions, -1),
        ).max(dim=1).values.float()
        # Candidate-id jitter is target-independent and only breaks exact ties.
        scores = last_occurrence + candidates.remainder(1009).float() * 1e-6
        batch_user_num = int(positive_u[-1].item()) + 1
        score_matrix = torch.full(
            (batch_user_num, dataset.item_num), -float('inf')
        )
        score_matrix[row_idx.long(), candidates.long()] = scores
        positive_rows = positive_u.long()
        positive_items = positive_i.long()
        positive_scores = score_matrix[
            positive_rows, positive_items
        ].unsqueeze(1)
        ranks = 1 + score_matrix[positive_rows].gt(
            positive_scores
        ).sum(dim=1)
        all_ranks.append(ranks)

    ranks = torch.cat(all_ranks)
    metrics = metrics_from_ranks(ranks)
    print(json.dumps({
        'evaluation_type': config['eval_args']['mode'],
        'baseline': 'last-occurrence repeat sanity check',
        'sample_count': int(ranks.numel()),
        'metrics': {key: round(value, 6) for key, value in metrics.items()},
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
