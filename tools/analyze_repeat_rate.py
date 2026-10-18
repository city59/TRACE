import argparse
import csv
import json
from pathlib import Path


def repeat_rate(path):
    rows = 0
    repeated = 0
    with path.open(encoding='utf-8') as source:
        reader = csv.DictReader(source, delimiter='\t')
        for row in reader:
            rows += 1
            history = set(row['item_id_list:token_seq'].split())
            repeated += row['item_id:token'] in history
    return {
        'rows': rows,
        'repeated_targets': repeated,
        'repeat_fraction': repeated / rows if rows else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, default=Path('dataset'))
    parser.add_argument(
        '--datasets', nargs='+', default=['retail_beh', 'tmall_beh', 'ijcai_beh']
    )
    args = parser.parse_args()

    result = {}
    for dataset in args.datasets:
        result[dataset] = {}
        for split in ('train', 'test'):
            path = args.dataset_root / dataset / f'{dataset}.{split}.inter'
            result[dataset][split] = repeat_rate(path)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
