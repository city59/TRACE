import argparse
import io
import json
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PURCHASE, CART, FAVORITE, VIEW = 0, 1, 2, 3
BEHAVIOR_NAMES = {PURCHASE: 'purchase', CART: 'cart', FAVORITE: 'favorite', VIEW: 'view'}
CHUNK = 5_000_000

MIN_ITEM_INTERACTIONS = 5
MIN_PURCHASES = 2
MAX_HISTORY = 200
NUM_NEGATIVES = 100
SEED = 2026
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / 'dataset'


def open_member(path, member):
    path = Path(path)
    if path.suffix == '.zip':
        archive = zipfile.ZipFile(path)
        name = next(n for n in archive.namelist() if n.endswith(member))
        return io.TextIOWrapper(archive.open(name), encoding='utf-8')
    return open(path, encoding='utf-8')


def read_chunks(handle, **kwargs):
    parts = []
    for chunk in pd.read_csv(handle, chunksize=CHUNK, **kwargs):
        parts.append(chunk)
        print(f'  read {sum(len(p) for p in parts):,} rows', flush=True)
    return parts


def load_taobao(path):
    mapping = {'buy': PURCHASE, 'cart': CART, 'fav': FAVORITE, 'pv': VIEW}
    users, items, behaviors, hours = [], [], [], []
    with open_member(path, 'UserBehavior.csv') as handle:
        for chunk in read_chunks(handle, header=None, usecols=[0, 1, 3, 4],
                                 names=['user', 'item', 'behavior', 'ts'],
                                 dtype={'user': np.int64, 'item': np.int64,
                                        'behavior': 'category', 'ts': np.int64}):
            users.append(chunk['user'].to_numpy())
            items.append(chunk['item'].to_numpy())
            behaviors.append(chunk['behavior'].map(mapping).astype(np.int8).to_numpy())
            hours.append(chunk['ts'].to_numpy() / 3600.0)
    return [np.concatenate(x) for x in (users, items, behaviors, hours)]


def load_retailrocket(path):
    mapping = {'transaction': PURCHASE, 'addtocart': CART, 'view': VIEW}
    with open_member(path, 'events.csv') as handle:
        frame = pd.concat(read_chunks(handle, usecols=['timestamp', 'visitorid', 'event', 'itemid'],
                                      dtype={'timestamp': np.int64, 'visitorid': np.int64,
                                             'event': 'category', 'itemid': np.int64}))
    return (frame['visitorid'].to_numpy(), frame['itemid'].to_numpy(),
            frame['event'].map(mapping).astype(np.int8).to_numpy(),
            frame['timestamp'].to_numpy() / 3_600_000.0)


def load_ijcai(path):
    mapping = {2: PURCHASE, 1: CART, 3: FAVORITE, 0: VIEW}
    day_of_year = {}
    users, items, behaviors, hours = [], [], [], []
    with open_member(path, 'user_log_format1.csv') as handle:
        for chunk in read_chunks(handle, usecols=['user_id', 'item_id', 'time_stamp', 'action_type'],
                                 dtype={'user_id': np.int64, 'item_id': np.int64,
                                        'time_stamp': str, 'action_type': np.int8}):
            stamps = chunk['time_stamp'].str.zfill(4)
            for s in stamps.unique():
                if s not in day_of_year:
                    day_of_year[s] = date(2014, int(s[:2]), int(s[2:])).timetuple().tm_yday
            users.append(chunk['user_id'].to_numpy())
            items.append(chunk['item_id'].to_numpy())
            behaviors.append(chunk['action_type'].map(mapping).astype(np.int8).to_numpy())
            hours.append(stamps.map(day_of_year).to_numpy(dtype=np.float64) * 24.0)
    return [np.concatenate(x) for x in (users, items, behaviors, hours)]


LOADERS = {'taobao': load_taobao, 'retailrocket': load_retailrocket, 'ijcai': load_ijcai}


def filter_log(user, item, behavior, min_item, min_purchase):
    keep = np.ones(len(user), dtype=bool)
    while True:
        _, item_inv, item_cnt = np.unique(item[keep], return_inverse=True, return_counts=True)
        item_ok = item_cnt[item_inv] >= min_item
        _, user_inv = np.unique(user[keep], return_inverse=True)
        purchases = np.bincount(user_inv, weights=(behavior[keep] == PURCHASE))
        user_ok = purchases[user_inv] >= min_purchase
        ok = item_ok & user_ok
        if ok.all():
            return keep
        idx = np.flatnonzero(keep)
        keep[idx[~ok]] = False
        print(f'  filtering: {keep.sum():,} interactions remain', flush=True)


def sample_negatives(rng, cumulative, item_ids, positive, count):
    chosen = []
    seen = {positive}
    while len(chosen) < count:
        draws = np.searchsorted(cumulative, rng.random(4 * count) * cumulative[-1], side='right')
        for item in item_ids[draws]:
            if item not in seen:
                seen.add(item)
                chosen.append(item)
                if len(chosen) == count:
                    break
    return chosen


def fmt_tokens(values):
    return ' '.join(map(str, values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=sorted(LOADERS), required=True)
    parser.add_argument('--input', type=Path, required=True)
    args = parser.parse_args()

    name = args.source
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{name}] loading {args.input}', flush=True)
    user, item, behavior, hours = LOADERS[args.source](args.input)
    raw_rows = len(user)
    if np.any(behavior < 0):
        raise ValueError('unknown behavior type in the raw log')

    keep = filter_log(user, item, behavior, MIN_ITEM_INTERACTIONS, MIN_PURCHASES)
    user, item, behavior, hours = user[keep], item[keep], behavior[keep], hours[keep]
    order = np.lexsort((hours, user))
    user, item, behavior, hours = user[order], item[order], behavior[order], hours[order]
    starts = np.flatnonzero(np.r_[True, user[1:] != user[:-1]])
    ends = np.r_[starts[1:], len(user)]

    in_train = np.zeros(len(user), dtype=bool)
    for s, e in zip(starts, ends):
        purchases = s + np.flatnonzero(behavior[s:e] == PURCHASE)
        in_train[s:purchases[-2]] = True
    pop_items, pop_counts = np.unique(item[in_train], return_counts=True)
    cumulative = np.cumsum(pop_counts, dtype=np.float64)
    rng = np.random.default_rng(SEED)

    header = ('session_id:token\titem_id_list:token_seq\titem_type_list:token_seq\t'
              'item_time_list:float_seq\titem_id:token\tneg_item_list:token_seq\n')
    files = {split: open(out_dir / f'{name}.{split}.inter', 'w', encoding='utf-8', newline='\n')
             for split in ('train', 'valid', 'test')}
    for f in files.values():
        f.write(header)
    counts = {'train': 0, 'valid': 0, 'test': 0}
    for n, (s, e) in enumerate(zip(starts, ends)):
        u_items, u_beh, u_time = item[s:e], behavior[s:e], hours[s:e] - hours[s]
        targets = np.flatnonzero(u_beh == PURCHASE)
        splits = ['train'] * (len(targets) - 2) + ['valid', 'test']
        for pos, split in zip(targets, splits):
            if pos == 0:
                continue
            lo = max(0, pos - MAX_HISTORY)
            negatives = '' if split == 'train' else fmt_tokens(sample_negatives(
                rng, cumulative, pop_items, u_items[pos], NUM_NEGATIVES))
            files[split].write('\t'.join((
                str(user[s]), fmt_tokens(u_items[lo:pos]), fmt_tokens(u_beh[lo:pos]),
                ' '.join(f'{t:.4f}' for t in u_time[lo:pos]), str(u_items[pos]), negatives)) + '\n')
            counts[split] += 1
        if (n + 1) % 50_000 == 0:
            print(f'  wrote {n + 1:,} users', flush=True)
    for f in files.values():
        f.close()

    n_users, n_items = len(starts), len(np.unique(item))
    stats = {
        'dataset': name, 'source': args.source, 'raw_interactions': int(raw_rows),
        'users': int(n_users), 'items': int(n_items), 'interactions': int(len(user)),
        'avg_length': round(len(user) / n_users, 2),
        'density': float(len(user) / (n_users * n_items)),
        'behaviors': {BEHAVIOR_NAMES[b]: int(c) for b, c in
                      zip(*np.unique(behavior, return_counts=True))},
        'samples': counts,
        'time_unit': 'hours',
        'settings': {'min_item_interactions': MIN_ITEM_INTERACTIONS, 'min_purchases': MIN_PURCHASES,
                     'max_history': MAX_HISTORY, 'num_negatives': NUM_NEGATIVES, 'seed': SEED},
    }
    (out_dir / f'{name}.stats.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
