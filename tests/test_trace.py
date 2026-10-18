"""Exactness and smoke tests for TRACE (run: python -m pytest tests/test_trace.py)."""

import itertools
import math
import unittest

import torch

from recbole.model.sequential_recommender.trace import ABLATIONS, TRACE


class FakeConfig(dict):
    def __getitem__(self, key):
        return self.get(key)


class FakeDataset:
    def __init__(self, n_items, n_behaviors):
        self.n_items = n_items
        self.n_behaviors = n_behaviors
        # token '0' (purchase) -> id 1, padding -> 0
        self.field2token_id = {
            'item_type_list': {'[PAD]': 0, **{str(i): i + 1 for i in range(n_behaviors - 1)}}
        }

    def num(self, field):
        return self.n_items if field == 'item_id' else self.n_behaviors


def make_model(ablation='none', omega=3, tau=0.5, blocks=2, heads=2):
    config = FakeConfig(
        USER_ID_FIELD='session_id', ITEM_ID_FIELD='item_id', LIST_SUFFIX='_list',
        ITEM_LIST_LENGTH_FIELD='item_length', NEG_PREFIX='neg_',
        MAX_ITEM_LIST_LENGTH=16, hidden_size=8, path_temperature=tau,
        n_path_blocks=blocks, max_alternatives=omega, reg_weight=1e-4,
        target_behavior_token='0', n_heads=heads, inner_size=16, dropout_prob=0.0,
        layer_norm_eps=1e-12, train_neg_num=3, candidate_chunk_size=2,
        train_micro_batch=2,
        trace_ablation=ablation,
    )
    torch.manual_seed(0)
    model = TRACE(config, FakeDataset(n_items=30, n_behaviors=4))
    model.eval()
    return model


def batch(lengths=(5, 3)):
    torch.manual_seed(1)
    width = max(lengths)
    items = torch.zeros(len(lengths), width, dtype=torch.long)
    types = torch.zeros_like(items)
    times = torch.zeros(len(lengths), width)
    for row, length in enumerate(lengths):
        items[row, :length] = torch.randint(1, 30, (length,))
        types[row, :length] = torch.randint(1, 4, (length,))
        times[row, :length] = torch.cumsum(torch.rand(length) * 3, 0)
    return items, types, times


def utilities(model, items, types, times, candidates):
    items, types, times, lengths = model._trim(items, types, times)
    valid = torch.arange(items.size(1)).unsqueeze(0) < lengths.unsqueeze(1)
    h, dh, m = model._encode_history(items, types, lengths)
    edge, _ = model._transition_utility(dh, types, times, valid)
    z = model._candidate_representation(candidates)
    node, endpoint = model._candidate_utilities(z, h, m, types, times, lengths, valid)
    return node, edge, endpoint, valid, lengths


def path_utility(node, edge, endpoint, b, c, path):
    value = sum(node[b, c, p] for p in path[:-1])
    value = value + sum(edge[b, p, q] for p, q in zip(path[:-1], path[1:]))
    return float(value + endpoint[b, c, path[-1]])


def all_paths(length):
    for k in range(1, length + 1):
        yield from itertools.combinations(range(length), k)


class TraceExactnessTest(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.items, self.types, self.times = batch()
        self.candidates = torch.tensor([[3, 7], [11, 2]])
        with torch.no_grad():
            (self.node, self.edge, self.endpoint,
             self.valid, self.lengths) = utilities(
                self.model, self.items, self.types, self.times, self.candidates)

    def test_ranked_paths_match_enumeration(self):
        paths, path_len = self.model._ranked_paths(
            self.node, self.edge, self.endpoint, self.valid)
        beam = self.model.omega + 1
        for b, length in enumerate(self.lengths.tolist()):
            for c in range(self.candidates.size(1)):
                brute = sorted(
                    (path_utility(self.node, self.edge, self.endpoint, b, c, p), p)
                    for p in all_paths(length)
                )[::-1]
                expected = brute[:min(beam, 2 ** length - 1)]
                found = []
                for n in range(beam):
                    k = int(path_len[b, c, n])
                    if k:
                        p = tuple(paths[b, c, n, :k].tolist())
                        found.append((path_utility(self.node, self.edge, self.endpoint, b, c, p), p))
                self.assertEqual(len(found), len(expected))
                self.assertEqual(len(set(p for _, p in found)), len(found))
                for (u_f, _), (u_e, _) in zip(found, expected):
                    self.assertAlmostEqual(u_f, u_e, places=4)
                self.assertEqual(found[0][1], expected[0][1])  # MAP path

    def test_suffix_dp_and_connection_probabilities(self):
        tau = self.model.tau
        log_beta = self.model._log_suffix_weights(
            self.node, self.edge, self.endpoint, self.valid)
        for b, length in enumerate(self.lengths.tolist()):
            for c in range(self.candidates.size(1)):
                weights = {p: math.exp(path_utility(self.node, self.edge, self.endpoint, b, c, p) / tau)
                           for p in all_paths(length)}
                # Denominator of Eq. (5) equals sum_l beta^l.
                partition = sum(weights.values())
                self.assertAlmostEqual(
                    float(torch.logsumexp(log_beta[b, c, :length], 0)),
                    math.log(partition), places=4)
                for p in range(length):
                    visits = sum(w for path, w in weights.items() if p in path)
                    stop = math.exp(float(self.endpoint[b, c, p]) / tau - float(log_beta[b, c, p]))
                    total = stop
                    for q in range(p + 1, length):
                        log_eta = (float(self.node[b, c, p] + self.edge[b, p, q]) / tau
                                   + float(log_beta[b, c, q] - log_beta[b, c, p]))
                        brute = sum(w for path, w in weights.items()
                                    if p in path and path.index(p) + 1 < len(path)
                                    and path[path.index(p) + 1] == q) / visits
                        self.assertAlmostEqual(math.exp(log_eta), brute, places=5)
                        total += math.exp(log_eta)
                    self.assertAlmostEqual(total, 1.0, places=5)


class TraceTrainingTest(unittest.TestCase):
    def test_loss_and_gradients(self):
        model = make_model()
        model.train()
        items, types, times = batch((6, 2, 1))
        interaction = {
            'item_id_list': items, 'item_type_list': types,
            'item_time_list': times, 'item_id': torch.tensor([4, 5, 6]),
        }
        loss = model.calculate_loss(interaction)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ('lambda_len_raw', 'lambda_tr_raw', 'lambda_end_raw',
                     'behavior_projection', 'gate.weight'):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
        self.assertGreater(float(model.lambda_len_raw.grad.abs()), 0.0)

    def test_scores_are_bounded_and_all_ablations_run(self):
        items, types, times = batch((5, 3))
        candidates = torch.tensor([[3, 7, 9], [11, 2, 1]])
        for ablation in ABLATIONS:
            model = make_model(ablation)
            with torch.no_grad():
                scores = model.candidate_scores(items, types, candidates, times)
            self.assertEqual(scores.shape, (2, 3), ablation)
            self.assertTrue(torch.isfinite(scores).all(), ablation)
            self.assertTrue(scores.abs().le(1.0 + 1e-5).all(), ablation)

    def test_heads_not_dividing_hidden_size(self):
        model = make_model(heads=10)  # hidden_size 8, 10 heads of size 1
        model.train()
        items, types, times = batch((5, 3))
        interaction = {'item_id_list': items, 'item_type_list': types,
                       'item_time_list': times, 'item_id': torch.tensor([4, 5])}
        loss = model.calculate_loss(interaction)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(model.blocks[0].query.out_features, 10)

    def test_grouped_predict_matches_rowwise(self):
        model = make_model()
        items, types, times = batch((5, 3))
        # Two evaluation instances with three candidate rows each, interleaved.
        rows = torch.tensor([0, 1, 0, 1, 0, 1])
        candidates = torch.tensor([4, 9, 7, 2, 12, 5])
        interaction = {'item_id_list': items[rows], 'item_type_list': types[rows],
                       'item_time_list': times[rows], 'item_id': candidates,
                       'session_id': rows + 10}
        with torch.no_grad():
            grouped = model.predict(interaction)
            rowwise = model.candidate_scores(
                items[rows], types[rows], candidates.view(-1, 1), times[rows]).squeeze(1)
        self.assertTrue(torch.allclose(grouped, rowwise, atol=1e-5))

    def test_predict_without_timestamps(self):
        model = make_model()
        items, types, _ = batch((4, 2))
        interaction = {'item_id_list': items, 'item_type_list': types,
                       'item_id': torch.tensor([3, 8])}
        with torch.no_grad():
            self.assertEqual(model.predict(interaction).shape, (2,))
            self.assertEqual(model.full_sort_predict(interaction).shape, (2, 30))


if __name__ == '__main__':
    unittest.main()
