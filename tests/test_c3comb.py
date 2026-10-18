import copy
import itertools
import math
import unittest

import torch
from torch import nn

from recbole.model.sequential_recommender.c3comb import C3Comb
from recbole.trainer.trainer import (
    RepeatGroupMetricAccumulator,
    warmup_cosine_multiplier,
)


def tiny_model(n_items=16, hidden_size=3):
    model = C3Comb.__new__(C3Comb)
    nn.Module.__init__(model)
    model.n_items = n_items
    model.hidden_size = hidden_size
    model.num_behaviors = 4
    model.buy_type = 3
    model.combo_min_temperature = 0.05
    model.combo_learnable_scale = True
    model.combo_pop_alpha = 0.75
    model.combo_uniform_weight = 0.4
    model.combo_pop_weight = 0.3
    model.combo_hard_weight = 0.3
    model.combo_candidate_chunk_size = 2
    model.ITEM_SEQ = 'item_id_list'
    model.POS_ITEM_ID = 'item_id'
    model.ITEM_ID = 'item_id'
    model.item_embedding = nn.Embedding(n_items, hidden_size, padding_idx=0)
    model.behavior_embedding = nn.Embedding(4, hidden_size, padding_idx=0)
    identity = torch.eye(hidden_size).view(1, 1, hidden_size, hidden_size)
    model.relation = nn.Parameter(identity.repeat(4, 4, 1, 1))
    model.message_gate_raw = nn.Parameter(torch.full((4, 4), -2.1972246))
    model.temporal_sensitivity_raw = nn.Parameter(torch.full((4, 4), -2.0))
    desired_offset = 0.7 - model.combo_min_temperature
    model.structural_temperature_raw = nn.Parameter(
        torch.tensor(math.log(math.expm1(desired_offset)))
    )
    model.logit_scale_raw = nn.Parameter(torch.tensor(math.log(math.expm1(4.0))))
    popularity = torch.arange(n_items, dtype=torch.float32)
    popularity[0] = 0
    model.register_buffer('item_popularity', popularity)
    model.register_buffer(
        'sampling_popularity', popularity.pow(model.combo_pop_alpha)
    )
    model.register_buffer('sampling_source_counts', torch.zeros(3, dtype=torch.long))
    model.register_buffer('sampling_requested_hard', torch.zeros((), dtype=torch.long))
    model.register_buffer('sampling_reallocated_hard', torch.zeros((), dtype=torch.long))
    model.register_buffer('sampling_rows', torch.zeros((), dtype=torch.long))
    model.register_buffer('sampling_rows_without_hard', torch.zeros((), dtype=torch.long))
    with torch.no_grad():
        model.item_embedding.weight.normal_(0.0, 0.2)
        model.behavior_embedding.weight.normal_(0.0, 0.2)
        model.item_embedding.weight[0].zero_()
        model.behavior_embedding.weight[0].zero_()
    return model


def tiny_linear_model(n_items=16, hidden_size=3, channels=2):
    model = tiny_model(n_items=n_items, hidden_size=hidden_size)
    model.combo_linear_kernel = True
    model.combo_closed_form_kernel = True
    model.combo_kernel_channels = channels
    model.sequence_encoder = nn.GRU(
        hidden_size, hidden_size, batch_first=True
    )
    kernel_shape = (4, 4, channels, hidden_size)
    model.kernel_source = nn.Parameter(torch.empty(kernel_shape))
    model.kernel_destination = nn.Parameter(torch.empty(kernel_shape))
    model.kernel_decay_raw = nn.Parameter(torch.full((4, 4, channels), -2.0))
    model.scan_source = nn.Parameter(torch.empty(4, hidden_size))
    model.scan_destination = nn.Parameter(torch.empty(4, hidden_size))
    model.scan_decay_raw = nn.Parameter(torch.tensor(-2.0))
    with torch.no_grad():
        model.kernel_source.normal_(0.0, 0.15)
        model.kernel_destination.normal_(0.0, 0.15)
        model.scan_source.normal_(0.0, 0.15)
        model.scan_destination.normal_(0.0, 0.15)
        model.kernel_source[0].zero_()
        model.kernel_source[:, 0].zero_()
        model.kernel_destination[0].zero_()
        model.kernel_destination[:, 0].zero_()
        model.scan_source[0].zero_()
        model.scan_destination[0].zero_()
    return model


class ExactCombinatorialInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.model = tiny_model()
        self.items = torch.tensor([[1, 2, 3, 0]])
        self.types = torch.tensor([[1, 2, 1, 0]])
        self.candidates = torch.tensor([[5]])
        self.positions = torch.tensor([3])
        self.times = torch.tensor([[0.0, 1.5, 5.0, 0.0]])

    def test_partition_and_all_marginals_match_enumeration(self):
        node, edge, terminal, valid, valid_edge, _ = (
            self.model._relation_potentials(
                self.items,
                self.types,
                self.candidates,
                self.positions,
                self.times,
            )
        )
        (
            temperature,
            log_partition,
            _,
            _,
            node_marginal,
            edge_marginal,
            terminal_marginal,
        ) = self.model._exact_posterior(
            node, edge, terminal, valid, valid_edge
        )

        paths = []
        energies = []
        for length in range(1, 4):
            for path in itertools.combinations(range(3), length):
                energy = node[0, 0, path[0]]
                for index in range(1, length):
                    energy = (
                        energy
                        + edge[0, path[index - 1], path[index]]
                        + node[0, 0, path[index]]
                    )
                energy = energy + terminal[0, path[-1]]
                paths.append(path)
                energies.append(energy)
        energies = torch.stack(energies)
        probabilities = torch.softmax(energies / temperature, dim=0)
        expected_log_partition = torch.logsumexp(energies / temperature, dim=0)
        expected_node = torch.zeros(3)
        expected_edge = torch.zeros(3, 3)
        expected_terminal = torch.zeros(3)
        for probability, path in zip(probabilities, paths):
            for node_index in path:
                expected_node[node_index] += probability
            for source, target in zip(path[:-1], path[1:]):
                expected_edge[source, target] += probability
            expected_terminal[path[-1]] += probability

        self.assertTrue(torch.allclose(log_partition[0, 0], expected_log_partition, atol=1e-5))
        self.assertTrue(torch.allclose(node_marginal[0, 0, :3], expected_node, atol=1e-5))
        self.assertTrue(torch.allclose(edge_marginal[0, 0, :3, :3], expected_edge, atol=1e-5))
        self.assertTrue(torch.allclose(terminal_marginal[0, 0, :3], expected_terminal, atol=1e-5))
        self.assertAlmostEqual(terminal_marginal[0, 0].sum().item(), 1.0, places=5)

    def test_gradients_reach_joint_embedding_structure_and_time_parameters(self):
        logits = self.model.candidate_logits(
            self.items,
            self.types,
            self.candidates,
            self.positions,
            self.times,
        )
        logits.sum().backward()
        self.assertIsNotNone(self.model.item_embedding.weight.grad)
        self.assertIsNotNone(self.model.relation.grad)
        self.assertIsNotNone(self.model.message_gate_raw.grad)
        self.assertIsNotNone(self.model.temporal_sensitivity_raw.grad)
        self.assertIsNotNone(self.model.structural_temperature_raw.grad)
        self.assertTrue(torch.isfinite(logits).all())

    def test_forward_only_logits_match_dense_prefix_suffix_inference(self):
        candidates = torch.tensor([[5, 6, 7]])
        forward_only = self.model.candidate_logits(
            self.items,
            self.types,
            candidates,
            self.positions,
            self.times,
        )
        dense, diagnostics = self.model.candidate_logits(
            self.items,
            self.types,
            candidates,
            self.positions,
            self.times,
            return_diagnostics=True,
        )
        self.assertTrue(torch.allclose(forward_only, dense, atol=1e-6))
        self.assertTrue(torch.isfinite(diagnostics['edge_marginal']).all())

    def test_physical_time_changes_edge_retention(self):
        short = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
        long = torch.tensor([[0.0, 10.0, 200.0, 0.0]])
        short_potentials = self.model._relation_potentials(
            self.items, self.types, self.candidates, self.positions, short
        )[1]
        long_potentials = self.model._relation_potentials(
            self.items, self.types, self.candidates, self.positions, long
        )[1]
        self.assertGreater(short_potentials[0, 0, 2], long_potentials[0, 0, 2])


class LinearSeparableInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2027)
        self.model = tiny_linear_model()
        self.items = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
        self.types = torch.tensor([[1, 2, 1, 0], [2, 3, 0, 0]])
        self.candidates = torch.tensor([[6, 7], [8, 9]])
        self.positions = torch.tensor([3, 2])
        self.times = torch.tensor(
            [[0.0, 1.5, 5.0, 0.0], [0.0, 12.0, 0.0, 0.0]]
        )

    def test_linear_logits_match_explicit_quadratic_kernel(self):
        linear = self.model.candidate_logits(
            self.items,
            self.types,
            self.candidates,
            self.positions,
            self.times,
        )
        dense, diagnostics = self.model.candidate_logits(
            self.items,
            self.types,
            self.candidates,
            self.positions,
            self.times,
            return_diagnostics=True,
        )
        self.assertTrue(torch.allclose(linear, dense, atol=2e-5, rtol=2e-5))
        self.assertTrue(torch.isfinite(diagnostics['log_partition']).all())

    def test_linear_and_quadratic_gradients_match(self):
        linear_model = copy.deepcopy(self.model)
        dense_model = copy.deepcopy(self.model)
        linear_loss = linear_model.candidate_logits(
            self.items,
            self.types,
            self.candidates,
            self.positions,
            self.times,
        ).square().sum()
        dense_loss = dense_model.candidate_logits(
            self.items,
            self.types,
            self.candidates,
            self.positions,
            self.times,
            return_diagnostics=True,
        )[0].square().sum()
        linear_loss.backward()
        dense_loss.backward()
        checked = (
            'item_embedding.weight',
            'behavior_embedding.weight',
            'relation',
            'temporal_sensitivity_raw',
            'sequence_encoder.weight_ih_l0',
            'sequence_encoder.weight_hh_l0',
            'scan_source',
            'scan_destination',
            'scan_decay_raw',
            'logit_scale_raw',
        )
        linear_parameters = dict(linear_model.named_parameters())
        dense_parameters = dict(dense_model.named_parameters())
        for name in checked:
            self.assertIsNotNone(linear_parameters[name].grad, name)
            self.assertIsNotNone(dense_parameters[name].grad, name)
            self.assertTrue(
                torch.allclose(
                    linear_parameters[name].grad,
                    dense_parameters[name].grad,
                    atol=5e-5,
                    rtol=5e-4,
                ),
                name,
            )


class StructureAwareNegativeSamplerTest(unittest.TestCase):
    def test_hard_sampler_uses_history_but_excludes_current_positive(self):
        model = tiny_model(n_items=128)
        model.combo_hard_weight = 1.0
        model.combo_pop_weight = 0.0
        model.combo_uniform_weight = 0.0
        negatives = model._sample_candidates(
            positives=torch.tensor([5]),
            item_seq=torch.tensor([[2, 3, 4, 0]]),
            type_seq=torch.tensor([[1, 2, 3, 0]]),
            count=3,
        )
        self.assertEqual(set(negatives[0].tolist()), {2, 3, 4})
        self.assertNotIn(5, negatives[0].tolist())

    def test_complete_prefix_keeps_all_valid_events(self):
        model = tiny_model(n_items=128)
        items = torch.tensor([
            [2, 3, 4, 5, 6, 0, 0],
            [8, 9, 10, 0, 0, 0, 0],
        ])
        types = torch.tensor([
            [1, 2, 3, 1, 2, 0, 0],
            [1, 3, 2, 0, 0, 0, 0],
        ])
        times = torch.tensor([
            [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0],
            [0.0, 2.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        ])
        complete_items, complete_types, complete_times, lengths = (
            model._complete_prefix(items, types, times)
        )
        self.assertEqual(complete_items.shape[1], 5)
        self.assertTrue(torch.equal(lengths, torch.tensor([5, 3])))
        self.assertTrue(torch.equal(complete_items[0], items[0, :5]))
        self.assertTrue(torch.equal(complete_types[0], types[0, :5]))
        self.assertTrue(torch.equal(complete_times[0], times[0, :5]))

    def test_candidate_chunking_matches_joint_scoring(self):
        model = tiny_model(n_items=128)
        model.eval()
        items = torch.tensor([[2, 3, 4, 0]])
        types = torch.tensor([[1, 2, 3, 0]])
        times = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
        candidates = torch.tensor([[5, 6, 7, 8, 9]])
        positions = torch.tensor([3])
        joint = model.candidate_logits(
            items, types, candidates, positions, times
        )
        chunked = model._chunked_candidate_logits(
            items, types, candidates, positions, times
        )
        self.assertTrue(torch.allclose(joint, chunked, atol=1e-6))

    def test_checkpointed_candidate_chunks_preserve_parameter_gradients(self):
        model = tiny_model(n_items=128)
        model.train()
        items = torch.tensor([[2, 3, 4, 0]])
        types = torch.tensor([[1, 2, 3, 0]])
        times = torch.tensor([[0.0, 1.0, 2.0, 0.0]])
        candidates = torch.tensor([[5, 6, 7, 8, 9]])
        logits = model._chunked_candidate_logits(
            items,
            types,
            candidates,
            torch.tensor([3]),
            times,
            checkpoint_grad=True,
        )
        logits.sum().backward()
        self.assertIsNotNone(model.relation.grad)
        self.assertIsNotNone(model.item_embedding.weight.grad)
        self.assertTrue(torch.isfinite(model.relation.grad).all())

    def test_missing_hard_slots_are_reallocated_and_recorded(self):
        model = tiny_model(n_items=128)
        model.combo_hard_weight = 1.0
        model.combo_pop_weight = 0.0
        model.combo_uniform_weight = 0.0
        negatives = model._sample_candidates(
            positives=torch.tensor([5]),
            item_seq=torch.tensor([[5, 5, 0, 0]]),
            type_seq=torch.tensor([[3, 3, 0, 0]]),
            count=3,
        )
        stats = model.sampling_statistics()
        self.assertTrue(torch.all(negatives.ne(0)))
        self.assertTrue(torch.all(negatives.ne(5)))
        self.assertEqual(stats['requested_hard_count'], 3)
        self.assertEqual(stats['reallocated_hard_count'], 3)
        self.assertEqual(stats['structural_hard_count'], 0)
        self.assertEqual(stats['rows_without_hard'], 1)


class TrainingProtocolTest(unittest.TestCase):
    def test_warmup_cosine_has_expected_boundaries(self):
        self.assertAlmostEqual(warmup_cosine_multiplier(0, 500, 2000), 0.002)
        self.assertAlmostEqual(warmup_cosine_multiplier(499, 500, 2000), 1.0)
        self.assertAlmostEqual(warmup_cosine_multiplier(2000, 500, 2000), 0.0)

    def test_repeat_group_metrics_macro_balance_groups(self):
        accumulator = RepeatGroupMetricAccumulator([1, 2])
        scores = torch.tensor([
            [3.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
        ])
        items = torch.tensor([[4, 5], [7, 8]])
        positives = torch.tensor([5, 9])
        accumulator.update(scores, items, positives)
        result = accumulator.result(decimal_place=6)
        self.assertEqual(result['repeat_count'], 1)
        self.assertEqual(result['nonrepeat_count'], 1)
        self.assertEqual(result['repeat_ndcg@1'], 1.0)
        self.assertEqual(result['nonrepeat_ndcg@1'], 0.0)
        self.assertEqual(result['balanced_ndcg@1'], 0.5)

    def test_repeat_group_metrics_accept_catalog_item_indices(self):
        accumulator = RepeatGroupMetricAccumulator([1, 2])
        scores = torch.tensor([
            [float('-inf'), 1.0, 3.0, 2.0],
            [float('-inf'), 3.0, 2.0, 1.0],
        ])
        items = torch.tensor([[2, 4], [3, 4]])
        positives = torch.tensor([2, 3])
        accumulator.update(
            scores, items, positives, positive_indices=positives
        )
        result = accumulator.result(decimal_place=6)
        self.assertEqual(result['repeat_count'], 2)
        self.assertEqual(result['overall_ndcg@1'], 0.5)


if __name__ == '__main__':
    unittest.main()
