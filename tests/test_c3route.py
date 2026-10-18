import unittest
import itertools

import torch
from torch import nn

from recbole.model.sequential_recommender.mbht import MBHT


class CandidateRouteScoreTest(unittest.TestCase):
    def setUp(self):
        self.model = MBHT.__new__(MBHT)
        nn.Module.__init__(self.model)
        self.model.n_items = 6
        self.model.mask_token = 6
        self.model.route_topk = 1
        self.model.item_embedding = nn.Embedding(7, 4, padding_idx=0)
        self.model.route_type_bias = nn.Embedding(6, 1, padding_idx=0)
        self.model.route_recency_raw = nn.Parameter(torch.tensor(-2.0))
        self.model.route_temperature_raw = nn.Parameter(torch.tensor(0.0))

        with torch.no_grad():
            self.model.item_embedding.weight.zero_()
            self.model.item_embedding.weight[1, 0] = 1.0
            self.model.item_embedding.weight[2, 1] = 1.0
            self.model.item_embedding.weight[3, 2] = 1.0
            self.model.route_type_bias.weight.zero_()

    def test_exact_candidate_match_receives_higher_route_score(self):
        item_seq = torch.tensor([[1, 2, 6, 0]])
        type_seq = torch.tensor([[1, 2, 0, 0]])
        candidates = torch.tensor([[1, 3]])
        target_positions = torch.tensor([2])

        scores = self.model.candidate_route_scores(
            item_seq, type_seq, candidates, target_positions
        )

        self.assertGreater(scores[0, 0].item(), scores[0, 1].item())

    def test_route_is_causal(self):
        item_seq = torch.tensor([[2, 1, 6, 0]])
        type_seq = torch.tensor([[1, 2, 0, 0]])
        candidates = torch.tensor([[1]])

        before_match = self.model.candidate_route_scores(
            item_seq, type_seq, candidates, torch.tensor([1])
        )
        after_match = self.model.candidate_route_scores(
            item_seq, type_seq, candidates, torch.tensor([2])
        )

        self.assertLess(before_match.item(), after_match.item())

    def test_soft_surrogate_reaches_route_parameters(self):
        item_seq = torch.tensor([[1, 2, 6, 0]])
        type_seq = torch.tensor([[1, 2, 0, 0]])
        candidates = torch.tensor([[1]])
        target_positions = torch.tensor([2])

        scores, coverage = self.model.candidate_route_scores(
            item_seq,
            type_seq,
            candidates,
            target_positions,
            return_coverage=True,
        )
        (-scores.mean() + coverage).backward()

        self.assertIsNotNone(self.model.route_type_bias.weight.grad)
        self.assertIsNotNone(self.model.route_recency_raw.grad)


class HypergraphConstructionTest(unittest.TestCase):
    def test_vectorized_builder_matches_author_reference(self):
        model = MBHT.__new__(MBHT)
        model.mask_token = 99
        seqs = torch.tensor([
            [5, 7, 5, 9, 99, 0, 0],
            [4, 4, 8, 3, 8, 99, 0],
        ])
        generator = torch.Generator().manual_seed(2026)
        raw = torch.rand((2, 7, 7), generator=generator)
        item_sim = (raw + raw.transpose(1, 2)) / 2
        item_sim.diagonal(dim1=1, dim2=2).fill_(1.0)

        fast = model.build_Gs_unique(seqs, item_sim.clone(), group_len=3)
        batched = model.build_Gs_unique(seqs, item_sim.clone(), group_len=3, batched=True)
        fully_batched = model.build_Gs_fully_batched(
            seqs, item_sim.clone(), group_len=3
        )
        reference = model.build_Gs_unique_reference(seqs, item_sim.clone(), group_len=3)
        self.assertTrue(torch.allclose(fast, reference, atol=1e-6, rtol=1e-6))
        sizes = torch.count_nonzero(seqs, dim=1).tolist()
        offset = 0
        for batch_idx, size in enumerate(sizes):
            expected = reference[offset:offset + size, offset:offset + size]
            self.assertTrue(torch.allclose(batched[batch_idx, :size, :size], expected))
            self.assertTrue(
                torch.allclose(
                    fully_batched[batch_idx, :size, :size],
                    expected,
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
            self.assertEqual(torch.count_nonzero(batched[batch_idx, size:]).item(), 0)
            self.assertEqual(torch.count_nonzero(batched[batch_idx, :, size:]).item(), 0)
            self.assertEqual(
                torch.count_nonzero(fully_batched[batch_idx, size:]).item(), 0
            )
            self.assertEqual(
                torch.count_nonzero(fully_batched[batch_idx, :, size:]).item(), 0
            )
            offset += size


class GlobalTrajectoryRouteTest(unittest.TestCase):
    def setUp(self):
        self.model = MBHT.__new__(MBHT)
        nn.Module.__init__(self.model)
        self.model.n_items = 8
        self.model.mask_token = 8
        self.model.hidden_size = 4
        self.model.route_mode = 'global'
        self.model.route_topk = 4
        self.model.route_stages = 3
        self.model.route_dp_temperature = 0.7
        self.model.item_embedding = nn.Embedding(9, 4, padding_idx=0)
        self.model.route_type_bias = nn.Embedding(6, 1, padding_idx=0)
        self.model.route_recency_raw = nn.Parameter(torch.tensor(-2.0))
        self.model.route_stage_query = nn.Parameter(torch.zeros(3, 4))
        self.model.route_transition_src = nn.Linear(4, 4, bias=False)
        self.model.route_transition_dst = nn.Linear(4, 4, bias=False)
        self.model.route_behavior_transition = nn.Parameter(torch.zeros(6, 6))
        self.model.route_gap_raw = nn.Parameter(torch.full((3,), -2.0))
        self.model.route_time_projection = nn.Linear(3, 4, bias=False)

        with torch.no_grad():
            self.model.item_embedding.weight.normal_(0.0, 0.2)
            self.model.item_embedding.weight[0].zero_()
            self.model.route_type_bias.weight.zero_()
            self.model.route_transition_src.weight.copy_(torch.eye(4))
            self.model.route_transition_dst.weight.copy_(torch.eye(4))

    def test_viterbi_energy_matches_exhaustive_enumeration(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 8, 0]])
        type_seq = torch.tensor([[1, 2, 3, 1, 0, 0]])
        candidates = torch.tensor([[5, 6]])
        _, diagnostics = self.model._candidate_global_route_scores(
            item_seq,
            type_seq,
            candidates,
            torch.tensor([4]),
            return_diagnostics=True,
        )

        unary = diagnostics['unary']
        transitions = diagnostics['transitions']
        terminal = diagnostics['terminal']
        for candidate_idx in range(candidates.size(1)):
            energies = []
            for path in itertools.combinations(range(self.model.route_topk), 3):
                energy = unary[0, candidate_idx, 0, path[0]]
                for stage in range(1, 3):
                    energy = energy + unary[0, candidate_idx, stage, path[stage]]
                    energy = energy + transitions[
                        0, candidate_idx, stage - 1, path[stage - 1], path[stage]
                    ]
                energy = energy + terminal[0, candidate_idx, path[-1]]
                energies.append(energy)
            expected = torch.stack(energies).max()
            self.assertTrue(
                torch.allclose(
                    diagnostics['best_energy'][0, candidate_idx],
                    expected,
                    atol=1e-6,
                    rtol=1e-6,
                )
            )

    def test_global_soft_path_reaches_transition_parameters(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 8, 0]])
        type_seq = torch.tensor([[1, 2, 3, 1, 0, 0]])
        candidates = torch.tensor([[5, 6]])
        scores, coverage = self.model.candidate_route_scores(
            item_seq,
            type_seq,
            candidates,
            torch.tensor([4]),
            return_coverage=True,
        )
        (-scores.mean() + coverage).backward()
        self.assertIsNotNone(self.model.route_transition_src.weight.grad)
        self.assertTrue(torch.isfinite(scores).all())

    def test_soft_marginals_match_enumeration_and_partition_gradient(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 8, 0]])
        type_seq = torch.tensor([[1, 2, 3, 1, 0, 0]])
        candidates = torch.tensor([[5]])
        _, diagnostics = self.model._candidate_global_route_scores(
            item_seq,
            type_seq,
            candidates,
            torch.tensor([4]),
            return_diagnostics=True,
        )
        unary = diagnostics['unary']
        transitions = diagnostics['transitions']
        terminal = diagnostics['terminal']
        paths = list(itertools.combinations(range(self.model.route_topk), 3))
        energies = []
        for path in paths:
            energy = unary[0, 0, 0, path[0]]
            for stage in range(1, 3):
                energy = energy + unary[0, 0, stage, path[stage]]
                energy = energy + transitions[
                    0, 0, stage - 1, path[stage - 1], path[stage]
                ]
            energies.append(energy + terminal[0, 0, path[-1]])
        probabilities = torch.softmax(
            torch.stack(energies) / self.model.route_dp_temperature, dim=0
        )
        expected = torch.zeros_like(diagnostics['soft_path'][0, 0])
        for probability, path in zip(probabilities, paths):
            for stage, node in enumerate(path):
                expected[stage, node] += probability
        self.assertTrue(
            torch.allclose(
                diagnostics['soft_path'][0, 0], expected, atol=1e-5, rtol=1e-5
            )
        )

        unary.retain_grad()
        diagnostics['log_partition'].sum().backward()
        self.assertTrue(
            torch.allclose(
                unary.grad[0, 0],
                diagnostics['soft_path'][0, 0],
                atol=1e-5,
                rtol=1e-5,
            )
        )

    def test_invalid_route_budget_is_rejected(self):
        self.model.route_topk = 2
        with self.assertRaises(ValueError):
            self.model.candidate_route_scores(
                torch.tensor([[1, 2, 3, 8]]),
                torch.tensor([[1, 1, 1, 0]]),
                torch.tensor([[4]]),
                torch.tensor([3]),
            )

    def test_invalid_nodes_never_beat_extremely_negative_valid_nodes(self):
        with torch.no_grad():
            self.model.route_type_bias.weight.fill_(-1e6)
            self.model.route_type_bias.weight[0].zero_()
        _, diagnostics = self.model._candidate_global_route_scores(
            torch.tensor([[1, 2, 3, 4, 8, 0]]),
            torch.tensor([[1, 2, 3, 1, 0, 0]]),
            torch.tensor([[5]]),
            torch.tensor([4]),
            return_diagnostics=True,
        )
        self.assertEqual(diagnostics['selected_valid'].sum().item(), 4)
        self.assertTrue(torch.isfinite(diagnostics['best_energy']).all())

    def test_too_short_history_returns_finite_zero_route_score(self):
        scores, diagnostics = self.model._candidate_global_route_scores(
            torch.tensor([[1, 2, 8, 0]]),
            torch.tensor([[1, 2, 0, 0]]),
            torch.tensor([[5]]),
            torch.tensor([2]),
            return_diagnostics=True,
        )
        self.assertTrue(torch.isfinite(scores).all())
        self.assertEqual(scores.item(), 0.0)
        self.assertEqual(diagnostics['hard_path'].sum().item(), 0.0)

    def test_tied_dynamic_program_is_deterministic(self):
        with torch.no_grad():
            self.model.item_embedding.weight.zero_()
            self.model.route_type_bias.weight.zero_()
            self.model.route_stage_query.zero_()
            self.model.route_behavior_transition.zero_()
            self.model.route_transition_src.weight.zero_()
            self.model.route_transition_dst.weight.zero_()
            self.model.route_recency_raw.fill_(float('-inf'))
            self.model.route_gap_raw.fill_(float('-inf'))
        kwargs = dict(
            item_seq=torch.tensor([[1, 2, 3, 4, 8, 0]]),
            type_seq=torch.tensor([[1, 1, 1, 1, 0, 0]]),
            candidates=torch.tensor([[5]]),
            target_positions=torch.tensor([4]),
            return_diagnostics=True,
        )
        _, first = self.model._candidate_global_route_scores(**kwargs)
        _, second = self.model._candidate_global_route_scores(**kwargs)
        self.assertTrue(torch.equal(first['hard_path'], second['hard_path']))
        selected = first['hard_path'][0, 0].argmax(dim=-1).tolist()
        self.assertEqual(selected, [0, 1, 2])

    def test_candidate_horizon_changes_global_route_score(self):
        item_seq = torch.tensor([[1, 2, 3, 4, 8, 0]])
        type_seq = torch.tensor([[1, 2, 3, 1, 0, 0]])
        candidates = torch.tensor([[5]])
        short = self.model.candidate_route_scores(
            item_seq, type_seq, candidates, torch.tensor([4]),
            target_horizons=torch.tensor([0.25])
        )
        long = self.model.candidate_route_scores(
            item_seq, type_seq, candidates, torch.tensor([4]),
            target_horizons=torch.tensor([168.0])
        )
        self.assertFalse(torch.allclose(short, long))


class ContinuousTimeStateTest(unittest.TestCase):
    def test_irregular_gaps_change_states_and_receive_gradients(self):
        model = MBHT.__new__(MBHT)
        nn.Module.__init__(model)
        model.mask_token = 8
        model.hidden_size = 4
        model.temporal_decay_raw = nn.Embedding(6, 4, padding_idx=0)
        model.temporal_center = nn.Embedding(6, 4, padding_idx=0)
        model.temporal_time_projection = nn.Linear(3, 4)
        model.temporal_gru = nn.GRUCell(4, 4)
        with torch.no_grad():
            model.temporal_decay_raw.weight.fill_(-2.0)
            model.temporal_decay_raw.weight[0].zero_()

        inputs = torch.randn(1, 4, 4, requires_grad=True)
        items = torch.tensor([[1, 2, 3, 0]])
        types = torch.tensor([[1, 2, 1, 0]])
        short_times = torch.tensor([[0.0, 0.1, 0.2, 0.0]])
        long_times = torch.tensor([[0.0, 10.0, 100.0, 0.0]])
        short_states = model._continuous_time_states(
            inputs, items, types, short_times
        )
        long_states = model._continuous_time_states(
            inputs, items, types, long_times
        )
        self.assertTrue(torch.isfinite(long_states).all())
        self.assertFalse(torch.allclose(short_states, long_states))
        long_states.sum().backward()
        self.assertIsNotNone(model.temporal_decay_raw.weight.grad)

    def test_vectorized_masking_matches_target_only_reference(self):
        model = MBHT.__new__(MBHT)
        model.mask_token = 99
        model.mask_ratio = 0.0
        model.mask_item_length = 4
        model.buy_type = 1
        items = torch.tensor([[5, 7, 0, 0], [4, 8, 3, 0]])
        types = torch.tensor([[2, 2, 0, 0], [3, 2, 2, 0]])
        targets = torch.tensor([11, 12])

        fast = model.reconstruct_train_data(items.clone(), types.clone(), targets.clone())
        reference = model.reconstruct_train_data_reference(
            items.clone(), types.clone(), targets.clone()
        )
        for fast_tensor, reference_tensor in zip(fast, reference):
            self.assertTrue(torch.equal(fast_tensor, reference_tensor))

    def test_batched_window_readout_matches_loop_reference(self):
        model = MBHT.__new__(MBHT)
        model.mask_token = 99
        model.sw_before = 2
        model.sw_follow = 2
        generator = torch.Generator().manual_seed(2026)
        embeddings = torch.rand((2, 7, 5), generator=generator)
        sequences = torch.tensor([
            [1, 2, 99, 4, 99, 0, 0],
            [5, 99, 7, 8, 9, 99, 0],
        ])
        masked_positions = torch.tensor([
            [0, 0, 2, 4],
            [0, 1, 3, 5],
        ])
        mask_counts = torch.tensor([2, 3])

        fast = model._masked_window_readout(
            embeddings, sequences, (masked_positions, mask_counts)
        )
        reference = model._masked_window_readout_reference(
            embeddings, sequences, (masked_positions, mask_counts)
        )
        self.assertTrue(torch.allclose(fast, reference, atol=1e-6, rtol=1e-6))

        fast_eval = model._masked_window_readout(embeddings, sequences)
        reference_eval = model._masked_window_readout_reference(embeddings, sequences)
        self.assertTrue(torch.allclose(fast_eval, reference_eval, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
