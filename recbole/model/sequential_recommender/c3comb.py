# -*- coding: utf-8 -*-
"""Candidate-conditioned combinatorial recommendation on MBHT datasets.

This module intentionally lives beside the original MBHT implementation.  It
uses the same RecBole data pipeline and item/behavior vocabularies, but replaces
the previous screened, fixed-length C3Route approximation with a complete-prefix
Gibbs dynamic program.  Its default separable positive edge kernel sums the
entire non-empty subsequence space in linear time; the unrestricted dense edge
model remains available as an explicit quadratic audit baseline.

The supervised objective is candidate-wise BCE.  Every candidate receives its
own posterior over all non-empty chronological subsequences, posterior-gated
message passing, and final history representation.  No path labels or auxiliary
transition losses are used.
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


class C3Comb(SequentialRecommender):
    """Exact candidate-conditioned combinatorial sequential recommender."""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(C3Comb, self).__init__(config, dataset)

        self.hidden_size = int(config['hidden_size'])
        self.initializer_range = float(config['initializer_range'])
        self.num_behaviors = int(dataset.num('item_type_list'))
        target_behavior_token = str(config['target_behavior_token'] or '0')
        behavior_token_ids = dataset.field2token_id['item_type_list']
        if target_behavior_token not in behavior_token_ids:
            raise ValueError(
                'Configured target behavior token {!r} is absent from the '
                'dataset behavior vocabulary {}.'.format(
                    target_behavior_token, sorted(behavior_token_ids)
                )
            )
        self.buy_type = behavior_token_ids[target_behavior_token]

        self.combo_negatives = int(config['combo_negatives'])
        self.combo_uniform_weight = float(config['combo_uniform_weight'])
        self.combo_pop_weight = float(config['combo_pop_weight'])
        self.combo_hard_weight = float(config['combo_hard_weight'])
        self.combo_pop_alpha = float(config['combo_pop_alpha'])
        self.combo_min_temperature = float(config['combo_min_temperature'])
        initial_temperature = float(config['combo_temperature'])
        # Checkpoints produced before candidate chunking do not contain this
        # runtime-only key.  The default preserves loadability without
        # restoring either the old history crop or the fixed evaluator.
        config_values = getattr(config, 'final_config_dict', {})
        self.combo_candidate_chunk_size = max(
            int(config_values.get('combo_candidate_chunk_size', 64)), 1
        )
        self.combo_linear_kernel = bool(
            int(config_values.get('combo_linear_kernel', 1))
        )
        self.combo_kernel_channels = max(
            int(config_values.get('combo_kernel_channels', 8)), 1
        )
        self.combo_closed_form_kernel = bool(
            int(config_values.get('combo_closed_form_kernel', 1))
        )
        self.combo_learnable_scale = bool(config['combo_learnable_scale'])
        initial_scale = float(config['combo_logit_scale'])

        mixture_sum = (
            self.combo_uniform_weight
            + self.combo_pop_weight
            + self.combo_hard_weight
        )
        if mixture_sum <= 0:
            raise ValueError('At least one negative-sampling weight must be positive.')
        self.combo_uniform_weight /= mixture_sum
        self.combo_pop_weight /= mixture_sum
        self.combo_hard_weight /= mixture_sum

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.behavior_embedding = nn.Embedding(
            self.num_behaviors, self.hidden_size, padding_idx=0
        )
        # A fused causal recurrent backbone supplies each event with its full
        # observed-prefix context before structured inference.  It is trained
        # jointly through the candidate-conditioned partition/readout; there
        # is no separately trained representation stage.
        self.sequence_encoder = nn.GRU(
            self.hidden_size, self.hidden_size, batch_first=True
        )

        # R[b, b'] is a full directed operator.  A zero-centred Xavier
        # initialization avoids giving an untrained model an identity/repeat
        # shortcut.  Propagation stability is handled explicitly by the
        # behavior-conditioned residual gate below.
        relation = torch.empty(
            self.num_behaviors,
            self.num_behaviors,
            self.hidden_size,
            self.hidden_size,
        )
        for source_behavior in range(1, self.num_behaviors):
            for destination_behavior in range(1, self.num_behaviors):
                nn.init.xavier_uniform_(
                    relation[source_behavior, destination_behavior]
                )
        relation[0].zero_()
        relation[:, 0].zero_()
        self.relation = nn.Parameter(relation)

        # g[b, b'] in (0, 1) controls the residual message magnitude for a
        # directed behavior pair.  Starting at 0.1 keeps the local event state
        # dominant without baking an identity relation into R.
        initial_gate = 0.1
        gate_logit = math.log(initial_gate / (1.0 - initial_gate))
        self.message_gate_raw = nn.Parameter(
            torch.full(
                (self.num_behaviors, self.num_behaviors), gate_logit
            )
        )

        # lambda[b, b'] = softplus(raw) is the non-negative physical-time
        # sensitivity in Eq. (11).  Padding relations never enter a valid edge.
        self.temporal_sensitivity_raw = nn.Parameter(
            torch.full((self.num_behaviors, self.num_behaviors), -3.0)
        )

        # Linear-C3Comb parameterizes the positive directed edge factor as a
        # finite sum of separable channels,
        #
        #   k(p,t) = H^{-1} sum_h exp(s_h(p) + d_h(t) - lambda_h dt).
        #
        # The resulting global path partition remains exact, while all past
        # events can be summarized by behavior-pair/channel accumulators.  No
        # edge screening, history crop, or top-k predecessor approximation is
        # introduced.
        kernel_shape = (
            self.num_behaviors,
            self.num_behaviors,
            self.combo_kernel_channels,
            self.hidden_size,
        )
        self.kernel_source = nn.Parameter(torch.empty(kernel_shape))
        self.kernel_destination = nn.Parameter(torch.empty(kernel_shape))
        self.kernel_decay_raw = nn.Parameter(
            torch.full(
                (
                    self.num_behaviors,
                    self.num_behaviors,
                    self.combo_kernel_channels,
                ),
                -3.0,
            )
        )
        self.scan_source = nn.Parameter(
            torch.empty(self.num_behaviors, self.hidden_size)
        )
        self.scan_destination = nn.Parameter(
            torch.empty(self.num_behaviors, self.hidden_size)
        )
        self.scan_decay_raw = nn.Parameter(torch.tensor(-3.0))
        temperature_offset = max(
            initial_temperature - self.combo_min_temperature, 1e-3
        )
        self.structural_temperature_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(temperature_offset)))
        )

        # A positive scale fixes the narrow dynamic range of cosine logits while
        # preserving their ranking.  Set combo_learnable_scale=0 and
        # combo_logit_scale=1 for the literal unscaled Eq. (34).
        initial_scale = max(initial_scale, 1e-3)
        inverse_softplus = math.log(math.expm1(initial_scale))
        scale_tensor = torch.tensor(inverse_softplus)
        if self.combo_learnable_scale:
            self.logit_scale_raw = nn.Parameter(scale_tensor)
        else:
            self.register_buffer('logit_scale_raw', scale_tensor)

        self.item_embedding.weight.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.behavior_embedding.weight.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.kernel_source.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.kernel_destination.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.scan_source.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.scan_destination.data.normal_(
            mean=0.0, std=self.initializer_range
        )
        self.item_embedding.weight.data[0].zero_()
        self.behavior_embedding.weight.data[0].zero_()
        self.kernel_source.data[0].zero_()
        self.kernel_source.data[:, 0].zero_()
        self.kernel_destination.data[0].zero_()
        self.kernel_destination.data[:, 0].zero_()
        self.scan_source.data[0].zero_()
        self.scan_destination.data[0].zero_()

        popularity = self._build_item_popularity(dataset)
        self.register_buffer('item_popularity', popularity)
        self.register_buffer(
            'sampling_popularity', popularity.pow(self.combo_pop_alpha)
        )
        # Source ids are [uniform, popularity, structural-hard].  These
        # counters make requested-vs-realized sampling auditable.
        self.register_buffer(
            'sampling_source_counts', torch.zeros(3, dtype=torch.long)
        )
        self.register_buffer(
            'sampling_requested_hard', torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            'sampling_reallocated_hard', torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            'sampling_rows', torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            'sampling_rows_without_hard', torch.zeros((), dtype=torch.long)
        )

    def _build_item_popularity(self, dataset):
        """Count training-prefix and target occurrences for reproducible sampling."""
        counts = torch.ones(self.n_items, dtype=torch.float32)
        counts[0] = 0.0
        interaction = dataset.inter_feat
        if self.ITEM_SEQ in interaction:
            sequence_items = interaction[self.ITEM_SEQ].reshape(-1).long().cpu()
            counts += torch.bincount(
                sequence_items, minlength=self.n_items
            )[:self.n_items].float()
        if self.ITEM_ID in interaction:
            target_items = interaction[self.ITEM_ID].reshape(-1).long().cpu()
            counts += torch.bincount(
                target_items, minlength=self.n_items
            )[:self.n_items].float()
        counts[0] = 0.0
        return counts

    def _temperature(self):
        return (
            F.softplus(self.structural_temperature_raw)
            + self.combo_min_temperature
        )

    def _logit_scale(self):
        return F.softplus(self.logit_scale_raw).clamp(max=100.0)

    def sampling_statistics(self):
        """Return realized sampler composition accumulated during training."""
        counts = self.sampling_source_counts.detach().cpu().tolist()
        total = max(sum(counts), 1)
        requested_hard = int(self.sampling_requested_hard.item())
        rows = max(int(self.sampling_rows.item()), 1)
        return {
            'uniform_count': int(counts[0]),
            'popularity_count': int(counts[1]),
            'structural_hard_count': int(counts[2]),
            'uniform_ratio': float(counts[0] / total),
            'popularity_ratio': float(counts[1] / total),
            'structural_hard_ratio': float(counts[2] / total),
            'requested_hard_count': requested_hard,
            'reallocated_hard_count': int(
                self.sampling_reallocated_hard.item()
            ),
            'hard_reallocation_ratio': float(
                self.sampling_reallocated_hard.item()
                / max(requested_hard, 1)
            ),
            'rows': int(self.sampling_rows.item()),
            'rows_without_hard': int(
                self.sampling_rows_without_hard.item()
            ),
            'rows_without_hard_ratio': float(
                self.sampling_rows_without_hard.item() / rows
            ),
        }

    def _time_values(self, item_seq, time_seq):
        if time_seq is not None:
            return time_seq.to(dtype=self.item_embedding.weight.dtype)
        positions = torch.arange(
            item_seq.size(1), device=item_seq.device,
            dtype=self.item_embedding.weight.dtype,
        )
        return positions.unsqueeze(0).expand(item_seq.size(0), -1)

    def _complete_prefix(self, item_seq, type_seq, time_seq=None):
        """Keep every valid prefix event; trim only all-padding columns."""
        lengths = torch.count_nonzero(item_seq, dim=1)
        if torch.any(lengths.eq(0)):
            raise ValueError('C3Comb requires a non-empty observable prefix.')
        batch_width = int(lengths.max().item())
        complete_items = item_seq[:, :batch_width]
        complete_types = type_seq[:, :batch_width]
        complete_times = (
            time_seq[:, :batch_width] if time_seq is not None else None
        )
        return complete_items, complete_types, complete_times, lengths

    def _relation_potentials(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Construct Eqs. (12)--(16) without materializing T^2 relation matrices."""
        batch_size, sequence_length = item_seq.shape
        candidate_count = candidates.size(1)
        device = item_seq.device

        positions = torch.arange(sequence_length, device=device)
        valid = (
            positions.unsqueeze(0) < target_positions.unsqueeze(1)
        ) & item_seq.ne(0) & type_seq.ne(0)

        safe_types = type_seq.clamp(min=0, max=self.num_behaviors - 1)
        item_states = F.normalize(self.item_embedding(item_seq), dim=-1)
        candidate_states = F.normalize(
            self.item_embedding(candidates), dim=-1
        )

        node = item_states.new_zeros(
            batch_size, candidate_count, sequence_length
        )
        for source_behavior in range(1, self.num_behaviors):
            source_mask = safe_types.eq(source_behavior)
            if not source_mask.any():
                continue
            transformed = F.linear(
                item_states, self.relation[source_behavior, self.buy_type]
            )
            source_scores = torch.einsum(
                'bcd,btd->bct', candidate_states, transformed
            )
            node = node + source_scores * source_mask.unsqueeze(1)

        edge_semantic = item_states.new_zeros(
            batch_size, sequence_length, sequence_length
        )
        lambda_edge = item_states.new_zeros(
            batch_size, sequence_length, sequence_length
        )
        for destination_behavior in range(1, self.num_behaviors):
            destination_mask = safe_types.eq(destination_behavior)
            if not destination_mask.any():
                continue
            operators = self.relation[safe_types, destination_behavior]
            transformed_sources = torch.einsum(
                'btij,btj->bti', operators, item_states
            )
            pair_scores = torch.einsum(
                'bpd,btd->bpt', transformed_sources, item_states
            )
            edge_semantic = edge_semantic + (
                pair_scores * destination_mask.unsqueeze(1)
            )
            source_lambdas = F.softplus(
                self.temporal_sensitivity_raw[
                    safe_types, destination_behavior
                ]
            )
            lambda_edge = lambda_edge + (
                source_lambdas.unsqueeze(-1)
                * destination_mask.unsqueeze(1)
            )

        times = self._time_values(item_seq, time_seq)
        elapsed = (
            times.unsqueeze(1) - times.unsqueeze(2)
        ).clamp_min(0.0)
        log_retention = -lambda_edge * torch.log1p(elapsed)
        edge = edge_semantic + log_retention

        chronological = positions.view(1, -1, 1) < positions.view(1, 1, -1)
        valid_edge = (
            chronological
            & valid.unsqueeze(-1)
            & valid.unsqueeze(1)
        )

        last_positions = (target_positions - 1).clamp_min(0)
        cutoff = times.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        age = (cutoff.unsqueeze(1) - times).clamp_min(0.0)
        terminal_lambda = F.softplus(
            self.temporal_sensitivity_raw[safe_types, self.buy_type]
        )
        terminal_correction = -terminal_lambda * torch.log1p(age)

        node = node.masked_fill(~valid.unsqueeze(1), -1e9)
        edge = edge.masked_fill(~valid_edge, -1e9)
        terminal_correction = terminal_correction.masked_fill(~valid, -1e9)
        return node, edge, terminal_correction, valid, valid_edge, log_retention

    def _exact_posterior(
        self, node, edge, terminal_correction, valid, valid_edge,
        temperature=None,
    ):
        """Exact prefix/suffix inference for all non-empty increasing paths."""
        _, _, sequence_length = node.shape
        if temperature is None:
            temperature = self._temperature()

        prefix_states = []
        for target in range(sequence_length):
            start = node.new_zeros(node.size(0), node.size(1), 1)
            if target == 0:
                alternatives = start
            else:
                extensions = (
                    torch.stack(prefix_states, dim=-1)
                    + edge[:, :target, target].unsqueeze(1) / temperature
                )
                extensions = extensions.masked_fill(
                    ~valid_edge[:, :target, target].unsqueeze(1), -1e9
                )
                alternatives = torch.cat((start, extensions), dim=-1)
            state = node[:, :, target] / temperature + torch.logsumexp(
                alternatives, dim=-1
            )
            state = state.masked_fill(~valid[:, target].unsqueeze(1), -1e9)
            prefix_states.append(state)
        prefix = torch.stack(prefix_states, dim=-1)

        suffix_states = [None] * sequence_length
        for source in range(sequence_length - 1, -1, -1):
            terminate = (
                terminal_correction[:, source].view(-1, 1, 1)
                / temperature
            ).expand(-1, node.size(1), -1)
            if source == sequence_length - 1:
                alternatives = terminate
            else:
                later_suffix = torch.stack(
                    suffix_states[source + 1:], dim=-1
                )
                continuations = (
                    edge[:, source, source + 1:].unsqueeze(1) / temperature
                    + node[:, :, source + 1:] / temperature
                    + later_suffix
                )
                continuations = continuations.masked_fill(
                    ~valid_edge[:, source, source + 1:].unsqueeze(1), -1e9
                )
                alternatives = torch.cat((terminate, continuations), dim=-1)
            state = torch.logsumexp(alternatives, dim=-1)
            state = state.masked_fill(~valid[:, source].unsqueeze(1), -1e9)
            suffix_states[source] = state
        suffix = torch.stack(suffix_states, dim=-1)

        terminal_scaled = terminal_correction.unsqueeze(1) / temperature
        log_partition = torch.logsumexp(prefix + terminal_scaled, dim=-1)

        node_marginal = torch.exp(
            prefix + suffix - log_partition.unsqueeze(-1)
        )
        node_marginal = node_marginal * valid.unsqueeze(1)

        log_edge_marginal = (
            prefix.unsqueeze(-1)
            + edge.unsqueeze(1) / temperature
            + node.unsqueeze(-2) / temperature
            + suffix.unsqueeze(-2)
            - log_partition.unsqueeze(-1).unsqueeze(-1)
        )
        edge_marginal = torch.exp(
            log_edge_marginal.masked_fill(
                ~valid_edge.unsqueeze(1), -1e9
            )
        ) * valid_edge.unsqueeze(1)

        terminal_marginal = torch.exp(
            prefix + terminal_scaled - log_partition.unsqueeze(-1)
        ) * valid.unsqueeze(1)
        return (
            temperature,
            log_partition,
            prefix,
            suffix,
            node_marginal,
            edge_marginal,
            terminal_marginal,
        )

    def _posterior_gated_representation(
        self, item_seq, type_seq, valid, edge_marginal,
        node_marginal, terminal_marginal, log_retention
    ):
        """Implement Eqs. (31)--(33) in chronological topological order."""
        batch_size, sequence_length = item_seq.shape
        candidate_count = node_marginal.size(1)
        safe_types = type_seq.clamp(min=0, max=self.num_behaviors - 1)
        local_states = (
            self.item_embedding(item_seq)
            + self.behavior_embedding(safe_types)
        )
        retention = torch.exp(log_retention)
        alpha = edge_marginal / node_marginal.unsqueeze(-2).clamp_min(1e-12)

        propagated = []
        transformed_for_destination = []
        for target in range(sequence_length):
            current = local_states[:, target].unsqueeze(1).expand(
                -1, candidate_count, -1
            )
            if target > 0:
                transformed_history = torch.stack(
                    transformed_for_destination, dim=2
                )
                destination_index = safe_types[:, target].view(
                    batch_size, 1, 1, 1, 1
                ).expand(
                    -1,
                    candidate_count,
                    target,
                    1,
                    self.hidden_size,
                )
                messages = torch.gather(
                    transformed_history, 3, destination_index
                ).squeeze(3)
                source_types = safe_types[:, :target]
                destination_types = safe_types[:, target].unsqueeze(1)
                residual_gate = torch.sigmoid(
                    self.message_gate_raw[source_types, destination_types]
                )
                weights = (
                    alpha[:, :, :target, target]
                    * retention[:, :target, target].unsqueeze(1)
                    * residual_gate.unsqueeze(1)
                )
                current = current + (
                    weights.unsqueeze(-1) * messages
                ).sum(dim=2)
            current = current * valid[:, target].view(batch_size, 1, 1)
            propagated.append(current)

            source_types = safe_types[:, target]
            all_destinations = []
            for destination_behavior in range(self.num_behaviors):
                operators = self.relation[source_types, destination_behavior]
                all_destinations.append(
                    torch.einsum('bij,bcj->bci', operators, current)
                )
            transformed_for_destination.append(
                torch.stack(all_destinations, dim=2)
            )

        propagated = torch.stack(propagated, dim=2)
        representation = (
            terminal_marginal.unsqueeze(-1) * propagated
        ).sum(dim=2)
        return representation, alpha, propagated

    def _forward_only_representation(
        self,
        item_seq,
        type_seq,
        node,
        edge,
        terminal_correction,
        valid,
        valid_edge,
        log_retention,
        temperature=None,
    ):
        """Exact prediction using prefix states only.

        The suffix state cancels from alpha[p,t] = nu[p,t] / mu[t].
        Terminal probabilities are a softmax of the corrected prefix states.
        Consequently, ordinary prediction does not need candidate-indexed
        T-by-T edge marginals.  The dense prefix-suffix path remains available
        through ``return_diagnostics=True`` for audits and visualizations.
        """
        batch_size, candidate_count, sequence_length = node.shape
        if temperature is None:
            temperature = self._temperature()
        safe_types = type_seq.clamp(min=0, max=self.num_behaviors - 1)
        local_states = (
            self.item_embedding(item_seq)
            + self.behavior_embedding(safe_types)
        )
        retention = torch.exp(log_retention)

        prefix_states = []
        propagated_states = []
        transformed_for_destination = []
        for target in range(sequence_length):
            node_scaled = node[:, :, target] / temperature
            if target == 0:
                state = node_scaled
                predecessor_probability = None
            else:
                previous_prefix = torch.stack(prefix_states, dim=-1)
                extensions = (
                    previous_prefix
                    + edge[:, :target, target].unsqueeze(1) / temperature
                )
                edge_is_valid = valid_edge[:, :target, target].unsqueeze(1)
                extensions = extensions.masked_fill(~edge_is_valid, -1e9)
                start = node.new_zeros(batch_size, candidate_count, 1)
                state = node_scaled + torch.logsumexp(
                    torch.cat((start, extensions), dim=-1), dim=-1
                )
                # Identity obtained by cancelling suffix and partition terms
                # from nu[p,t] / mu[t].
                predecessor_probability = torch.exp(
                    extensions + node_scaled.unsqueeze(-1)
                    - state.unsqueeze(-1)
                ) * edge_is_valid

            target_is_valid = valid[:, target].unsqueeze(1)
            state = state.masked_fill(~target_is_valid, -1e9)
            prefix_states.append(state)

            current = local_states[:, target].unsqueeze(1).expand(
                -1, candidate_count, -1
            )
            if target > 0:
                transformed_history = torch.stack(
                    transformed_for_destination, dim=2
                )
                destination_index = safe_types[:, target].view(
                    batch_size, 1, 1, 1, 1
                ).expand(
                    -1,
                    candidate_count,
                    target,
                    1,
                    self.hidden_size,
                )
                messages = torch.gather(
                    transformed_history, 3, destination_index
                ).squeeze(3)
                source_types = safe_types[:, :target]
                destination_types = safe_types[:, target].unsqueeze(1)
                residual_gate = torch.sigmoid(
                    self.message_gate_raw[source_types, destination_types]
                )
                weights = (
                    predecessor_probability
                    * retention[:, :target, target].unsqueeze(1)
                    * residual_gate.unsqueeze(1)
                )
                current = current + (
                    weights.unsqueeze(-1) * messages
                ).sum(dim=2)
            current = current * target_is_valid.unsqueeze(-1)
            propagated_states.append(current)

            source_types = safe_types[:, target]
            all_destinations = []
            for destination_behavior in range(self.num_behaviors):
                operators = self.relation[source_types, destination_behavior]
                all_destinations.append(
                    torch.einsum('bij,bcj->bci', operators, current)
                )
            transformed_for_destination.append(
                torch.stack(all_destinations, dim=2)
            )

        prefix = torch.stack(prefix_states, dim=-1)
        propagated = torch.stack(propagated_states, dim=2)
        corrected_terminal = (
            prefix + terminal_correction.unsqueeze(1) / temperature
        ).masked_fill(~valid.unsqueeze(1), -1e9)
        terminal_marginal = torch.softmax(corrected_terminal, dim=-1)
        representation = (
            terminal_marginal.unsqueeze(-1) * propagated
        ).sum(dim=2)
        return representation, prefix, terminal_marginal, propagated

    def _separable_common_potentials(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Candidate nodes and terminal terms shared by both linear audits.

        The separable model is deliberately a unit-temperature CRF: raising a
        sum of positive channels to a learned inverse temperature would destroy
        separability.  Scale is learned inside the channel projections and the
        final calibrated cosine score instead.
        """
        batch_size, sequence_length = item_seq.shape
        candidate_count = candidates.size(1)
        positions = torch.arange(sequence_length, device=item_seq.device)
        valid = (
            positions.unsqueeze(0) < target_positions.unsqueeze(1)
        ) & item_seq.ne(0) & type_seq.ne(0)
        safe_types = type_seq.clamp(min=0, max=self.num_behaviors - 1)
        local_states = (
            self.item_embedding(item_seq)
            + self.behavior_embedding(safe_types)
        )
        contextual_states, _ = self.sequence_encoder(local_states)
        item_states = F.normalize(contextual_states, dim=-1)
        target_behavior = torch.full_like(candidates, self.buy_type)
        candidate_states = F.normalize(
            self.item_embedding(candidates)
            + self.behavior_embedding(target_behavior),
            dim=-1,
        )

        node = item_states.new_zeros(
            batch_size, candidate_count, sequence_length
        )
        for source_behavior in range(1, self.num_behaviors):
            source_mask = safe_types.eq(source_behavior)
            if not source_mask.any():
                continue
            transformed = F.linear(
                item_states, self.relation[source_behavior, self.buy_type]
            )
            source_scores = torch.einsum(
                'bcd,btd->bct', candidate_states, transformed
            )
            node = node + source_scores * source_mask.unsqueeze(1)

        times = self._time_values(item_seq, time_seq)
        last_positions = (target_positions - 1).clamp_min(0)
        cutoff = times.gather(1, last_positions.unsqueeze(1)).squeeze(1)
        age = (cutoff.unsqueeze(1) - times).clamp_min(0.0)
        terminal_lambda = F.softplus(
            self.temporal_sensitivity_raw[safe_types, self.buy_type]
        )
        terminal_correction = -terminal_lambda * torch.log1p(age)
        node = node.masked_fill(~valid.unsqueeze(1), -1e9)
        terminal_correction = terminal_correction.masked_fill(~valid, -1e9)
        return (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        )

    def _separable_relation_potentials(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Materialize the separable kernel only for small exact audits."""
        (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        ) = self._separable_common_potentials(
            item_seq, type_seq, candidates, target_positions, time_seq
        )
        batch_size, sequence_length = item_seq.shape
        positions = torch.arange(sequence_length, device=item_seq.device)
        elapsed = (
            times.unsqueeze(1) - times.unsqueeze(2)
        ).clamp_min(0.0)
        edge = item_states.new_zeros(
            batch_size, sequence_length, sequence_length
        )
        decay = F.softplus(self.kernel_decay_raw)
        log_channels = math.log(float(self.combo_kernel_channels))
        for source_behavior in range(1, self.num_behaviors):
            source_mask = safe_types.eq(source_behavior)
            if not source_mask.any():
                continue
            for destination_behavior in range(1, self.num_behaviors):
                destination_mask = safe_types.eq(destination_behavior)
                if not destination_mask.any():
                    continue
                source_score = torch.einsum(
                    'btd,hd->bth',
                    item_states,
                    self.kernel_source[
                        source_behavior, destination_behavior
                    ],
                )
                destination_score = torch.einsum(
                    'btd,hd->bth',
                    item_states,
                    self.kernel_destination[
                        source_behavior, destination_behavior
                    ],
                )
                channel_score = (
                    source_score.unsqueeze(2)
                    + destination_score.unsqueeze(1)
                    - decay[source_behavior, destination_behavior].view(
                        1, 1, 1, -1
                    ) * elapsed.unsqueeze(-1)
                )
                pair_score = torch.logsumexp(
                    channel_score, dim=-1
                ) - log_channels
                pair_mask = (
                    source_mask.unsqueeze(2)
                    & destination_mask.unsqueeze(1)
                )
                edge = edge + pair_score * pair_mask

        chronological = positions.view(1, -1, 1) < positions.view(1, 1, -1)
        valid_edge = (
            chronological & valid.unsqueeze(-1) & valid.unsqueeze(1)
        )
        edge = edge.masked_fill(~valid_edge, -1e9)
        log_retention = torch.zeros_like(edge)
        return (
            node,
            edge,
            terminal_correction,
            valid,
            valid_edge,
            log_retention,
            contextual_states,
        )

    def _closed_form_relation_potentials(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Explicit quadratic edge matrix for auditing the closed-form scan."""
        (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        ) = self._separable_common_potentials(
            item_seq, type_seq, candidates, target_positions, time_seq
        )
        sequence_length = item_seq.size(1)
        positions = torch.arange(sequence_length, device=item_seq.device)
        source_score = torch.einsum(
            'btd,btd->bt', item_states, self.scan_source[safe_types]
        )
        destination_score = torch.einsum(
            'btd,btd->bt',
            item_states,
            self.scan_destination[safe_types],
        )
        elapsed = (
            times.unsqueeze(1) - times.unsqueeze(2)
        ).clamp_min(0.0)
        decay = F.softplus(self.scan_decay_raw)
        edge = (
            source_score.unsqueeze(2)
            + destination_score.unsqueeze(1)
            - decay * elapsed
        )
        chronological = positions.view(1, -1, 1) < positions.view(1, 1, -1)
        valid_edge = (
            chronological & valid.unsqueeze(-1) & valid.unsqueeze(1)
        )
        edge = edge.masked_fill(~valid_edge, -1e9)
        return (
            node,
            edge,
            terminal_correction,
            valid,
            valid_edge,
            torch.zeros_like(edge),
            contextual_states,
        )

    def _closed_form_partition_representation(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Vectorized exact global inference for the rank-one positive kernel.

        Let ``A_t`` be total emitted path mass after event ``t``.  With
        ``k(p,t)=u_p v_t``, the recurrence is

            A_t = (1 + u_t v_t exp(a_t)) A_{t-1}
                  + u_t exp(a_t).

        Products of its positive affine coefficients are evaluated stably in
        log space with cumulative sums and ``logcumsumexp``.  This removes the
        Python time loop while retaining every non-empty increasing path.
        """
        (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        ) = self._separable_common_potentials(
            item_seq, type_seq, candidates, target_positions, time_seq
        )
        source_score = torch.einsum(
            'btd,btd->bt', item_states, self.scan_source[safe_types]
        )
        destination_score = torch.einsum(
            'btd,btd->bt',
            item_states,
            self.scan_destination[safe_types],
        )
        decay = F.softplus(self.scan_decay_raw)
        log_u = source_score + decay * times
        log_v = destination_score - decay * times

        # log_a and log_b are the positive affine recurrence coefficients.
        log_a = F.softplus(
            node + source_score.unsqueeze(1) + destination_score.unsqueeze(1)
        )
        log_b = node + log_u.unsqueeze(1)
        log_a = log_a.masked_fill(~valid.unsqueeze(1), 0.0)
        log_b = log_b.masked_fill(~valid.unsqueeze(1), -1e9)
        log_product = torch.cumsum(log_a, dim=-1)
        log_emitted_mass = log_product + torch.logcumsumexp(
            log_b - log_product, dim=-1
        )
        initial_log_zero = node.new_full(
            (node.size(0), node.size(1), 1), -1e9
        )
        previous_mass = torch.cat(
            (initial_log_zero, log_emitted_mass[:, :, :-1]), dim=-1
        )
        prefix = node + torch.logaddexp(
            torch.zeros_like(node),
            log_v.unsqueeze(1) + previous_mass,
        )
        prefix = prefix.masked_fill(~valid.unsqueeze(1), -1e9)
        corrected_terminal = (
            prefix + terminal_correction.unsqueeze(1)
        ).masked_fill(~valid.unsqueeze(1), -1e9)
        terminal_marginal = torch.softmax(corrected_terminal, dim=-1)
        representation = (
            terminal_marginal.unsqueeze(-1)
            * contextual_states.unsqueeze(1)
        ).sum(dim=2)
        return (
            representation,
            prefix,
            terminal_marginal,
            node,
            terminal_correction,
            valid,
        )

    def _linear_partition_representation(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Exact linear-time partition recurrence with contextual readout.

        The causal GRU produces event representations once.  Candidate node
        energies and the separable transition kernel then sum every non-empty
        increasing path exactly.  The normalized mass of paths ending at each
        event is used to read out the same contextual states, coupling the
        representation learner and combinatorial objective end to end without
        a candidate-by-channel vector message accumulator.
        """
        (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        ) = self._separable_common_potentials(
            item_seq, type_seq, candidates, target_positions, time_seq
        )
        batch_size, candidate_count, sequence_length = node.shape
        behavior_count = self.num_behaviors
        channel_count = self.combo_kernel_channels
        decay = F.softplus(self.kernel_decay_raw)
        log_channels = math.log(float(channel_count))
        log_mass = node.new_full(
            (
                batch_size,
                candidate_count,
                behavior_count,
                behavior_count,
                channel_count,
            ),
            -1e9,
        )
        batch_index = torch.arange(batch_size, device=item_seq.device)
        prefix_states = []
        for target in range(sequence_length):
            current_type = safe_types[:, target]
            current_time = times[:, target]
            target_is_valid = valid[:, target]
            current_item = item_states[:, target]

            destination_all = torch.einsum(
                'bk,ijhk->bijh', current_item, self.kernel_destination
            )
            destination_score = destination_all.gather(
                2,
                current_type.view(batch_size, 1, 1, 1).expand(
                    -1, behavior_count, 1, channel_count
                ),
            ).squeeze(2)
            destination_decay = decay[:, current_type, :].permute(1, 0, 2)
            selected_log_mass = log_mass.gather(
                3,
                current_type.view(batch_size, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    behavior_count,
                    1,
                    channel_count,
                ),
            ).squeeze(3)
            component_log_mass = (
                selected_log_mass
                + destination_score.unsqueeze(1)
                - destination_decay.unsqueeze(1)
                * current_time.view(batch_size, 1, 1, 1)
                - log_channels
            )
            predecessor_log_mass = torch.logsumexp(
                component_log_mass.reshape(
                    batch_size, candidate_count, -1
                ),
                dim=-1,
            )
            state = node[:, :, target] + torch.logaddexp(
                torch.zeros_like(predecessor_log_mass),
                predecessor_log_mass,
            )
            state = state.masked_fill(~target_is_valid.unsqueeze(1), -1e9)
            prefix_states.append(state)

            source_all = torch.einsum(
                'bk,ijhk->bijh', current_item, self.kernel_source
            )
            source_score = source_all[batch_index, current_type]
            source_decay = decay[current_type]
            new_log_weight = (
                state.unsqueeze(-1).unsqueeze(-1)
                + source_score.unsqueeze(1)
                + source_decay.unsqueeze(1)
                * current_time.view(batch_size, 1, 1, 1)
            )
            selected_old_log = log_mass.gather(
                2,
                current_type.view(batch_size, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    1,
                    behavior_count,
                    channel_count,
                ),
            ).squeeze(2)
            combined_log = torch.logaddexp(
                selected_old_log, new_log_weight
            )
            source_mask = F.one_hot(
                current_type, num_classes=behavior_count
            ).bool().view(batch_size, 1, behavior_count, 1, 1)
            update_mask = source_mask & target_is_valid.view(
                batch_size, 1, 1, 1, 1
            )
            log_mass = torch.where(
                update_mask, combined_log.unsqueeze(2), log_mass
            )

        prefix = torch.stack(prefix_states, dim=-1)
        corrected_terminal = (
            prefix + terminal_correction.unsqueeze(1)
        ).masked_fill(~valid.unsqueeze(1), -1e9)
        terminal_marginal = torch.softmax(corrected_terminal, dim=-1)
        representation = (
            terminal_marginal.unsqueeze(-1)
            * contextual_states.unsqueeze(1)
        ).sum(dim=2)
        return (
            representation,
            prefix,
            terminal_marginal,
            node,
            terminal_correction,
            valid,
        )

    def _linear_separable_representation(
        self, item_seq, type_seq, candidates, target_positions, time_seq=None
    ):
        """Exact O(T) global path inference for the separable edge family.

        ``log_mass[b,c,s,d,h]`` stores the log total prefix mass emitted by
        source behavior ``s`` toward destination behavior ``d`` in channel
        ``h``. ``mean_message`` stores the corresponding normalized sufficient
        statistic of the propagated embeddings.  Both are updated once per
        event, so the whole non-empty subsequence space is summed without
        enumerating any of its O(T^2) possible edges.
        """
        (
            node,
            terminal_correction,
            valid,
            safe_types,
            item_states,
            times,
            contextual_states,
        ) = self._separable_common_potentials(
            item_seq, type_seq, candidates, target_positions, time_seq
        )
        batch_size, candidate_count, sequence_length = node.shape
        behavior_count = self.num_behaviors
        channel_count = self.combo_kernel_channels
        local_states = (
            self.item_embedding(item_seq)
            + self.behavior_embedding(safe_types)
        )
        decay = F.softplus(self.kernel_decay_raw)
        log_channels = math.log(float(channel_count))
        # A finite log-zero avoids undefined (-inf)-(-inf) derivatives in
        # autograd while underflowing cleanly to zero in float32.
        negative_infinity = -1e9
        log_mass = node.new_full(
            (
                batch_size,
                candidate_count,
                behavior_count,
                behavior_count,
                channel_count,
            ),
            negative_infinity,
        )
        mean_message = node.new_zeros(
            (
                batch_size,
                candidate_count,
                behavior_count,
                behavior_count,
                channel_count,
                self.hidden_size,
            )
        )
        batch_index = torch.arange(batch_size, device=item_seq.device)
        prefix_states = []
        propagated_states = []

        for target in range(sequence_length):
            current_type = safe_types[:, target]
            current_time = times[:, target]
            target_is_valid = valid[:, target]
            current_item = item_states[:, target]

            destination_all = torch.einsum(
                'bk,ijhk->bijh', current_item, self.kernel_destination
            )
            destination_score = destination_all.gather(
                2,
                current_type.view(batch_size, 1, 1, 1).expand(
                    -1, behavior_count, 1, channel_count
                ),
            ).squeeze(2)
            destination_decay = decay[:, current_type, :].permute(1, 0, 2)
            selected_log_mass = log_mass.gather(
                3,
                current_type.view(batch_size, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    behavior_count,
                    1,
                    channel_count,
                ),
            ).squeeze(3)
            selected_mean = mean_message.gather(
                3,
                current_type.view(batch_size, 1, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    behavior_count,
                    1,
                    channel_count,
                    self.hidden_size,
                ),
            ).squeeze(3)
            component_log_mass = (
                selected_log_mass
                + destination_score.unsqueeze(1)
                - destination_decay.unsqueeze(1)
                * current_time.view(batch_size, 1, 1, 1)
                - log_channels
            )
            predecessor_log_mass = torch.logsumexp(
                component_log_mass.reshape(
                    batch_size, candidate_count, -1
                ),
                dim=-1,
            )
            denominator = torch.logaddexp(
                torch.zeros_like(predecessor_log_mass),
                predecessor_log_mass,
            )
            state = node[:, :, target] + denominator
            state = state.masked_fill(~target_is_valid.unsqueeze(1), -1e9)
            prefix_states.append(state)

            predecessor_weight = torch.exp(
                component_log_mass - denominator.unsqueeze(-1).unsqueeze(-1)
            )
            message = (
                predecessor_weight.unsqueeze(-1) * selected_mean
            ).sum(dim=(2, 3))
            current = local_states[:, target].unsqueeze(1) + message
            current = current * target_is_valid.view(batch_size, 1, 1)
            propagated_states.append(current)

            source_all = torch.einsum(
                'bk,ijhk->bijh', current_item, self.kernel_source
            )
            source_score = source_all[batch_index, current_type]
            source_decay = decay[current_type]
            new_log_weight = (
                state.unsqueeze(-1).unsqueeze(-1)
                + source_score.unsqueeze(1)
                + source_decay.unsqueeze(1)
                * current_time.view(batch_size, 1, 1, 1)
            )

            operators = self.relation[current_type]
            new_message = torch.einsum(
                'bdij,bcj->bcdi', operators, current
            )
            residual_gate = torch.sigmoid(
                self.message_gate_raw[current_type]
            )
            new_message = new_message * residual_gate.unsqueeze(1).unsqueeze(-1)
            new_message = new_message.unsqueeze(3).expand(
                -1, -1, -1, channel_count, -1
            )

            selected_old_log = log_mass.gather(
                2,
                current_type.view(batch_size, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    1,
                    behavior_count,
                    channel_count,
                ),
            ).squeeze(2)
            selected_old_mean = mean_message.gather(
                2,
                current_type.view(batch_size, 1, 1, 1, 1, 1).expand(
                    -1,
                    candidate_count,
                    1,
                    behavior_count,
                    channel_count,
                    self.hidden_size,
                ),
            ).squeeze(2)
            combined_log = torch.logaddexp(
                selected_old_log, new_log_weight
            )
            old_coefficient = torch.where(
                torch.isfinite(selected_old_log),
                torch.exp(selected_old_log - combined_log),
                torch.zeros_like(combined_log),
            )
            new_coefficient = torch.exp(new_log_weight - combined_log)
            combined_mean = (
                old_coefficient.unsqueeze(-1) * selected_old_mean
                + new_coefficient.unsqueeze(-1) * new_message
            )
            source_mask = F.one_hot(
                current_type, num_classes=behavior_count
            ).bool().view(batch_size, 1, behavior_count, 1, 1)
            update_mask = source_mask & target_is_valid.view(
                batch_size, 1, 1, 1, 1
            )
            log_mass = torch.where(
                update_mask, combined_log.unsqueeze(2), log_mass
            )
            mean_message = torch.where(
                update_mask.unsqueeze(-1),
                combined_mean.unsqueeze(2),
                mean_message,
            )

        prefix = torch.stack(prefix_states, dim=-1)
        propagated = torch.stack(propagated_states, dim=2)
        corrected_terminal = (
            prefix + terminal_correction.unsqueeze(1)
        ).masked_fill(~valid.unsqueeze(1), -1e9)
        terminal_marginal = torch.softmax(corrected_terminal, dim=-1)
        representation = (
            terminal_marginal.unsqueeze(-1) * propagated
        ).sum(dim=2)
        return (
            representation,
            prefix,
            terminal_marginal,
            propagated,
            node,
            terminal_correction,
            valid,
        )

    def candidate_logits(
        self,
        item_seq,
        type_seq,
        candidates,
        target_positions=None,
        time_seq=None,
        return_diagnostics=False,
    ):
        """Score candidates through the exact candidate-specific computation."""
        if target_positions is None:
            target_positions = torch.count_nonzero(item_seq, dim=1)
        linear_kernel = getattr(self, 'combo_linear_kernel', False)
        closed_form_kernel = (
            linear_kernel
            and getattr(self, 'combo_closed_form_kernel', False)
        )
        if linear_kernel and not return_diagnostics:
            (
                representation,
                _,
                _,
                _,
                _,
                _,
            ) = (
                self._closed_form_partition_representation(
                    item_seq,
                    type_seq,
                    candidates,
                    target_positions,
                    time_seq=time_seq,
                )
                if closed_form_kernel
                else self._linear_partition_representation(
                    item_seq,
                    type_seq,
                    candidates,
                    target_positions,
                    time_seq=time_seq,
                )
            )
        else:
            potential_builder = (
                self._closed_form_relation_potentials
                if closed_form_kernel
                else self._separable_relation_potentials
                if linear_kernel else self._relation_potentials
            )
            potentials = potential_builder(
                item_seq,
                type_seq,
                candidates,
                target_positions,
                time_seq=time_seq,
            )
            if linear_kernel:
                (
                    node,
                    edge,
                    terminal_correction,
                    valid,
                    valid_edge,
                    log_retention,
                    contextual_states,
                ) = potentials
            else:
                (
                    node,
                    edge,
                    terminal_correction,
                    valid,
                    valid_edge,
                    log_retention,
                ) = potentials
        if return_diagnostics:
            diagnostic_temperature = (
                node.new_tensor(1.0) if linear_kernel else None
            )
            (
                temperature,
                log_partition,
                prefix,
                suffix,
                node_marginal,
                edge_marginal,
                terminal_marginal,
            ) = self._exact_posterior(
                node,
                edge,
                terminal_correction,
                valid,
                valid_edge,
                temperature=diagnostic_temperature,
            )
            if linear_kernel:
                alpha = edge_marginal / node_marginal.unsqueeze(
                    -2
                ).clamp_min(1e-12)
                propagated = contextual_states.unsqueeze(1).expand(
                    -1, candidates.size(1), -1, -1
                )
                representation = (
                    terminal_marginal.unsqueeze(-1) * propagated
                ).sum(dim=2)
            else:
                representation, alpha, propagated = (
                    self._posterior_gated_representation(
                    item_seq,
                    type_seq,
                    valid,
                    edge_marginal,
                    node_marginal,
                    terminal_marginal,
                    log_retention,
                )
                )
        elif not linear_kernel:
            representation, _, _, _ = self._forward_only_representation(
                item_seq,
                type_seq,
                node,
                edge,
                terminal_correction,
                valid,
                valid_edge,
                log_retention,
            )
        target_behavior = torch.full_like(candidates, self.buy_type)
        target_representation = (
            self.item_embedding(candidates)
            + self.behavior_embedding(target_behavior)
        )
        cosine = F.cosine_similarity(
            representation, target_representation, dim=-1
        )
        logits = self._logit_scale() * cosine

        if not return_diagnostics:
            return logits
        diagnostics = {
            'node_utility': node,
            'edge_utility': edge,
            'terminal_correction': terminal_correction,
            'temperature': temperature,
            'log_partition': log_partition,
            'soft_optimum': temperature * log_partition,
            'prefix': prefix,
            'suffix': suffix,
            'node_marginal': node_marginal,
            'edge_marginal': edge_marginal,
            'terminal_marginal': terminal_marginal,
            'predecessor_probability': alpha,
            'propagated_states': propagated,
            'history_representation': representation,
            'cosine': cosine,
            'message_gate': torch.sigmoid(self.message_gate_raw),
        }
        return logits, diagnostics

    def _chunked_candidate_logits(
        self,
        item_seq,
        type_seq,
        candidates,
        target_positions,
        time_seq=None,
        checkpoint_grad=False,
    ):
        """Score arbitrary candidate sets without changing their semantics.

        Candidate posteriors are conditionally independent given the complete
        prefix. Splitting only the candidate dimension therefore changes peak
        memory, not the exact trajectory space or any candidate score.
        """
        blocks = []
        for candidate_block in candidates.split(
            self.combo_candidate_chunk_size, dim=1
        ):
            if checkpoint_grad and self.training and torch.is_grad_enabled():
                anchor = torch.zeros(
                    (),
                    device=item_seq.device,
                    dtype=self.item_embedding.weight.dtype,
                    requires_grad=True,
                )

                def score_block(block, grad_anchor):
                    return self.candidate_logits(
                        item_seq,
                        type_seq,
                        block,
                        target_positions=target_positions,
                        time_seq=time_seq,
                    ) + grad_anchor * 0.0

                block_logits = checkpoint(
                    score_block, candidate_block, anchor
                )
            else:
                block_logits = self.candidate_logits(
                    item_seq,
                    type_seq,
                    candidate_block,
                    target_positions=target_positions,
                    time_seq=time_seq,
                )
            blocks.append(block_logits)
        return torch.cat(blocks, dim=1)

    @staticmethod
    def _quota(total, weight):
        return int(round(total * weight))

    def _sample_candidates(self, positives, item_seq, type_seq, count):
        """Mixture sampler aligned with candidate-conditioned structure learning.

        The hard component samples prefix items that can form deceptively
        strong historical structures.  This includes both
        auxiliary-behavior items and previous target-behavior items, so a repeat
        target cannot win merely because every other seen item was removed.
        Popular and uniform components retain exposure and catalog coverage.
        Only padding and the current positive item are forbidden.
        """
        device = positives.device
        batch_size, sequence_length = item_seq.shape
        hard_quota = self._quota(count, self.combo_hard_weight)
        pop_quota = self._quota(count, self.combo_pop_weight)
        uniform_quota = max(count - hard_quota - pop_quota, 0)
        candidate_parts = []
        candidate_sources = []

        if hard_quota:
            historical = item_seq.ne(0) & item_seq.ne(
                positives.unsqueeze(1)
            )
            # Keep only the most recent occurrence of each item, so a hard
            # quota is reallocated only when there are genuinely too few
            # distinct legal historical candidates.
            positions = torch.arange(sequence_length, device=device)
            same_item = item_seq.unsqueeze(2).eq(item_seq.unsqueeze(1))
            later_position = positions.view(1, -1, 1) < positions.view(
                1, 1, -1
            )
            has_later_duplicate = (
                same_item
                & later_position
                & historical.unsqueeze(1)
            ).any(dim=2)
            historical = historical & ~has_later_duplicate
            position_weight = 1.0 + torch.arange(
                sequence_length, device=device, dtype=torch.float32
            ) / max(sequence_length - 1, 1)
            behavior_weight = torch.where(
                type_seq.eq(self.buy_type), 1.5, 1.0
            )
            random_uniform = torch.rand(
                batch_size, sequence_length, device=device
            ).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(random_uniform))
            hard_scores = (
                torch.log(position_weight).unsqueeze(0)
                + torch.log(behavior_weight)
                + gumbel
            ).masked_fill(~historical, -1e9)
            available_hard = min(hard_quota, sequence_length)
            hard_positions = torch.topk(
                hard_scores, available_hard, dim=1, sorted=False
            ).indices
            hard_items = item_seq.gather(1, hard_positions)
            hard_valid = historical.gather(1, hard_positions)
            hard_items = hard_items.masked_fill(~hard_valid, 0)
            if available_hard < hard_quota:
                hard_items = F.pad(
                    hard_items, (0, hard_quota - available_hard)
                )
            candidate_parts.append(hard_items)
            candidate_sources.append(torch.full_like(hard_items, 2))

        if pop_quota:
            popular = torch.multinomial(
                self.sampling_popularity,
                batch_size * pop_quota,
                replacement=True,
            ).view(batch_size, pop_quota)
            candidate_parts.append(popular)
            candidate_sources.append(torch.full_like(popular, 1))

        if uniform_quota:
            uniform = torch.randint(
                1,
                self.n_items,
                (batch_size, uniform_quota),
                device=device,
            )
            candidate_parts.append(uniform)
            candidate_sources.append(torch.zeros_like(uniform))
        candidates = torch.cat(candidate_parts, dim=1)
        sources = torch.cat(candidate_sources, dim=1)

        lower_triangle = torch.tril(
            torch.ones(count, count, dtype=torch.bool, device=device),
            diagonal=-1,
        )

        def invalid_mask(values):
            duplicate = (
                values.unsqueeze(-1).eq(values.unsqueeze(1))
                & lower_triangle.unsqueeze(0)
            ).any(dim=-1)
            return (
                values.eq(0)
                | values.eq(positives.unsqueeze(1))
                | duplicate
            )

        # Fixed-count vectorized rejection keeps the sampler on GPU.  Invalid
        # hard slots are explicitly relabelled and reallocated to the catalog
        # mixture if a prefix cannot provide enough distinct candidates.
        global_mass = self.combo_uniform_weight + self.combo_pop_weight
        popular_probability = (
            self.combo_pop_weight / global_mass if global_mass > 0 else 0.5
        )
        initial_invalid = invalid_mask(candidates)
        reallocated_hard = initial_invalid & sources.eq(2)
        for _ in range(8):
            invalid = invalid_mask(candidates)
            popular_draws = torch.multinomial(
                self.sampling_popularity,
                batch_size * count,
                replacement=True,
            ).view(batch_size, count)
            uniform_draws = torch.randint(
                1, self.n_items, (batch_size, count), device=device
            )
            choose_popular = torch.rand(
                batch_size, count, device=device
            ).lt(popular_probability)
            replacements = torch.where(
                choose_popular, popular_draws, uniform_draws
            )
            candidates = torch.where(invalid, replacements, candidates)
            replacement_sources = choose_popular.long()
            sources = torch.where(invalid, replacement_sources, sources)

        invalid = invalid_mask(candidates)
        row_ids = torch.arange(batch_size, device=device).unsqueeze(1)
        column_ids = torch.arange(count, device=device).unsqueeze(0)
        for step in range(32):
            fallback = 1 + (
                positives.unsqueeze(1) * 104729
                + row_ids * 13007
                + column_ids * 9176
                + step * 8191
            ) % (self.n_items - 1)
            candidates = torch.where(invalid, fallback, candidates)
            sources = torch.where(invalid, torch.zeros_like(sources), sources)
            invalid = invalid_mask(candidates)
        if invalid.any():
            raise RuntimeError('Unable to construct a legal negative set.')

        with torch.no_grad():
            self.sampling_source_counts.add_(
                torch.bincount(sources.reshape(-1), minlength=3)
            )
            self.sampling_requested_hard.add_(batch_size * hard_quota)
            self.sampling_reallocated_hard.add_(
                reallocated_hard.sum()
            )
            self.sampling_rows.add_(batch_size)
            self.sampling_rows_without_hard.add_(
                (~historical.any(dim=1)).sum()
            )
        return candidates

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction['item_type_list']
        positives = interaction[self.POS_ITEM_ID]
        time_seq = (
            interaction['item_time_list'].float()
            if 'item_time_list' in interaction else None
        )
        item_seq, type_seq, time_seq, target_positions = self._complete_prefix(
            item_seq, type_seq, time_seq
        )
        negatives = self._sample_candidates(
            positives, item_seq, type_seq, self.combo_negatives
        )
        candidates = torch.cat((positives.unsqueeze(1), negatives), dim=1)
        logits = self._chunked_candidate_logits(
            item_seq,
            type_seq,
            candidates,
            target_positions,
            time_seq=time_seq,
            checkpoint_grad=True,
        )
        positive_loss = F.softplus(-logits[:, 0])
        negative_loss = F.softplus(logits[:, 1:]).mean(dim=1)
        return (positive_loss + negative_loss).mean()

    def customized_sort_predict(self, interaction):
        raise NotImplementedError(
            'C3Comb receives candidates from the configured RecBole '
            'evaluation protocol; set customized_eval=0.'
        )

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction['item_type_list']
        candidates = interaction[self.ITEM_ID].view(-1, 1)
        time_seq = (
            interaction['item_time_list'].float()
            if 'item_time_list' in interaction else None
        )
        item_seq, type_seq, time_seq, target_positions = self._complete_prefix(
            item_seq, type_seq, time_seq
        )
        return self.candidate_logits(
            item_seq,
            type_seq,
            candidates,
            target_positions=target_positions,
            time_seq=time_seq,
        ).squeeze(1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction['item_type_list']
        time_seq = (
            interaction['item_time_list'].float()
            if 'item_time_list' in interaction else None
        )
        item_seq, type_seq, time_seq, target_positions = self._complete_prefix(
            item_seq, type_seq, time_seq
        )
        candidates = torch.arange(
            1, self.n_items, device=item_seq.device
        ).unsqueeze(0).expand(item_seq.size(0), -1)
        nonpadding_scores = self._chunked_candidate_logits(
            item_seq,
            type_seq,
            candidates,
            target_positions,
            time_seq=time_seq,
        )
        padding_scores = torch.full(
            (item_seq.size(0), 1),
            -float('inf'),
            device=item_seq.device,
            dtype=nonpadding_scores.dtype,
        )
        return torch.cat((padding_scores, nonpadding_scores), dim=1)
