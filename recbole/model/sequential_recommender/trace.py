# -*- coding: utf-8 -*-
"""TRACE: Target Reasoning via Adaptive Combinatorial Evidence Optimization.

Implementation of the DASFAA submission "Target Reasoning via Adaptive
Combinatorial Evidence Optimization for Recommendation".  Equation numbers in
the comments refer to that paper.

Pipeline for a history S_u = ((i^1, b^1, theta^1), ..., (i^L, b^L, theta^L))
and a candidate item c:

1. Historical interaction encoding (Sec. 4.2)
       e^l = e_i + e_b,  h^l = GRU(e^l, h^{l-1}),  dh^l = h^l - h^{l-1},
       m^l = A_{b^l} dh^l,  z_c = e_c + e_{b*}.
2. Path utility (Eqs. 1-4)
       N^l(c)   = cos(z_c, m^l) - lambda_len
       E^{p,l}  = cos(dh^p, dh^l) - lambda_tr[b^p, b^l] log(1 + theta^l - theta^p)
       F^l(c)   = cos(z_c, h^l) - lambda_end[b^l] log(1 + theta^L - theta^l)
       U(P, c)  = sum_{j<K} N^{l_j} + sum_{j>=2} E^{l_{j-1}, l_j} + F^{l_K}.
3. Exact ranked-path DP (Eqs. 6-8): the MAP path and the Omega best
   alternatives.  Path indices are recomputed every forward pass and are not
   differentiated; their utilities are.
4. Suffix DP over the full path space (Eqs. 9-10): conditional next-link
   probabilities eta^{p,l}(c).
5. Structure-guided path transformer (Eqs. 11-12) with candidate-conditioned
   queries and the structural bias log R^{j,k}.
6. MAP / utility-weighted soft branches, learned gate and cosine score
   (Eqs. 13-15), trained with the BCE objective of Eq. (16).
"""

import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.utils import InputType


# Finite "log zero" for differentiable quantities.  Using -inf there would
# produce NaN gradients in logsumexp/softmax rows that are entirely masked.
LOG_ZERO = -1e9

ABLATIONS = (
    'none',
    'wo_N',        # N^l(c) = 0
    'wo_E',        # E^{p,l} = 0
    'wo_F',        # F^l(c) = 0
    'map_only',    # Omega = 0
    'soft_only',   # remove the MAP branch from the score
    'wo_bias',     # remove log R^{j,k} from attention
    'wo_cq',       # remove z_c from the attention query
    'wo_tf',       # remove the path transformer
    'wo_gru',      # cumulative sums of interaction embeddings instead of GRU
    'shared_A',    # one projection A shared across behaviors
)


class PathTransformerBlock(nn.Module):
    """One block of the structure-guided path transformer (Eq. 12)."""

    def __init__(self, hidden_size, n_heads, inner_size, dropout_prob,
                 layer_norm_eps, candidate_query=True):
        super(PathTransformerBlock, self).__init__()
        self.n_heads = n_heads
        # Each head projects to ceil(d/H) dimensions so that H need not divide
        # d; the concatenated heads are mapped back to d by the output
        # projection.  Scores are scaled by sqrt(d/H) as in Eq. (12).
        self.head_size = math.ceil(hidden_size / n_heads)
        self.attention_size = self.head_size * n_heads
        self.scale = math.sqrt(hidden_size / n_heads)
        self.candidate_query = candidate_query
        query_input = 2 * hidden_size if candidate_query else hidden_size
        self.query = nn.Linear(query_input, self.attention_size)
        self.key = nn.Linear(hidden_size, self.attention_size)
        self.value = nn.Linear(hidden_size, self.attention_size)
        self.output = nn.Linear(self.attention_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout_prob)
        self.hidden_dropout = nn.Dropout(dropout_prob)
        self.attn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.GELU(),
            nn.Linear(inner_size, hidden_size),
        )
        self.ffn_dropout = nn.Dropout(dropout_prob)
        self.ffn_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def _split_heads(self, x):
        n, k, _ = x.shape
        return x.view(n, k, self.n_heads, self.head_size).transpose(1, 2)

    def forward(self, x, candidate, attention_bias):
        """
        Args:
            x: [N, K, d] path-position embeddings.
            candidate: [N, d] candidate representation z_c.
            attention_bias: [N, K, K] log R^{j,k} with LOG_ZERO on masked
                (k > j or padded) entries.
        """
        n, k, d = x.shape
        if self.candidate_query:
            query_input = torch.cat(
                (x, candidate.unsqueeze(1).expand(-1, k, -1)), dim=-1
            )
        else:
            query_input = x
        query = self._split_heads(self.query(query_input))
        key = self._split_heads(self.key(x))
        value = self._split_heads(self.value(x))

        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / self.scale + attention_bias.unsqueeze(1)
        weights = self.attn_dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, value).transpose(1, 2).reshape(
            n, k, self.attention_size
        )

        x = self.attn_norm(x + self.hidden_dropout(self.output(context)))
        x = self.ffn_norm(x + self.ffn_dropout(self.ffn(x)))
        return x


class TRACE(SequentialRecommender):
    """Candidate-specific combinatorial evidence optimization (TRACE)."""

    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(TRACE, self).__init__(config, dataset)

        self.hidden_size = int(config['hidden_size'])
        self.tau = float(config['path_temperature'])
        self.n_blocks = int(config['n_path_blocks'])
        self.n_heads = int(config['n_heads'])
        self.inner_size = int(config['inner_size'])
        self.dropout_prob = float(config['dropout_prob'])
        self.layer_norm_eps = float(config['layer_norm_eps'])
        self.reg_weight = float(config['reg_weight'])
        self.train_neg_num = max(int(config['train_neg_num']), 1)
        self.candidate_chunk_size = max(int(config['candidate_chunk_size']), 1)
        self.train_micro_batch = max(int(config['train_micro_batch'] or 256), 1)
        self.ablation = str(config['trace_ablation'] or 'none')
        if self.ablation not in ABLATIONS:
            raise ValueError(
                'Unknown trace_ablation {!r}; choose from {}.'.format(
                    self.ablation, ABLATIONS
                )
            )
        if self.tau <= 0:
            raise ValueError('path_temperature must be positive.')
        omega = int(config['max_alternatives'])
        if self.ablation == 'map_only':
            omega = 0
        if self.ablation == 'soft_only' and omega < 1:
            raise ValueError('soft_only requires max_alternatives >= 1.')
        self.omega = max(omega, 0)

        self.num_behaviors = int(dataset.num('item_type_list'))
        target_token = str(config['target_behavior_token'] or '0')
        behavior_tokens = dataset.field2token_id['item_type_list']
        if target_token not in behavior_tokens:
            raise ValueError(
                'Target behavior token {!r} is absent from the dataset '
                'behavior vocabulary {}.'.format(
                    target_token, sorted(behavior_tokens)
                )
            )
        self.target_behavior = behavior_tokens[target_token]

        d = self.hidden_size
        self.item_embedding = nn.Embedding(self.n_items, d, padding_idx=0)
        self.behavior_embedding = nn.Embedding(
            self.num_behaviors, d, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, d)
        self.gru = nn.GRU(d, d, batch_first=True)

        # A_b in R^{d x d}, indexed only by the observed behavior (Sec. 4.2).
        projection_count = 1 if self.ablation == 'shared_A' else self.num_behaviors
        self.behavior_projection = nn.Parameter(
            torch.empty(projection_count, d, d)
        )

        # Non-negative structural cost coefficients, lambda = softplus(raw).
        initial_cost = math.log(math.expm1(0.1))
        self.lambda_len_raw = nn.Parameter(torch.tensor(initial_cost))
        self.lambda_tr_raw = nn.Parameter(
            torch.full((self.num_behaviors, self.num_behaviors), initial_cost)
        )
        self.lambda_end_raw = nn.Parameter(
            torch.full((self.num_behaviors,), initial_cost)
        )

        self.blocks = nn.ModuleList([
            PathTransformerBlock(
                d, self.n_heads, self.inner_size, self.dropout_prob,
                self.layer_norm_eps,
                candidate_query=self.ablation != 'wo_cq',
            )
            for _ in range(self.n_blocks)
        ])
        # w_gamma in R^{3d}, b_gamma in R (Eq. 14).
        self.gate = nn.Linear(3 * d, 1)

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier initialization for embedding and projection matrices."""
        for module in self.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        for projection in self.behavior_projection.data:
            nn.init.xavier_uniform_(projection)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()
            self.behavior_embedding.weight[0].zero_()

    # ------------------------------------------------------------------
    # Sec. 4.2 Historical interaction encoding
    # ------------------------------------------------------------------
    def _trim(self, item_seq, type_seq, time_seq):
        lengths = torch.count_nonzero(item_seq, dim=1)
        if torch.any(lengths.eq(0)):
            raise ValueError('TRACE requires a non-empty history.')
        width = int(lengths.max().item())
        item_seq = item_seq[:, :width]
        type_seq = type_seq[:, :width]
        if time_seq is not None:
            time_seq = time_seq[:, :width].float()
        else:
            # Without recorded timestamps the historical position is used as a
            # consistent time unit.
            time_seq = torch.arange(
                width, device=item_seq.device, dtype=torch.float
            ).unsqueeze(0).expand(item_seq.size(0), -1)
        return item_seq, type_seq, time_seq, lengths

    def _encode_history(self, item_seq, type_seq, lengths):
        """Return contextual states h, state changes dh and messages m."""
        e = self.item_embedding(item_seq) + self.behavior_embedding(type_seq)
        if self.ablation == 'wo_gru':
            h = torch.cumsum(e, dim=1)
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                e, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_h, _ = self.gru(packed)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                packed_h, batch_first=True, total_length=e.size(1)
            )
        h_prev = F.pad(h[:, :-1], (0, 0, 1, 0))  # h^0 = 0
        dh = h - h_prev

        if self.ablation == 'shared_A':
            m = F.linear(dh, self.behavior_projection[0])
        else:
            m = torch.zeros_like(dh)
            for behavior in range(1, self.num_behaviors):
                mask = type_seq.eq(behavior)
                if mask.any():
                    m = m + F.linear(
                        dh, self.behavior_projection[behavior]
                    ) * mask.unsqueeze(-1)
        return h, dh, m

    def _candidate_representation(self, candidates):
        target = torch.full_like(candidates, self.target_behavior)
        return self.item_embedding(candidates) + self.behavior_embedding(target)

    # ------------------------------------------------------------------
    # Sec. 4.3 Path utility, Eqs. (1)-(3)
    # ------------------------------------------------------------------
    def _transition_utility(self, dh, type_seq, time_seq, valid):
        """E^{p,l}, shared across candidates.  Shape [B, L, L] (p, l)."""
        batch_size, length = type_seq.shape
        positions = torch.arange(length, device=type_seq.device)
        forward_edge = (
            positions.view(1, -1, 1) < positions.view(1, 1, -1)
        ) & valid.unsqueeze(2) & valid.unsqueeze(1)
        if self.ablation == 'wo_E':
            edge = dh.new_zeros(batch_size, length, length)
        else:
            dh_unit = F.normalize(dh, dim=-1)
            similarity = torch.bmm(dh_unit, dh_unit.transpose(1, 2))
            decay = F.softplus(self.lambda_tr_raw)[
                type_seq.unsqueeze(2), type_seq.unsqueeze(1)
            ]
            gap = (time_seq.unsqueeze(1) - time_seq.unsqueeze(2)).clamp_min(0.0)
            edge = similarity - decay * torch.log1p(gap)
        return edge.masked_fill(~forward_edge, LOG_ZERO), forward_edge

    def _candidate_utilities(self, z, h, m, type_seq, time_seq, lengths, valid):
        """N^l(c) and F^l(c).  Shapes [B, C, L]."""
        batch_size, length = type_seq.shape
        z_unit = F.normalize(z, dim=-1)
        if self.ablation == 'wo_N':
            node = z.new_zeros(batch_size, z.size(1), length)
        else:
            m_unit = F.normalize(m, dim=-1)
            node = torch.einsum('bcd,bld->bcl', z_unit, m_unit)
            node = node - F.softplus(self.lambda_len_raw)
        if self.ablation == 'wo_F':
            endpoint = z.new_zeros(batch_size, z.size(1), length)
        else:
            h_unit = F.normalize(h, dim=-1)
            last_time = time_seq.gather(1, (lengths - 1).unsqueeze(1))
            age = (last_time - time_seq).clamp_min(0.0)
            decay = F.softplus(self.lambda_end_raw)[type_seq]
            endpoint = torch.einsum('bcd,bld->bcl', z_unit, h_unit)
            endpoint = endpoint - (decay * torch.log1p(age)).unsqueeze(1)
        mask = ~valid.unsqueeze(1)
        return node.masked_fill(mask, LOG_ZERO), endpoint.masked_fill(mask, LOG_ZERO)

    # ------------------------------------------------------------------
    # Sec. 4.3 Path selection, Eqs. (6)-(8)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _ranked_paths(self, node, edge, endpoint, valid):
        """Exact top-(Omega+1) paths under U(P, c).

        Returns:
            paths: [B, C, M, K] increasing historical positions, padded with 0.
            path_len: [B, C, M] number of selected positions (0 if the entry
                does not exist because fewer than M paths are feasible).
        Entry 0 is the MAP path P*; entries 1..M-1 are the alternatives.
        """
        batch_size, candidate_count, length = node.shape
        beam = self.omega + 1
        device = node.device
        neg_inf = float('-inf')
        valid_c = valid.unsqueeze(1)
        node = node.masked_fill(~valid_c, neg_inf)
        endpoint = endpoint.masked_fill(~valid_c, neg_inf)
        edge = edge.masked_fill(edge.le(LOG_ZERO / 2), neg_inf)

        # D^l: partial paths ending at l, excluding the node/endpoint term of l.
        scores = node.new_full((batch_size, candidate_count, length, beam), neg_inf)
        back_pos = torch.full(
            (batch_size, candidate_count, length, beam), -1,
            dtype=torch.long, device=device,
        )
        back_rank = torch.zeros_like(back_pos)
        start = node.new_zeros(batch_size, candidate_count, 1)
        for l in range(length):
            if l == 0:
                pool = start
            else:
                extension = (
                    scores[:, :, :l]
                    + (node[:, :, :l] + edge[:, :l, l].unsqueeze(1)).unsqueeze(-1)
                )
                pool = torch.cat(
                    (start, extension.reshape(batch_size, candidate_count, -1)),
                    dim=-1,
                )
            k = min(beam, pool.size(-1))
            top_scores, top_index = pool.topk(k, dim=-1)
            from_start = top_index.eq(0)
            source = top_index - 1
            scores[:, :, l, :k] = top_scores
            back_pos[:, :, l, :k] = torch.where(
                from_start, torch.full_like(source, -1), torch.div(source, beam, rounding_mode='floor')
            )
            back_rank[:, :, l, :k] = torch.where(
                from_start, torch.zeros_like(source), source % beam
            )
        scores = scores.masked_fill(~valid_c.unsqueeze(-1), neg_inf)

        final = (scores + endpoint.unsqueeze(-1)).reshape(
            batch_size, candidate_count, -1
        )
        final_scores, final_index = final.topk(beam, dim=-1)
        exists = torch.isfinite(final_scores)
        current_pos = torch.where(exists, torch.div(final_index, beam, rounding_mode='floor'), torch.full_like(final_index, -1))
        current_rank = final_index % beam

        # Backtracking with predecessor pointers (positions in reverse order).
        flat_pos = back_pos.reshape(batch_size, candidate_count, -1)
        flat_rank = back_rank.reshape(batch_size, candidate_count, -1)
        reversed_positions = []
        while bool(current_pos.ge(0).any()):
            reversed_positions.append(current_pos)
            pointer = (current_pos.clamp_min(0) * beam + current_rank)
            next_pos = flat_pos.gather(2, pointer)
            next_rank = flat_rank.gather(2, pointer)
            alive = current_pos.ge(0)
            current_pos = torch.where(alive, next_pos, current_pos)
            current_rank = torch.where(alive, next_rank, current_rank)
        positions = torch.stack(reversed_positions, dim=-1)  # [B, C, M, K]
        path_len = positions.ge(0).sum(dim=-1)
        # Sort into chronological order; padding (-1 -> length) goes last.
        positions = positions.masked_fill(positions.lt(0), length)
        positions, _ = positions.sort(dim=-1)
        positions = positions.masked_fill(positions.ge(length), 0)
        return positions, path_len

    # ------------------------------------------------------------------
    # Sec. 4.3 Connection probabilities, Eqs. (9)-(10)
    # ------------------------------------------------------------------
    def _log_suffix_weights(self, node, edge, endpoint, valid):
        """log beta^l(c) for every position, computed in reverse order."""
        batch_size, candidate_count, length = node.shape
        stop = endpoint / self.tau
        log_beta = [None] * length
        for l in range(length - 1, -1, -1):
            if l == length - 1:
                current = stop[:, :, l]
            else:
                later = torch.stack(log_beta[l + 1:], dim=-1)
                continuation = (
                    node[:, :, l:l + 1] + edge[:, l, l + 1:].unsqueeze(1)
                ) / self.tau + later
                continuation = continuation.masked_fill(
                    edge[:, l, l + 1:].le(LOG_ZERO / 2).unsqueeze(1), LOG_ZERO
                )
                current = torch.logsumexp(
                    torch.cat((stop[:, :, l:l + 1], continuation), dim=-1),
                    dim=-1,
                )
            log_beta[l] = current.masked_fill(~valid[:, l].unsqueeze(1), LOG_ZERO)
        return torch.stack(log_beta, dim=-1)

    # ------------------------------------------------------------------
    # Scoring a candidate set, Eqs. (11)-(15)
    # ------------------------------------------------------------------
    def _score(self, h, dh, m, type_seq, time_seq, lengths, valid,
               edge, candidates):
        batch_size, length = type_seq.shape
        candidate_count = candidates.size(1)
        beam = self.omega + 1
        d = self.hidden_size

        z = self._candidate_representation(candidates)  # [B, C, d]
        node, endpoint = self._candidate_utilities(
            z, h, m, type_seq, time_seq, lengths, valid
        )
        paths, path_len = self._ranked_paths(node, edge, endpoint, valid)
        max_len = paths.size(-1)
        path_exists = path_len.gt(0)
        step_mask = (
            torch.arange(max_len, device=paths.device).view(1, 1, 1, -1)
            < path_len.unsqueeze(-1)
        )
        link_mask = step_mask[..., 1:]

        def gather_candidate(values, index):
            # values [B, C, L], index [B, C, M, K] -> [B, C, M, K]
            flat = index.reshape(batch_size, candidate_count, -1)
            return values.gather(2, flat).view_as(index)

        def gather_edges(values, source, target):
            # values [B, L, L] -> [B, C, M, K-1]
            flat = values.reshape(batch_size, -1)
            index = (source * length + target).reshape(batch_size, -1)
            return flat.gather(1, index).view_as(source)

        endpoint_index = (path_len - 1).clamp_min(0)

        # Differentiable path utilities U(P, c), Eq. (4).
        node_on_path = gather_candidate(node, paths)
        edge_on_path = gather_edges(edge, paths[..., :-1], paths[..., 1:])
        endpoint_pos = paths.gather(3, endpoint_index.unsqueeze(-1)).squeeze(-1)
        endpoint_on_path = endpoint.gather(2, endpoint_pos)
        nonterminal = step_mask & ~(
            torch.arange(max_len, device=paths.device).view(1, 1, 1, -1)
            .eq(endpoint_index.unsqueeze(-1))
        )
        utility = (
            (node_on_path * nonterminal).sum(-1)
            + (edge_on_path * link_mask).sum(-1)
            + endpoint_on_path
        )
        utility = utility.masked_fill(~path_exists, LOG_ZERO)

        # Next-link log-probabilities log eta^{l_v, l_{v+1}}(c), Eq. (10).
        log_beta = self._log_suffix_weights(node, edge, endpoint, valid)
        log_eta = (
            (node_on_path[..., :-1] + edge_on_path) / self.tau
            + gather_candidate(log_beta, paths[..., 1:])
            - gather_candidate(log_beta, paths[..., :-1])
        )
        log_eta = log_eta * link_mask

        # Structural bias log R^{j,k} = sum_{v=k}^{j-1} log eta, Eq. (11).
        cumulative = F.pad(torch.cumsum(log_eta, dim=-1), (1, 0))
        log_r = cumulative.unsqueeze(-1) - cumulative.unsqueeze(-2)
        causal = torch.tril(torch.ones(
            max_len, max_len, dtype=torch.bool, device=paths.device
        ))
        attend = causal & step_mask.unsqueeze(-2)
        if self.ablation == 'wo_bias':
            log_r = torch.zeros_like(log_r)
        attention_bias = log_r.masked_fill(~attend, LOG_ZERO)

        # Path-position inputs x^j = m^{l_j} + p_{l_j}.
        path_count = batch_size * candidate_count * beam
        flat_paths = paths.reshape(batch_size, -1)
        messages = m.gather(
            1, flat_paths.unsqueeze(-1).expand(-1, -1, d)
        ).view(path_count, max_len, d)
        position_ids = paths.reshape(path_count, max_len).clamp(
            max=self.max_seq_length - 1
        )
        x = messages + self.position_embedding(position_ids)
        flat_step_mask = step_mask.reshape(path_count, max_len)
        flat_len = path_len.reshape(path_count)
        flat_endpoint = endpoint_index.reshape(path_count)

        if self.ablation == 'wo_tf':
            weights = flat_step_mask.float()
            path_repr = (x * weights.unsqueeze(-1)).sum(1) / weights.sum(
                1, keepdim=True
            ).clamp_min(1.0)
        else:
            z_flat = z.unsqueeze(2).expand(-1, -1, beam, -1).reshape(path_count, d)
            bias_flat = attention_bias.reshape(path_count, max_len, max_len)
            for block in self.blocks:
                x = block(x, z_flat, bias_flat)
            path_repr = x[torch.arange(path_count, device=x.device), flat_endpoint]
        path_repr = path_repr * flat_len.gt(0).unsqueeze(-1)
        path_repr = path_repr.view(batch_size, candidate_count, beam, d)

        # MAP and utility-weighted soft branches, Eq. (13).
        r_map = path_repr[:, :, 0]
        score_map = F.cosine_similarity(r_map, z, dim=-1)
        if self.omega == 0:
            return score_map
        alternative_exists = path_exists[:, :, 1:]
        has_alternative = alternative_exists.any(dim=-1)
        rho = torch.softmax(
            (utility[:, :, 1:] / self.tau).masked_fill(~alternative_exists, LOG_ZERO),
            dim=-1,
        ) * alternative_exists
        r_soft = (rho.unsqueeze(-1) * path_repr[:, :, 1:]).sum(dim=2)
        score_soft = F.cosine_similarity(r_soft, z, dim=-1)

        if self.ablation == 'soft_only':
            score = score_soft
        else:
            # Candidate-dependent gate, Eqs. (14)-(15).
            gamma = torch.sigmoid(
                self.gate(torch.cat((r_map, r_soft, z), dim=-1))
            ).squeeze(-1)
            score = gamma * score_map + (1.0 - gamma) * score_soft
        # When Omega_u = 0 (a single feasible path) only the MAP branch exists.
        return torch.where(has_alternative, score, score_map)

    def candidate_scores(self, item_seq, type_seq, candidates, time_seq=None):
        """o_u(c) for every candidate in ``candidates`` ([B, C])."""
        item_seq, type_seq, time_seq, lengths = self._trim(
            item_seq, type_seq, time_seq
        )
        positions = torch.arange(item_seq.size(1), device=item_seq.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        h, dh, m = self._encode_history(item_seq, type_seq, lengths)
        edge, _ = self._transition_utility(dh, type_seq, time_seq, valid)
        blocks = []
        for block in candidates.split(self.candidate_chunk_size, dim=1):
            blocks.append(self._score(
                h, dh, m, type_seq, time_seq, lengths, valid, edge, block
            ))
        return torch.cat(blocks, dim=1)

    # ------------------------------------------------------------------
    # Sec. 4.5 Training objective, Eq. (16)
    # ------------------------------------------------------------------
    def _sample_negatives(self, positives):
        """Uniformly sampled negatives from I \\ {c+} (distinct per row)."""
        count = self.train_neg_num
        negatives = torch.randint(
            1, self.n_items, (positives.size(0), count), device=positives.device
        )
        for _ in range(16):
            duplicate = negatives.unsqueeze(-1).eq(negatives.unsqueeze(1)).tril(-1).any(-1)
            invalid = negatives.eq(positives.unsqueeze(1)) | duplicate
            if not invalid.any():
                break
            resample = torch.randint_like(negatives, 1, self.n_items)
            negatives = torch.where(invalid, resample, negatives)
        return negatives

    @staticmethod
    def _time_sequence(interaction):
        if 'item_time_list' in interaction:
            return interaction['item_time_list']
        return None

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction['item_type_list']
        positives = interaction[self.POS_ITEM_ID]
        time_seq = self._time_sequence(interaction)
        negatives = self._sample_negatives(positives)
        candidates = torch.cat((positives.unsqueeze(1), negatives), dim=1)
        # The whole batch forms one optimization step; rows are processed in
        # gradient-checkpointed micro-batches only to bound activation memory.
        # Rows are grouped by history length so that each micro-batch is padded
        # to a similar width; the mean loss does not depend on row order.
        anchor = torch.zeros((), device=item_seq.device, requires_grad=True)
        by_length = torch.argsort(torch.count_nonzero(item_seq, dim=1))
        chunks = []
        for rows in by_length.split(self.train_micro_batch):
            def score_rows(grad_anchor, rows=rows):
                return self.candidate_scores(
                    item_seq[rows], type_seq[rows], candidates[rows],
                    None if time_seq is None else time_seq[rows],
                ) + grad_anchor * 0.0

            if torch.is_grad_enabled():
                chunks.append(checkpoint(score_rows, anchor))
            else:
                chunks.append(score_rows(anchor))
        scores = torch.cat(chunks, dim=0)
        # -log sigma(o+) - mean log sigma(-o-), written with softplus.
        loss = F.softplus(-scores[:, 0]) + F.softplus(scores[:, 1:]).mean(dim=1)
        loss = loss.mean()
        if self.reg_weight > 0:
            l2 = sum(p.pow(2).sum() for p in self.parameters() if p.requires_grad)
            loss = loss + self.reg_weight * l2
        return loss

    def predict(self, interaction):
        """Score (history, candidate) rows of a sampled-ranking batch.

        Rows of the same evaluation instance share one history, so each
        history is encoded once and all of its candidates are scored together.
        Candidates are scored independently, so this does not change any score.
        """
        item_seq = interaction[self.ITEM_SEQ]
        type_seq = interaction['item_type_list']
        time_seq = self._time_sequence(interaction)
        candidates = interaction[self.ITEM_ID]
        if self.USER_ID not in interaction:
            return self.candidate_scores(
                item_seq, type_seq, candidates.view(-1, 1), time_seq
            ).squeeze(1)

        _, group = torch.unique(interaction[self.USER_ID], return_inverse=True)
        row_count = group.numel()
        order = torch.argsort(group * row_count + torch.arange(
            row_count, device=group.device
        ))
        counts = torch.bincount(group)
        starts = torch.cumsum(counts, 0) - counts
        slot = torch.empty_like(group)
        slot[order] = torch.arange(row_count, device=group.device) - starts[group[order]]
        first_row = order[starts]

        grouped_candidates = candidates.new_zeros(counts.numel(), int(counts.max()))
        grouped_candidates[group, slot] = candidates
        scores = self.candidate_scores(
            item_seq[first_row], type_seq[first_row], grouped_candidates,
            None if time_seq is None else time_seq[first_row],
        )
        return scores[group, slot]

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        candidates = torch.arange(1, self.n_items, device=item_seq.device)
        candidates = candidates.unsqueeze(0).expand(item_seq.size(0), -1)
        scores = self.candidate_scores(
            item_seq,
            interaction['item_type_list'],
            candidates,
            self._time_sequence(interaction),
        )
        return F.pad(scores, (1, 0), value=-float('inf'))

    def customized_sort_predict(self, interaction):
        raise NotImplementedError(
            'TRACE scores the candidates supplied by RecBole; set customized_eval=0.'
        )
