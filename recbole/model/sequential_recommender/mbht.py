# -*- coding: utf-8 -*-
# @Time    : 2020/9/18 12:08
# @Author  : Hui Wang
# @Email   : hui.wang@ruc.edu.cn

import math
import random
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder, HGNN


def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.matmul(z1, z2.permute(0,2,1))

class MBHT(SequentialRecommender):

    def __init__(self, config, dataset):
        super(MBHT, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.hidden_size = config['hidden_size']  # same as embedding_size
        self.inner_size = config['inner_size']  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.attn_dropout_prob = config['attn_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = config['layer_norm_eps']

        self.mask_ratio = config['mask_ratio']

        self.loss_type = config['loss_type']
        self.initializer_range = config['initializer_range']

        self.hglen = config['hyper_len']
        self.enable_hg = config['enable_hg']
        self.enable_ms = config['enable_ms']
        self.dataset = config['dataset']

        # C3Route is a candidate-conditioned, hard-budget reranker.  It is
        # disabled by default so the original MBHT remains an exact baseline.
        self.enable_c3route = bool(config['enable_c3route'])
        self.route_topk = int(config['route_topk'])
        self.route_weight = float(config['route_weight'])
        self.route_aux_weight = float(config['route_aux_weight'])
        self.route_coverage_weight = float(config['route_coverage_weight'])
        self.route_negatives = int(config['route_negatives'])
        self.route_mode = str(config['route_mode'])
        self.route_stages = int(config['route_stages'])
        self.route_dp_temperature = float(config['route_temperature'])
        self.route_rerank_topr = int(config['route_rerank_topr'])
        self.enable_temporal = bool(config['enable_temporal'])
        self.horizon_bins = [float(value) for value in config['horizon_bins']]

        self.buy_type = dataset.field2token_id["item_type_list"]['0']

        # load dataset info
        self.mask_token = self.n_items
        self.mask_item_length = int(self.mask_ratio * self.max_seq_length)

        # define layers and loss
        self.type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.hidden_size, padding_idx=0)  # mask token add 1
        self.position_embedding = nn.Embedding(self.max_seq_length + 1, self.hidden_size)  # add mask_token at the last
        if self.enable_ms:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=True,
                scales=config["scales"]
            )
        else:
            self.trm_encoder = TransformerEncoder(
                n_layers=self.n_layers,
                n_heads=self.n_heads,
                hidden_size=self.hidden_size,
                inner_size=self.inner_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                attn_dropout_prob=self.attn_dropout_prob,
                hidden_act=self.hidden_act,
                layer_norm_eps=self.layer_norm_eps,
                multiscale=False
            )
        self.hgnn_layer = HGNN(self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.hg_type_embedding = nn.Embedding(6, self.hidden_size, padding_idx=0)
        self.metric_w1 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.metric_w2 = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.gating_weight = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.gating_bias = nn.Parameter(torch.Tensor(1, self.hidden_size))
        self.attn_weights = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.attn = nn.Parameter(torch.Tensor(1, self.hidden_size))
        if self.enable_c3route:
            self.route_type_bias = nn.Embedding(6, 1, padding_idx=0)
            self.route_recency_raw = nn.Parameter(torch.tensor(-2.0))
            self.route_temperature_raw = nn.Parameter(torch.tensor(0.0))
            self.route_stage_query = nn.Parameter(
                torch.empty(self.route_stages, self.hidden_size)
            )
            self.route_transition_src = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False
            )
            self.route_transition_dst = nn.Linear(
                self.hidden_size, self.hidden_size, bias=False
            )
            self.route_behavior_transition = nn.Parameter(torch.zeros(6, 6))
            self.route_gap_raw = nn.Parameter(torch.full((self.route_stages,), -2.0))
            self.route_time_projection = nn.Linear(3, self.hidden_size, bias=False)
            # Start as a residual correction to a warm-started backbone.  The
            # gate is learned, so routing is admitted only when validation
            # gradients show that it improves the ranking objective.
            self.route_mix_raw = nn.Parameter(torch.tensor(-2.0))
        if self.enable_temporal:
            self.temporal_decay_raw = nn.Embedding(6, self.hidden_size, padding_idx=0)
            self.temporal_center = nn.Embedding(6, self.hidden_size, padding_idx=0)
            self.temporal_time_projection = nn.Linear(3, self.hidden_size)
            self.temporal_gru = nn.GRUCell(self.hidden_size, self.hidden_size)
            # A small initial residual avoids erasing a converged discrete-time
            # representation before the irregular-time branch has adapted.
            self.temporal_mix_raw = nn.Parameter(torch.tensor(-4.0))
        nn.init.normal_(self.attn, std=0.02)
        nn.init.normal_(self.attn_weights, std=0.02)
        # nn.init.normal_(self.gating_bias, std=0.02)
        nn.init.normal_(self.gating_weight, std=0.02)
        nn.init.normal_(self.metric_w1, std=0.02)
        nn.init.normal_(self.metric_w2, std=0.02)

        if self.dataset == "retail_beh":
            self.sw_before = 10
            self.sw_follow = 6
        elif self.dataset == "ijcai_beh":
            self.sw_before = 30
            self.sw_follow = 18
        elif self.dataset == "tmall_beh":
            self.sw_before = 20
            self.sw_follow = 12
        else:
            self.sw_before = 10
            self.sw_follow = 6

        self.hypergraphs = dict()
        # we only need compute the loss at the masked position
        try:
            assert self.loss_type in ['BPR', 'CE']
        except AssertionError:
            raise AssertionError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        if self.enable_c3route:
            nn.init.zeros_(self.route_type_bias.weight)
            nn.init.normal_(self.route_stage_query, std=self.initializer_range)
        if self.enable_temporal:
            nn.init.constant_(self.temporal_decay_raw.weight, -3.0)
            self.temporal_decay_raw.weight.data[0].zero_()

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_attention_mask(self, item_seq):
        """Generate bidirectional attention mask for multi-scale attention."""
        if self.enable_ms:
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1)
            return extended_attention_mask
        else:
            """Generate bidirectional attention mask for multi-head attention."""
            attention_mask = (item_seq > 0).long()
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
            # bidirectional mask
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            return extended_attention_mask

    def _padding_sequence(self, sequence, max_length):
        pad_len = max_length - len(sequence)
        sequence = [0] * pad_len + sequence
        sequence = sequence[-max_length:]  # truncate according to the max_length
        return sequence

    def reconstruct_train_data(self, item_seq, type_seq, last_buy):
        """Vectorized masked-item construction with the target kept last.

        It preserves the author protocol: append the target purchase, mask
        historical positions independently, keep the last ``mask_item_length``
        reconstruction targets, and place the target position in the final
        column used by the C3Route auxiliary loss.
        """
        device = item_seq.device
        batch_size = item_seq.size(0)
        target_positions = torch.count_nonzero(item_seq, dim=1)
        zero_padding = torch.zeros(batch_size, dtype=torch.long, device=device)
        sequence = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)
        behavior = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        rows = torch.arange(batch_size, device=device)
        sequence[rows, target_positions] = last_buy
        behavior[rows, target_positions] = self.buy_type

        positions = torch.arange(sequence.size(1), device=device).unsqueeze(0)
        history_mask = positions < target_positions.unsqueeze(1)
        random_mask = torch.rand(sequence.shape, device=device).lt(self.mask_ratio)
        selected_mask = (random_mask & history_mask) | positions.eq(target_positions.unsqueeze(1))

        positive_source = sequence.clone()
        masked_sequence = sequence.masked_fill(selected_mask, self.mask_token)
        masked_behavior = behavior.masked_fill(selected_mask, 0)

        ranked_positions = torch.where(selected_mask, positions.expand_as(sequence), -1)
        keep = min(self.mask_item_length, sequence.size(1))
        selected_positions = torch.topk(
            ranked_positions, keep, dim=1, largest=True, sorted=True
        ).values.flip(1)
        valid = selected_positions.ge(0)
        gather_positions = selected_positions.clamp_min(0)
        pos_items = positive_source.gather(1, gather_positions) * valid.long()
        masked_index = gather_positions * valid.long()

        if keep < self.mask_item_length:
            pad = self.mask_item_length - keep
            pos_items = F.pad(pos_items, (pad, 0))
            masked_index = F.pad(masked_index, (pad, 0))
        return masked_sequence, pos_items, masked_index, masked_behavior

    def reconstruct_train_data_reference(self, item_seq, type_seq, last_buy):
        """
        Mask item sequence for training.
        """
        last_buy = last_buy.tolist()
        device = item_seq.device
        batch_size = item_seq.size(0)

        zero_padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)
        item_seq = torch.cat((item_seq, zero_padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        type_seq = torch.cat((type_seq, zero_padding.unsqueeze(-1)), dim=-1)
        n_objs = (torch.count_nonzero(item_seq, dim=1)+1).tolist()
        for batch_id in range(batch_size):
            n_obj = n_objs[batch_id]
            item_seq[batch_id][n_obj-1] = last_buy[batch_id]
            type_seq[batch_id][n_obj-1] = self.buy_type

        sequence_instances = item_seq.cpu().numpy().tolist()
        type_instances = type_seq.cpu().numpy().tolist()

        # Masked Item Prediction
        # [B * Len]
        masked_item_sequence = []
        pos_items = []
        masked_index = []

        for instance_idx, instance in enumerate(sequence_instances):
            # WE MUST USE 'copy()' HERE!
            masked_sequence = instance.copy()
            pos_item = []
            index_ids = []
            for index_id, item in enumerate(instance):
                # padding is 0, the sequence is end
                if index_id == n_objs[instance_idx]-1:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)
                    break
                prob = random.random()
                if prob < self.mask_ratio:
                    pos_item.append(item)
                    masked_sequence[index_id] = self.mask_token
                    type_instances[instance_idx][index_id] = 0
                    index_ids.append(index_id)

            masked_item_sequence.append(masked_sequence)
            pos_items.append(self._padding_sequence(pos_item, self.mask_item_length))
            masked_index.append(self._padding_sequence(index_ids, self.mask_item_length))

        # [B Len]
        masked_item_sequence = torch.tensor(masked_item_sequence, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        pos_items = torch.tensor(pos_items, dtype=torch.long, device=device).view(batch_size, -1)
        # [B mask_len]
        masked_index = torch.tensor(masked_index, dtype=torch.long, device=device).view(batch_size, -1)
        type_instances = torch.tensor(type_instances, dtype=torch.long, device=device).view(batch_size, -1)
        return masked_item_sequence, pos_items, masked_index, type_instances

    def reconstruct_test_data(self, item_seq, item_seq_len, item_type):
        """
        Add mask token at the last position according to the lengths of item_seq
        """
        padding = torch.zeros(item_seq.size(0), dtype=torch.long, device=item_seq.device)  # [B]
        item_seq = torch.cat((item_seq, padding.unsqueeze(-1)), dim=-1)  # [B max_len+1]
        item_type = torch.cat((item_type, padding.unsqueeze(-1)), dim=-1)
        for batch_id, last_position in enumerate(item_seq_len):
            item_seq[batch_id][last_position] = self.mask_token
        return item_seq, item_type

    def _masked_window_readout(self, embeddings, item_seq, mask_positions_nums=None):
        """Apply MBHT's local mean readout without per-example GPU syncs.

        Training masks are processed from left to right, exactly as in the
        author implementation, so a later masked position can consume an
        earlier updated readout.  Work within a mask slot is batched.
        """
        output = embeddings.clone()
        batch_size, sequence_length, hidden_size = output.shape
        rows = torch.arange(batch_size, device=output.device)
        n_objects = torch.count_nonzero(item_seq, dim=1)

        if mask_positions_nums is None:
            positions = item_seq.eq(self.mask_token).long().argmax(dim=1)
            starts = (positions - self.sw_before).clamp_min(0)
            prefix = F.pad(output.cumsum(dim=1), (0, 0, 1, 0))
            sums = prefix[rows, positions] - prefix[rows, starts]
            counts = (positions - starts).clamp_min(1).to(output.dtype)
            output[rows, positions] = sums / counts.unsqueeze(1)
            return output

        masked_positions, mask_counts = mask_positions_nums
        slot_count = masked_positions.size(1)
        for slot in range(slot_count):
            positions = masked_positions[:, slot]
            active = slot >= (slot_count - mask_counts)
            active = active & positions.gt(0)

            starts = (positions - self.sw_before).clamp_min(0)
            ends = torch.minimum(
                positions + self.sw_follow,
                (n_objects - 1).clamp_min(0),
            )
            prefix = F.pad(output.cumsum(dim=1), (0, 0, 1, 0))
            after_starts = torch.minimum(
                (positions + 1).clamp_max(sequence_length), ends
            )
            sums = (
                prefix[rows, positions]
                - prefix[rows, starts]
                + prefix[rows, ends]
                - prefix[rows, after_starts]
            )
            raw_counts = (positions - starts) + (ends - positions - 1).clamp_min(0)
            counts = raw_counts.clamp_min(1).to(output.dtype)
            readout = sums / counts.unsqueeze(1)
            valid = active & raw_counts.gt(0)
            output[rows[valid], positions[valid]] = readout[valid]
        return output

    def _masked_window_readout_reference(self, embeddings, item_seq, mask_positions_nums=None):
        """Loop-based reference retained for regression tests."""
        batch_size = item_seq.shape[0]
        n_objects = torch.count_nonzero(item_seq, dim=1).tolist()
        outputs = []
        for batch_idx in range(batch_size):
            n_obj = n_objects[batch_idx]
            embs = embeddings[batch_idx].clone()
            if mask_positions_nums is not None:
                mask_len = mask_positions_nums[1][batch_idx]
                positions = mask_positions_nums[0][batch_idx][-mask_len:].tolist()
                for pos in positions:
                    if pos == 0:
                        continue
                    start = pos - self.sw_before if pos - self.sw_before > -1 else 0
                    end = pos + self.sw_follow if pos + self.sw_follow < n_obj else n_obj - 1
                    embs[pos] = torch.mean(
                        torch.cat((embs[start:pos], embs[pos + 1:end]), dim=0),
                        dim=0,
                    )
            else:
                pos = (item_seq[batch_idx] == self.mask_token).nonzero(as_tuple=True)[0][0]
                start = pos - self.sw_before if pos - self.sw_before > -1 else 0
                embs[pos] = torch.mean(embs[start:pos], dim=0)
            outputs.append(embs)
        return torch.stack(outputs, dim=0)

    @staticmethod
    def _time_features(values):
        log_values = torch.log1p(values.clamp_min(0.0))
        return torch.stack(
            (log_values, torch.sin(log_values), torch.cos(log_values)), dim=-1
        )

    def _route_scale(self):
        return self.route_weight * torch.sigmoid(self.route_mix_raw)

    def _continuous_time_states(self, input_emb, item_seq, type_seq, time_seq):
        """Bounded decay sub-step followed by an event-driven GRU jump."""
        gaps = torch.zeros_like(time_seq, dtype=input_emb.dtype)
        gaps[:, 1:] = (time_seq[:, 1:] - time_seq[:, :-1]).clamp_min(0.0)
        gaps = gaps.clamp_max(24.0 * 365.0)
        time_messages = self.temporal_time_projection(self._time_features(gaps))
        state = torch.zeros(
            input_emb.size(0), input_emb.size(-1),
            device=input_emb.device, dtype=input_emb.dtype
        )
        states = []
        valid = item_seq.ne(0) & item_seq.ne(self.mask_token)
        for position in range(input_emb.size(1)):
            behavior = type_seq[:, position]
            rate = F.softplus(self.temporal_decay_raw(behavior))
            decay = torch.exp(-rate * gaps[:, position].unsqueeze(-1))
            center = self.temporal_center(behavior)
            decayed = decay * state + (1.0 - decay) * center
            jumped = self.temporal_gru(
                input_emb[:, position] + time_messages[:, position], decayed
            )
            state = torch.where(valid[:, position].unsqueeze(-1), jumped, state)
            states.append(state)
        return torch.stack(states, dim=1)

    def forward(
        self, item_seq, type_seq, mask_positions_nums=None, session_id=None,
        time_seq=None
    ):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)
        type_embedding = self.type_embedding(type_seq)
        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding + type_embedding
        if self.enable_temporal and time_seq is not None:
            temporal_states = self._continuous_time_states(
                input_emb, item_seq, type_seq, time_seq
            )
            input_emb = input_emb + torch.sigmoid(
                self.temporal_mix_raw
            ) * temporal_states
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)
        extended_attention_mask = self.get_attention_mask(item_seq)
        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]

        if self.enable_hg:
            x_raw = item_emb
            x_raw = x_raw * torch.sigmoid(x_raw.matmul(self.gating_weight)+self.gating_bias)
            # b, l, l
            x_m = torch.stack((self.metric_w1*x_raw, self.metric_w2*x_raw)).mean(0)
            item_sim = sim(x_m, x_m)
            item_sim[item_sim < 0] = 0.01

            Gs = self.build_Gs_fully_batched(item_seq, item_sim, self.hglen)
            # Gs = self.build_Gs_light(item_seq, item_sim, self.hglen)

            hgnn_embs = self.hgnn_layer(x_raw, Gs)
            hgnn_embs = self._masked_window_readout(
                hgnn_embs, item_seq, mask_positions_nums
            )
            # x = x_raw
            # 2, b, l, dim
            mixed_x = torch.stack((output, hgnn_embs), dim=0)
            weights = (torch.matmul(mixed_x, self.attn_weights.unsqueeze(0).unsqueeze(0))*self.attn).sum(-1)
            # 2, b, l, 1
            score = F.softmax(weights, dim=0).unsqueeze(-1)
            mixed_x = (mixed_x*score).sum(0)
            # mixed_x = self.bert.forward_from_emb(tokens, beh_types, mixed_x)
            # b, s, n
            # mixed_x = self.out(mixed_x)
            assert not torch.isnan(mixed_x).any()

            # mask_pos = item_seq==self.mask_token
            # attn_score_mask = score.squeeze().permute(1,2,0)[mask_pos]
            # attn_score_mask = score.squeeze().permute(1,2,0)
            # out_self_attn = torch.matmul(output, output.transpose(-2,-1)) / math.sqrt(output.size(-1))
            # out_self_attn = F.softmax(out_self_attn, -1)
            return mixed_x
        return output  # [B L H]

    def multi_hot_embed(self, masked_index, max_length):
        """
        For memory, we only need calculate loss for masked position.
        Generate a multi-hot vector to indicate the masked position for masked sequence, and then is used for
        gathering the masked position hidden representation.

        Examples:
            sequence: [1 2 3 4 5]

            masked_sequence: [1 mask 3 mask 5]

            masked_index: [1, 3]

            max_length: 5

            multi_hot_embed: [[0 1 0 0 0], [0 0 0 1 0]]
        """
        masked_index = masked_index.view(-1)
        multi_hot = torch.zeros(masked_index.size(0), max_length, device=masked_index.device)
        multi_hot[torch.arange(masked_index.size(0)), masked_index] = 1
        return multi_hot

    def _sample_route_negatives(self, positives):
        """Sample fixed-count negatives for the route auxiliary objective."""
        negatives = torch.randint(
            1,
            self.n_items,
            (positives.size(0), self.route_negatives),
            device=positives.device,
        )
        collision = negatives.eq(positives.unsqueeze(1))
        while collision.any():
            negatives[collision] = torch.randint(
                1, self.n_items, (int(collision.sum().item()),), device=positives.device
            )
            collision = negatives.eq(positives.unsqueeze(1))
        return negatives

    def _candidate_local_route_scores(
        self,
        item_seq,
        type_seq,
        candidates,
        target_positions,
        return_coverage=False,
    ):
        """Score reranking candidates through a causal hard Top-K history route.

        The forward pass uses exactly ``route_topk`` historical events whenever
        that many causal events exist.  The backward pass uses soft weights as a
        straight-through surrogate, keeping the discrete communication budget
        inspectable while allowing behavior and recency utilities to learn.
        """
        history = F.normalize(self.item_embedding(item_seq), dim=-1)
        query = F.normalize(self.item_embedding(candidates), dim=-1)
        semantic = torch.einsum('bch,blh->bcl', query, history)

        type_bias = self.route_type_bias(type_seq).squeeze(-1).unsqueeze(1)
        positions = torch.arange(item_seq.size(1), device=item_seq.device)
        causal_mask = positions.view(1, 1, -1) < target_positions.view(-1, 1, 1)
        history_mask = item_seq.ne(0).unsqueeze(1) & item_seq.ne(self.mask_token).unsqueeze(1)
        valid_mask = causal_mask & history_mask

        distance = target_positions.view(-1, 1, 1) - positions.view(1, 1, -1)
        recency_scale = F.softplus(self.route_recency_raw)
        recency = -recency_scale * torch.log1p(distance.clamp_min(0).float())
        utility = semantic + type_bias + recency
        utility = utility.masked_fill(~valid_mask, -1e4)

        temperature = F.softplus(self.route_temperature_raw) + 0.1
        soft_weights = F.softmax(utility / temperature, dim=-1)
        soft_weights = soft_weights * valid_mask.float()
        soft_weights = soft_weights / soft_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        budget = min(self.route_topk, item_seq.size(1))
        top_indices = torch.topk(utility, budget, dim=-1, sorted=False).indices
        hard_mask = torch.zeros_like(utility).scatter_(-1, top_indices, 1.0)
        hard_mask = hard_mask * valid_mask.float()
        selected_utility = utility.masked_fill(hard_mask.eq(0), -1e4)
        hard_weights = F.softmax(selected_utility / temperature, dim=-1) * hard_mask
        hard_weights = hard_weights / hard_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        # Hard routes in the forward pass; soft route gradients in the backward pass.
        route_weights = hard_weights.detach() + soft_weights - soft_weights.detach()
        route_score = (route_weights * semantic).sum(dim=-1)
        has_history = valid_mask.any(dim=-1)
        route_score = route_score * has_history.float()

        if not return_coverage:
            return route_score

        type_one_hot = F.one_hot(type_seq.clamp(min=0, max=5), num_classes=6).float()
        group_mass = torch.einsum('bcl,blt->bct', soft_weights, type_one_hot)
        present = torch.einsum('bcl,blt->bct', valid_mask.float(), type_one_hot).gt(0)
        present_count = present.sum(dim=-1).clamp_min(1).float()
        concentration = group_mass.square().sum(dim=-1) - present_count.reciprocal()
        coverage_loss = concentration.clamp_min(0).mean()
        return route_score, coverage_loss

    def candidate_route_scores(
        self,
        item_seq,
        type_seq,
        candidates,
        target_positions,
        return_coverage=False,
        history_states=None,
        target_horizons=None,
    ):
        """Dispatch to the local Top-K baseline or global trajectory router."""
        if getattr(self, 'route_mode', 'local') == 'global':
            return self._candidate_global_route_scores(
                item_seq,
                type_seq,
                candidates,
                target_positions,
                return_coverage=return_coverage,
                history_states=history_states,
                target_horizons=target_horizons,
            )
        return self._candidate_local_route_scores(
            item_seq,
            type_seq,
            candidates,
            target_positions,
            return_coverage=return_coverage,
        )

    def _candidate_global_route_scores(
        self,
        item_seq,
        type_seq,
        candidates,
        target_positions,
        return_coverage=False,
        history_states=None,
        target_horizons=None,
        return_diagnostics=False,
    ):
        """Optimize a complete causal evidence path with batched Viterbi DP.

        Candidate-specific screening first keeps ``route_topk`` events.  The
        retained events are restored to chronological order and form a layered
        DAG.  Viterbi gives the exact maximum for the declared unary/pairwise
        energy on that screened DAG.  A forward--backward relaxation supplies
        soft path marginals for the straight-through gradient estimator.
        """
        if self.route_stages < 1:
            raise ValueError('route_stages must be positive')
        if self.route_topk < self.route_stages:
            raise ValueError('route_topk must be at least route_stages')

        batch_size, candidate_count = candidates.shape
        sequence_length = item_seq.size(1)
        hidden_size = self.item_embedding.embedding_dim
        device = item_seq.device

        raw_history = (
            history_states
            if history_states is not None
            else self.item_embedding(item_seq)
        )
        history = F.normalize(raw_history, dim=-1)
        query_raw = self.item_embedding(candidates)
        if target_horizons is not None:
            if target_horizons.dim() == 1:
                target_horizons = target_horizons.unsqueeze(1)
            target_horizons = target_horizons.expand(-1, candidate_count)
            query_raw = query_raw + self.route_time_projection(
                self._time_features(target_horizons.to(query_raw.dtype))
            )
        query = F.normalize(query_raw, dim=-1)
        semantic = torch.einsum('bch,blh->bcl', query, history)

        positions = torch.arange(sequence_length, device=device)
        causal = positions.unsqueeze(0) < target_positions.unsqueeze(1)
        valid_history = causal & item_seq.ne(0) & item_seq.ne(self.mask_token)
        type_bias = self.route_type_bias(type_seq).squeeze(-1).unsqueeze(1)
        distance = target_positions.view(-1, 1, 1) - positions.view(1, 1, -1)
        recency = -F.softplus(self.route_recency_raw) * torch.log1p(
            distance.clamp_min(0).float()
        )

        screen_utility = semantic + type_bias + recency
        neg_inf = torch.tensor(
            float('-inf'), device=device, dtype=screen_utility.dtype
        )
        screen_utility = screen_utility.masked_fill(
            ~valid_history.unsqueeze(1), neg_inf
        )
        screen_size = min(self.route_topk, sequence_length)
        screen_indices = torch.topk(
            screen_utility, screen_size, dim=-1, sorted=False
        ).indices
        screen_indices = torch.sort(screen_indices, dim=-1).values

        expanded_valid = valid_history.unsqueeze(1).expand(
            -1, candidate_count, -1
        )
        selected_valid = torch.gather(expanded_valid, 2, screen_indices)
        selected_positions = positions.view(1, 1, -1).expand(
            batch_size, candidate_count, -1
        ).gather(2, screen_indices)

        node_index = screen_indices.unsqueeze(-1).expand(
            -1, -1, -1, hidden_size
        )
        selected_nodes = torch.gather(
            history.unsqueeze(1).expand(-1, candidate_count, -1, -1),
            2,
            node_index,
        )
        selected_semantic = torch.gather(semantic, 2, screen_indices)
        selected_types = torch.gather(
            type_seq.unsqueeze(1).expand(-1, candidate_count, -1),
            2,
            screen_indices,
        )
        selected_type_bias = self.route_type_bias(selected_types).squeeze(-1)
        selected_recency = torch.gather(
            recency.expand(-1, candidate_count, -1), 2, screen_indices
        )

        stage_affinity = torch.einsum(
            'bckh,sh->bcsk', selected_nodes, self.route_stage_query
        ) / math.sqrt(hidden_size)
        unary = (
            selected_semantic.unsqueeze(2)
            + stage_affinity
            + selected_type_bias.unsqueeze(2)
            + selected_recency.unsqueeze(2)
        )
        invalid_node = ~selected_valid.unsqueeze(2)
        hard_unary = unary.masked_fill(invalid_node, neg_inf)
        # The soft dynamic program is a gradient surrogate.  A finite floor
        # prevents undefined gradients through all--infinity logsumexp slices;
        # the hard Viterbi recurrence below still uses structural -infinity.
        unary = unary.masked_fill(invalid_node, -1e4)

        source = F.normalize(self.route_transition_src(selected_nodes), dim=-1)
        destination = F.normalize(
            self.route_transition_dst(selected_nodes), dim=-1
        )
        base_transition = torch.einsum(
            'bckh,bcjh->bckj', source, destination
        ) / math.sqrt(hidden_size)
        behavior_transition = self.route_behavior_transition[
            selected_types.unsqueeze(-1), selected_types.unsqueeze(-2)
        ]
        gaps = (
            selected_positions.unsqueeze(-2) - selected_positions.unsqueeze(-1)
        )
        edge_valid = (
            gaps.gt(0)
            & selected_valid.unsqueeze(-1)
            & selected_valid.unsqueeze(-2)
        )

        temperature = max(float(self.route_dp_temperature), 1e-3)
        hard_value = hard_unary[:, :, 0]
        soft_forward = [unary[:, :, 0]]
        predecessors = []
        transitions = [None]
        for stage in range(1, self.route_stages):
            gap_penalty = F.softplus(self.route_gap_raw[stage]) * torch.log1p(
                gaps.clamp_min(0).float()
            )
            transition_raw = base_transition + behavior_transition - gap_penalty
            hard_transition = transition_raw.masked_fill(~edge_valid, neg_inf)
            transition = transition_raw.masked_fill(~edge_valid, -1e4)
            transitions.append(transition)

            hard_candidates = hard_value.unsqueeze(-1) + hard_transition
            best_prefix, predecessor = hard_candidates.max(dim=-2)
            predecessors.append(predecessor)
            hard_value = hard_unary[:, :, stage] + best_prefix

            soft_candidates = (
                soft_forward[-1].unsqueeze(-1) + transition
            )
            soft_value = unary[:, :, stage] + temperature * torch.logsumexp(
                soft_candidates / temperature, dim=-2
            )
            soft_forward.append(soft_value)

        hard_terminal = selected_semantic.masked_fill(~selected_valid, neg_inf)
        terminal = selected_semantic.masked_fill(~selected_valid, -1e4)
        terminal_value = hard_value + hard_terminal
        best_energy, current = terminal_value.max(dim=-1)
        has_path = selected_valid.sum(dim=-1).ge(self.route_stages)

        hard_path = torch.zeros(
            batch_size,
            candidate_count,
            self.route_stages,
            screen_size,
            dtype=history.dtype,
            device=device,
        )
        hard_path[:, :, -1].scatter_(-1, current.unsqueeze(-1), 1.0)
        for stage in range(self.route_stages - 1, 0, -1):
            current = torch.gather(
                predecessors[stage - 1], -1, current.unsqueeze(-1)
            ).squeeze(-1)
            hard_path[:, :, stage - 1].scatter_(
                -1, current.unsqueeze(-1), 1.0
            )
        hard_path = hard_path * has_path.unsqueeze(-1).unsqueeze(-1).float()

        soft_backward = [None] * self.route_stages
        soft_backward[-1] = terminal
        for stage in range(self.route_stages - 2, -1, -1):
            transition = transitions[stage + 1]
            suffix_candidates = (
                transition
                + unary[:, :, stage + 1].unsqueeze(-2)
                + soft_backward[stage + 1].unsqueeze(-2)
            )
            soft_backward[stage] = temperature * torch.logsumexp(
                suffix_candidates / temperature, dim=-1
            )

        log_partition = temperature * torch.logsumexp(
            (soft_forward[-1] + terminal) / temperature, dim=-1
        )
        safe_log_partition = torch.where(
            has_path, log_partition, torch.zeros_like(log_partition)
        )
        marginals = []
        for stage in range(self.route_stages):
            marginal = torch.exp(
                (
                    soft_forward[stage]
                    + soft_backward[stage]
                    - safe_log_partition.unsqueeze(-1)
                )
                / temperature
            )
            marginal = torch.nan_to_num(marginal, nan=0.0, posinf=0.0)
            marginal = marginal * selected_valid.float()
            marginal = marginal / marginal.sum(dim=-1, keepdim=True).clamp_min(
                1e-12
            )
            marginals.append(marginal)
        soft_path = torch.stack(marginals, dim=2)
        soft_path = soft_path * has_path.unsqueeze(-1).unsqueeze(-1).float()

        hard_representation = (
            hard_path.unsqueeze(-1) * selected_nodes.unsqueeze(2)
        ).sum(dim=-2).mean(dim=2)
        soft_representation = (
            soft_path.unsqueeze(-1) * selected_nodes.unsqueeze(2)
        ).sum(dim=-2).mean(dim=2)
        route_representation = (
            hard_representation.detach()
            + soft_representation
            - soft_representation.detach()
        )
        # The path must affect ranking through the declared global objective,
        # not only through a pooled representation of whichever nodes Viterbi
        # happened to select.  The forward value is the exact hard optimum;
        # gradients follow the corresponding soft log partition.
        safe_best_energy = torch.where(
            has_path, best_energy, torch.zeros_like(best_energy)
        )
        structured_energy = (
            safe_best_energy.detach()
            + safe_log_partition
            - safe_log_partition.detach()
        ) / float(self.route_stages + 1)
        route_score = (
            (route_representation * query).sum(dim=-1)
            + structured_energy
        )
        route_score = route_score * has_path.float()

        diagnostics = None
        if return_diagnostics:
            diagnostics = {
                'best_energy': best_energy,
                'hard_path': hard_path,
                'unary': unary,
                'transitions': torch.stack(transitions[1:], dim=2),
                'terminal': terminal,
                'selected_positions': selected_positions,
                'selected_valid': selected_valid,
                'soft_path': soft_path,
                'log_partition': log_partition,
            }

        if not return_coverage:
            if return_diagnostics:
                return route_score, diagnostics
            return route_score

        type_one_hot = F.one_hot(
            selected_types.clamp(min=0, max=5), num_classes=6
        ).float()
        mean_marginal = soft_path.mean(dim=2)
        group_mass = torch.einsum('bck,bckt->bct', mean_marginal, type_one_hot)
        present = torch.einsum(
            'bck,bckt->bct', selected_valid.float(), type_one_hot
        ).gt(0)
        present_count = present.sum(dim=-1).clamp_min(1).float()
        concentration = group_mass.square().sum(dim=-1) - present_count.reciprocal()
        coverage_loss = concentration.clamp_min(0).mean()
        if return_diagnostics:
            return route_score, coverage_loss, diagnostics
        return route_score, coverage_loss

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        session_id = interaction['session_id']
        item_type = interaction["item_type_list"]
        last_buy = interaction["item_id"]
        masked_item_seq, pos_items, masked_index, item_type_seq = self.reconstruct_train_data(item_seq, item_type, last_buy)

        time_seq = None
        target_horizons = None
        if self.enable_temporal and 'item_time_list' in interaction:
            source_times = interaction['item_time_list'].float()
            time_seq = F.pad(source_times, (0, 1))
            target_positions_for_time = torch.count_nonzero(item_seq, dim=1)
            rows = torch.arange(item_seq.size(0), device=item_seq.device)
            last_positions = (target_positions_for_time - 1).clamp_min(0)
            last_times = source_times[rows, last_positions]
            target_horizons = interaction['target_delta'].float().clamp_min(0.0)
            time_seq[rows, target_positions_for_time] = last_times + target_horizons

        mask_nums = torch.count_nonzero(pos_items, dim=1)
        sequence_states = self.forward(
            masked_item_seq,
            item_type_seq,
            mask_positions_nums=(masked_index, mask_nums),
            session_id=session_id,
            time_seq=time_seq,
        )
        pred_index_map = self.multi_hot_embed(masked_index, masked_item_seq.size(-1))  # [B*mask_len max_len]
        # [B mask_len] -> [B mask_len max_len] multi hot
        pred_index_map = pred_index_map.view(masked_index.size(0), masked_index.size(1), -1)  # [B mask_len max_len]
        # [B mask_len max_len] * [B max_len H] -> [B mask_len H]
        # only calculate loss for masked position
        seq_output = torch.bmm(pred_index_map, sequence_states)  # [B mask_len H]

        loss_fct = nn.CrossEntropyLoss(reduction='none')
        test_item_emb = self.item_embedding.weight  # [item_num H]
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))  # [B mask_len item_num]
        targets = (masked_index > 0).float().view(-1)  # [B*mask_len]

        loss = torch.sum(loss_fct(logits.view(-1, test_item_emb.size(0)), pos_items.view(-1)) * targets) \
                / torch.sum(targets)

        if self.enable_c3route:
            # The last masked position is always the target purchase appended by
            # reconstruct_train_data; random BERT masks remain trained by CE.
            target_positions = masked_index[:, -1]
            positives = pos_items[:, -1]
            negatives = self._sample_route_negatives(positives)
            candidates = torch.cat((positives.unsqueeze(1), negatives), dim=1)

            route_scores, coverage_loss = self.candidate_route_scores(
                masked_item_seq,
                item_type_seq,
                candidates,
                target_positions,
                return_coverage=True,
                history_states=sequence_states,
                target_horizons=target_horizons,
            )
            target_states = self.gather_indexes(sequence_states, target_positions)
            candidate_emb = self.item_embedding(candidates)
            base_scores = torch.bmm(candidate_emb, target_states.unsqueeze(-1)).squeeze(-1)
            rerank_scores = base_scores + self._route_scale() * route_scores
            route_targets = torch.zeros(rerank_scores.size(0), dtype=torch.long, device=rerank_scores.device)
            route_loss = F.cross_entropy(rerank_scores, route_targets)
            loss = loss + self.route_aux_weight * route_loss \
                + self.route_coverage_weight * coverage_loss
        return loss

    def full_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        time_seq = None
        if self.enable_temporal and 'item_time_list' in interaction:
            time_seq = F.pad(interaction['item_time_list'].float(), (0, 1))
        sequence_states = self.forward(item_seq, type_seq, time_seq=time_seq)
        seq_output = self.gather_indexes(sequence_states, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding.weight[:self.n_items]  # delete masked token
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B, item_num]
        if self.enable_c3route:
            rerank_count = min(self.route_rerank_topr, scores.size(1) - 1)
            _, rerank_candidates = torch.topk(
                scores[:, 1:], rerank_count, dim=1, sorted=False
            )
            rerank_candidates = rerank_candidates + 1
            if self.enable_temporal:
                horizon_scores = []
                for horizon in self.horizon_bins:
                    target_horizons = torch.full(
                        (item_seq.size(0),), horizon,
                        device=item_seq.device, dtype=sequence_states.dtype
                    )
                    horizon_scores.append(self.candidate_route_scores(
                        item_seq,
                        type_seq,
                        rerank_candidates,
                        item_seq_len,
                        history_states=sequence_states,
                        target_horizons=target_horizons,
                    ))
                route_scores = torch.logsumexp(
                    torch.stack(horizon_scores, dim=0), dim=0
                ) - math.log(len(horizon_scores))
            else:
                route_scores = self.candidate_route_scores(
                    item_seq,
                    type_seq,
                    rerank_candidates,
                    item_seq_len,
                    history_states=sequence_states,
                )
            scores.scatter_add_(
                1, rerank_candidates, self._route_scale() * route_scores
            )
        return scores

    def customized_sort_predict(self, interaction):
        item_seq = interaction['item_id_list']
        type_seq = interaction['item_type_list']
        truth = interaction['item_id']
        session_ids = interaction['session_id']
        if self.dataset == "ijcai_beh":
            raw_candidates = [73, 3050, 22557, 5950, 4391, 6845, 1800, 2261, 13801, 2953, 4164, 32090, 3333, 44733, 7380, 790, 1845, 2886, 2366, 21161, 6512, 1689, 337, 3963, 3108, 715, 169, 2558, 6623, 888, 6708, 3585, 501, 308, 9884, 1405, 5494, 6609, 7433, 25101, 3580, 145, 3462, 5340, 1131, 6681, 7776, 8678, 52852, 19229, 4160, 33753, 4356, 920, 15312, 43106, 16669, 1850, 2855, 43807, 15, 8719, 89, 3220, 36, 2442, 9299, 8189, 701, 300, 526, 4564, 516, 1184, 178, 2834, 16455, 9392, 22037, 344, 15879, 3374, 2984, 3581, 11479, 6927, 779, 5298, 10195, 39739, 663, 9137, 24722, 7004, 7412, 89534, 2670, 100, 6112, 1355]
        elif self.dataset == "retail_beh":
            raw_candidates = [101, 11, 14, 493, 163, 593, 1464, 12, 297, 123, 754, 790, 243, 250, 508, 673, 1161, 523, 41, 561, 2126, 196, 1499, 1093, 1138, 1197, 745, 1431, 682, 1567, 440, 1604, 145, 1109, 2146, 209, 2360, 426, 1756, 46, 1906, 520, 3956, 447, 1593, 1119, 894, 2561, 381, 939, 213, 1343, 733, 554, 2389, 1191, 1330, 1264, 2466, 2072, 1024, 2015, 739, 144, 1004, 314, 1868, 3276, 1184, 866, 1020, 2940, 5966, 3805, 221, 11333, 5081, 685, 87, 2458, 415, 669, 1336, 3419, 2758, 2300, 1681, 2876, 2612, 2405, 585, 702, 3876, 1416, 466, 7628, 572, 3385, 220, 772]
        elif self.dataset == "tmall_beh":
            raw_candidates = [2544, 7010, 4193, 32270, 22086, 7768, 647, 7968, 26512, 4575, 63971, 2121, 7857, 5134, 416, 1858, 34198, 2146, 778, 12583, 13899, 7652, 4552, 14410, 1272, 21417, 2985, 5358, 36621, 10337, 13065, 1235, 3410, 14180, 5083, 5089, 4240, 10863, 3397, 4818, 58422, 8353, 14315, 14465, 30129, 4752, 5853, 1312, 3890, 6409, 7664, 1025, 16740, 14185, 4535, 670, 17071, 12579, 1469, 853, 775, 12039, 3853, 4307, 5729, 271, 13319, 1548, 449, 2771, 4727, 903, 594, 28184, 126, 27306, 20603, 40630, 907, 5118, 3472, 7012, 10055, 1363, 9086, 5806, 8204, 41711, 10174, 12900, 4435, 35877, 8679, 10369, 2865, 14830, 175, 4434, 11444, 701]
        customized_candidates = list()
        for batch_idx in range(item_seq.shape[0]):
            seen = item_seq[batch_idx].cpu().tolist()
            cands = raw_candidates.copy()
            for i in range(len(cands)):
                if cands[i] in seen:
                    # Evaluation candidate sets must be invariant to batch order
                    # and repeated calls.  Use a session-indexed deterministic
                    # probe instead of runtime Python RNG.
                    session_seed = int(session_ids[batch_idx].item())
                    new_cand = 1 + (
                        session_seed * 1000003
                        + (i + 1) * 9176
                        + int(cands[i]) * 37
                    ) % (self.n_items - 1)
                    while new_cand in seen or new_cand in cands[:i]:
                        new_cand = 1 + (new_cand + 104729) % (self.n_items - 1)
                    cands[i] = new_cand
            cands.insert(0, truth[batch_idx].item())
            customized_candidates.append(cands)
        candidates = torch.LongTensor(customized_candidates).to(item_seq.device)
        item_seq_len = torch.count_nonzero(item_seq, 1)
        item_seq, type_seq = self.reconstruct_test_data(item_seq, item_seq_len, type_seq)
        time_seq = None
        if self.enable_temporal and 'item_time_list' in interaction:
            time_seq = F.pad(interaction['item_time_list'].float(), (0, 1))
        sequence_states = self.forward(item_seq, type_seq, time_seq=time_seq)
        seq_output = self.gather_indexes(sequence_states, item_seq_len)  # [B H]
        test_items_emb = self.item_embedding(candidates)  # delete masked token
        scores = torch.bmm(test_items_emb, seq_output.unsqueeze(-1)).squeeze()  # [B, item_num]
        if self.enable_c3route:
            route_scores = self.candidate_route_scores(
                item_seq,
                type_seq,
                candidates,
                item_seq_len,
                history_states=sequence_states,
            )
            scores = scores + self._route_scale() * route_scores
        return scores

    def build_Gs_unique(self, seqs, item_sim, group_len, batched=False):
        """Build the original MBHT hypergraph without CPU round trips.

        The author implementation below repeatedly converts CUDA tensors to
        Python lists and performs ``list.index`` inside nested loops.  This
        version preserves the same incidence construction while using sorted
        unique-item indices and tensor scatter operations on the input device.
        Column order of duplicate-item hyperedges may differ, but ``G`` is
        invariant to a permutation of incidence-matrix columns.
        """
        graphs = []
        n_objects = torch.count_nonzero(seqs, dim=1).tolist()
        for batch_idx in range(seqs.shape[0]):
            n_obj = n_objects[batch_idx]
            seq = seqs[batch_idx, :n_obj]
            unique, inverse, counts = torch.unique(
                seq, sorted=True, return_inverse=True, return_counts=True
            )
            duplicate_items = unique[(counts > 1) & unique.ne(self.mask_token)]
            n_unique = unique.numel()

            seq_item_sim = item_sim[batch_idx, :n_obj, :n_obj]
            neighbor_count = min(group_len, n_obj)
            metrics, neighbor_positions = torch.topk(
                seq_item_sim, neighbor_count, dim=-1, sorted=False
            )
            similar_items = seq[neighbor_positions]
            mask_hits = similar_items.eq(self.mask_token)
            if mask_hits.any():
                row_ids = torch.arange(n_obj, device=seq.device).unsqueeze(1).expand_as(similar_items)
                similar_items = torch.where(mask_hits, seq[row_ids], similar_items)
                metrics = torch.where(mask_hits, torch.ones_like(metrics), metrics)

            similar_unique_idx = torch.searchsorted(unique, similar_items)
            incidence = torch.zeros(
                (n_obj, n_unique + duplicate_items.numel()),
                device=metrics.device,
                dtype=metrics.dtype,
            )
            incidence.scatter_(1, similar_unique_idx, metrics)

            rows = torch.arange(n_obj, device=seq.device)
            incidence[seq.eq(self.mask_token)] = 0.0
            incidence[rows, inverse] = 1.0
            if duplicate_items.numel() > 0:
                duplicate_idx = torch.searchsorted(duplicate_items, seq)
                safe_idx = duplicate_idx.clamp_max(duplicate_items.numel() - 1)
                is_duplicate = duplicate_items[safe_idx].eq(seq)
                incidence[
                    rows[is_duplicate], n_unique + safe_idx[is_duplicate]
                ] = 1.0

            vertex_degree = incidence.sum(dim=1).clamp_min(1e-12)
            edge_degree = incidence.sum(dim=0).clamp_min(1e-12)
            normalized_incidence = incidence / vertex_degree.unsqueeze(1)
            graph = (normalized_incidence / edge_degree.unsqueeze(0)).mm(incidence.t())
            assert not torch.isnan(graph).any()
            graphs.append(graph)
        if batched:
            seq_len = seqs.size(1)
            padded_graphs = [
                F.pad(graph, (0, seq_len - graph.size(1), 0, seq_len - graph.size(0)))
                for graph in graphs
            ]
            return torch.stack(padded_graphs, dim=0)
        return torch.block_diag(*graphs)

    def build_Gs_fully_batched(self, seqs, item_sim, group_len):
        """Construct the same normalized hypergraph for a whole batch.

        Hyperedge identifiers are arbitrary in ``H D_e^-1 H^T``.  We use the
        first occurrence position of each item as its unique-item edge and the
        corresponding position plus ``L`` as its repeated-behavior edge.  The
        unused columns are zero and therefore make no contribution to ``G``.
        This removes the per-sequence Python loop while preserving the graph.
        """
        batch_size, sequence_length = seqs.shape
        device = seqs.device
        dtype = item_sim.dtype
        positions = torch.arange(sequence_length, device=device)
        valid = positions.unsqueeze(0) < torch.count_nonzero(seqs, dim=1).unsqueeze(1)

        # first_occurrence[b, i] is the first position carrying seqs[b, i].
        same_item = seqs.unsqueeze(2).eq(seqs.unsqueeze(1))
        same_item = same_item & valid.unsqueeze(1)
        first_occurrence = same_item.to(torch.int64).argmax(dim=2)

        neighbor_count = min(group_len, sequence_length)
        candidate_sim = item_sim.masked_fill(~valid.unsqueeze(1), float('-inf'))
        metrics, neighbor_positions = torch.topk(
            candidate_sim, neighbor_count, dim=-1, sorted=False
        )
        neighbor_valid = valid.gather(1, neighbor_positions.view(batch_size, -1))
        neighbor_valid = neighbor_valid.view_as(neighbor_positions) & valid.unsqueeze(2)
        metrics = torch.where(neighbor_valid, metrics, torch.zeros_like(metrics))

        neighbor_items = seqs.gather(1, neighbor_positions.view(batch_size, -1))
        neighbor_items = neighbor_items.view_as(neighbor_positions)
        neighbor_edges = first_occurrence.gather(
            1, neighbor_positions.view(batch_size, -1)
        ).view_as(neighbor_positions)
        mask_hits = neighbor_items.eq(self.mask_token) & neighbor_valid
        neighbor_edges = torch.where(
            mask_hits, first_occurrence.unsqueeze(2), neighbor_edges
        )
        metrics = torch.where(mask_hits, torch.ones_like(metrics), metrics)

        incidence = torch.zeros(
            (batch_size, sequence_length, 2 * sequence_length),
            device=device,
            dtype=dtype,
        )
        incidence.scatter_(2, neighbor_edges, metrics)
        incidence = incidence.masked_fill(seqs.eq(self.mask_token).unsqueeze(2), 0.0)

        ego_edges = first_occurrence.unsqueeze(2)
        incidence.scatter_(2, ego_edges, valid.to(dtype).unsqueeze(2))

        occurrence_count = same_item.sum(dim=2)
        repeated = occurrence_count.gt(1) & seqs.ne(self.mask_token) & valid
        repeated_edges = (first_occurrence + sequence_length).unsqueeze(2)
        incidence.scatter_(2, repeated_edges, repeated.to(dtype).unsqueeze(2))

        vertex_degree = incidence.sum(dim=2).clamp_min(1e-12)
        edge_degree = incidence.sum(dim=1).clamp_min(1e-12)
        normalized = incidence / vertex_degree.unsqueeze(2)
        normalized = normalized / edge_degree.unsqueeze(1)
        graph = torch.bmm(normalized, incidence.transpose(1, 2))
        assert not torch.isnan(graph).any()
        return graph

    def build_Gs_unique_reference(self, seqs, item_sim, group_len):
        Gs = []
        n_objs = torch.count_nonzero(seqs, dim=1).tolist()
        for batch_idx in range(seqs.shape[0]):
            seq = seqs[batch_idx]
            n_obj = n_objs[batch_idx]
            seq = seq[:n_obj].cpu()
            seq_list = seq.tolist()
            unique = torch.unique(seq)
            unique = unique.tolist()
            n_unique = len(unique)

            multibeh_group = seq.tolist()
            for x in unique:
                multibeh_group.remove(x)
            multibeh_group = list(set(multibeh_group))
            try:
                multibeh_group.remove(self.mask_token)
            except:
                pass
            
            # l', l'
            seq_item_sim = item_sim[batch_idx][:n_obj, :][:, :n_obj]            
            # l', group_len
            if group_len>n_obj:
                metrics, sim_items = torch.topk(seq_item_sim, n_obj, sorted=False)
            else:
                metrics, sim_items = torch.topk(seq_item_sim, group_len, sorted=False)
            # map indices to item tokens
            sim_items = seq[sim_items]
            row_idx, masked_pos = torch.nonzero(sim_items==self.mask_token, as_tuple=True)
            sim_items[row_idx, masked_pos] = seq[row_idx]
            metrics[row_idx, masked_pos] = 1.0
            # print(sim_items.detach().cpu().tolist())
            multibeh_group = seq.tolist()
            for x in unique:
                multibeh_group.remove(x)
            multibeh_group = list(set(multibeh_group))
            try:
                multibeh_group.remove(self.mask_token)
            except:
                pass
            n_edge = n_unique+len(multibeh_group)
            # hyper graph: n_obj, n_edge
            H = torch.zeros((n_obj, n_edge), device=metrics.device)
            normal_item_indexes = torch.nonzero((seq != self.mask_token), as_tuple=True)[0]
            for idx in normal_item_indexes:
                sim_items_i = sim_items[idx].tolist()
                map_f = lambda x: unique.index(x)
                unique_idx = list(map(map_f, sim_items_i))
                H[idx, unique_idx] = metrics[idx]

            for i, item in enumerate(seq_list):
                ego_idx = unique.index(item)
                H[i, ego_idx] = 1.0
                # multi-behavior hyperedge
                if item in multibeh_group:
                    H[i, n_unique+multibeh_group.index(item)] = 1.0
            # print(H.detach().cpu().tolist())
            # W = torch.ones(n_edge, device=H.device)
            # W = torch.diag(W)
            DV = torch.sum(H, dim=1)
            DE = torch.sum(H, dim=0)
            invDE = torch.diag(torch.pow(DE, -1))
            invDV = torch.diag(torch.pow(DV, -1))
            # DV2 = torch.diag(torch.pow(DV, -0.5))
            HT = H.t()
            G = invDV.mm(H).mm(invDE).mm(HT)
            # G = DV2.mm(H).mm(invDE).mm(HT).mm(DV2)
            assert not torch.isnan(G).any()
            Gs.append(G.to(seqs.device))
        Gs_block_diag = torch.block_diag(*Gs)
        return Gs_block_diag
