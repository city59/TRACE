import copy

import numpy as np
from numpy.random import sample
import torch
from collections import Counter


class AbstractSampler(object):
    def __init__(self, distribution):
        self.distribution = ''
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        self.distribution = distribution
        if distribution == 'popularity':
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        raise NotImplementedError('Method [_uni_sampling] should be implemented')

    def _get_candidates_list(self):
        raise NotImplementedError('Method [_get_candidates_list] should be implemented')

    def _build_alias_table(self):
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        del self.prob[0]
        keys = list(self.prob.keys())
        self.keys = sorted(keys, key=(lambda x: self.prob[x]), reverse=True)[:200]

        self.alias = self.prob.copy()


    def _pop_sampling_static(self, sample_num):
        if not self.keys:
            raise ValueError('Popularity evaluation requires legal items.')
        return np.resize(np.asarray(self.keys, dtype=np.int64), sample_num)

    def _pop_sampling(self, sample_num):
        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num, uni=False):
        if uni:
            return self._uni_sampling(sample_num)
        if self.distribution == 'uniform':
            return self._uni_sampling(sample_num)
        elif self.distribution == 'popularity':
            return self._pop_sampling_static(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def get_used_ids(self):
        raise NotImplementedError('Method [get_used_ids] should be implemented')

    def sample_by_key_ids(self, key_ids, num):
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list), uni=True)
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list), uni=True)
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids)


class Sampler(AbstractSampler):
    def __init__(self, phases, datasets, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        if not isinstance(datasets, list):
            datasets = [datasets]
        if len(phases) != len(datasets):
            raise ValueError(f'Phases {phases} and datasets {datasets} should have the same length.')

        self.phases = phases
        self.datasets = datasets
        self.dataset_name = datasets[0].dataset_name

        self.uid_field = datasets[0].uid_field
        self.iid_field = datasets[0].iid_field

        self.user_num = datasets[0].user_num
        self.item_num = datasets[0].item_num

        super().__init__(distribution=distribution)

    def _get_candidates_list(self):
        candidates_list = []
        training_dataset = self.datasets[0]
        if 'item_id_list' in training_dataset.inter_feat:
            candidates_list.extend(
                training_dataset.inter_feat['item_id_list'].numpy().flatten()
            )
        candidates_list.extend(
            training_dataset.inter_feat[self.iid_field].numpy().flatten()
        )
        return candidates_list

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        used_item_id = dict()
        last = [set() for _ in range(self.user_num)]
        for phase, dataset in zip(self.phases, self.datasets):
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(dataset.inter_feat[self.uid_field].numpy(), dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def set_phase(self, phase):
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        if self.distribution == 'popularity':
            user_ids = np.asarray(user_ids, dtype=np.int64)
            sampled = np.empty((num, len(user_ids)), dtype=np.int64)
            popularity_order = list(self.keys)
            popularity_set = set(popularity_order)
            fallback_order = popularity_order + [
                item for item in range(1, self.item_num)
                if item not in popularity_set
            ]
            for column, user_id in enumerate(user_ids):
                used = self.used_ids[user_id]
                legal = [item for item in fallback_order if item not in used]
                if len(legal) < num:
                    raise ValueError(
                        'The dataset protocol cannot provide enough distinct '
                        'negative candidates for user {}'.format(user_id)
                    )
                sampled[:, column] = legal[:num]
            return torch.tensor(sampled.reshape(-1))
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')


class KGSampler(AbstractSampler):
    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.hid_field = dataset.head_entity_field
        self.tid_field = dataset.tail_entity_field
        self.hid_list = dataset.head_entities
        self.tid_list = dataset.tail_entities

        self.head_entities = set(dataset.head_entities)
        self.entity_num = dataset.entity_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.entity_num, sample_num)

    def _get_candidates_list(self):
        return list(self.hid_list) + list(self.tid_list)

    def get_used_ids(self):
        used_tail_entity_id = np.array([set() for _ in range(self.entity_num)])
        for hid, tid in zip(self.hid_list, self.tid_list):
            used_tail_entity_id[hid].add(tid)

        for used_tail_set in used_tail_entity_id:
            if len(used_tail_set) + 1 == self.entity_num:
                raise ValueError(
                    'Some head entities have relation with all entities, '
                    'which we can not sample negative entities for them.'
                )
        return used_tail_entity_id

    def sample_by_entity_ids(self, head_entity_ids, num=1):
        try:
            return self.sample_by_key_ids(head_entity_ids, num)
        except IndexError:
            for head_entity_id in head_entity_ids:
                if head_entity_id not in self.head_entities:
                    raise ValueError(f'head_entity_id [{head_entity_id}] not exist.')


class RepeatableSampler(AbstractSampler):
    def __init__(self, phases, dataset, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]
        self.phases = phases
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def _get_candidates_list(self):
        return list(self.dataset.inter_feat[self.iid_field].numpy())

    def get_used_ids(self):
        return np.array([set() for _ in range(self.user_num)])

    def sample_by_user_ids(self, user_ids, item_ids, num):
        try:
            self.used_ids = np.array([{i} for i in item_ids])
            return self.sample_by_key_ids(np.arange(len(user_ids)), num)
        except IndexError:
            for user_id in user_ids:
                if user_id < 0 or user_id >= self.user_num:
                    raise ValueError(f'user_id [{user_id}] not exist.')

    def set_phase(self, phase):
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        return new_sampler


class SeqSampler(AbstractSampler):
    def __init__(self, dataset, distribution='uniform'):
        self.dataset = dataset

        self.iid_field = dataset.iid_field
        self.user_num = dataset.user_num
        self.item_num = dataset.item_num

        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.randint(1, self.item_num, sample_num)

    def get_used_ids(self):
        pass

    def sample_neg_sequence(self, pos_sequence):
        total_num = len(pos_sequence)
        value_ids = np.zeros(total_num, dtype=np.int64)
        check_list = np.arange(total_num)
        while len(check_list) > 0:
            value_ids[check_list] = self.sampling(len(check_list))
            check_index = np.where(value_ids[check_list] == pos_sequence[check_list])
            check_list = check_list[check_index]

        return torch.tensor(value_ids)
