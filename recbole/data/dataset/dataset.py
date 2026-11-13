import copy
import pickle
import os
import yaml
from collections import Counter, defaultdict
from logging import getLogger

import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils
from scipy.sparse import coo_matrix

from recbole.data.interaction import Interaction
from recbole.utils import FeatureSource, FeatureType, get_local_time, set_color


class Dataset(object):
    def __init__(self, config):
        self.config = config
        self.dataset_name = config['dataset']
        self.logger = getLogger()
        self._from_scratch()

    def _from_scratch(self):
        self.logger.debug(set_color(f'Loading {self.__class__} from scratch.', 'green'))

        self._get_preset()
        self._get_field_from_config()
        self._load_data(self.dataset_name, self.dataset_path)
        self._init_alias()
        self._data_processing()

    def _get_preset(self):
        self.dataset_path = self.config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.field2seqlen = self.config['seq_len'] or {}
        self.alias = {}
        self._preloaded_weight = {}
        self.benchmark_filename_list = self.config['benchmark_filename']

    def _get_field_from_config(self):
        self.uid_field = self.config['USER_ID_FIELD']
        self.iid_field = self.config['ITEM_ID_FIELD']
        self.label_field = self.config['LABEL_FIELD']
        self.time_field = self.config['TIME_FIELD']

        if (self.uid_field is None) ^ (self.iid_field is None):
            raise ValueError(
                'USER_ID_FIELD and ITEM_ID_FIELD need to be set at the same time or not set at the same time.'
            )

        self.logger.debug(set_color('uid_field', 'blue') + f': {self.uid_field}')
        self.logger.debug(set_color('iid_field', 'blue') + f': {self.iid_field}')

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()
        if self.benchmark_filename_list is None:
            self._data_filtering()

        self._remap_ID_all()
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()

    def _data_filtering(self):
        self._filter_nan_user_or_item()
        self._remove_duplication()
        self._filter_by_field_value()
        self._filter_inter_by_user_or_item()
        self._filter_by_inter_num()
        self._reset_index()

    def _build_feat_name_list(self):
        feat_name_list = [
            feat_name for feat_name in ['inter_feat', 'user_feat', 'item_feat']
            if getattr(self, feat_name, None) is not None
        ]
        if self.config['additional_feat_suffix'] is not None:
            for suf in self.config['additional_feat_suffix']:
                if getattr(self, f'{suf}_feat', None) is not None:
                    feat_name_list.append(f'{suf}_feat')
        return feat_name_list

    def _load_data(self, token, dataset_path):
        if not os.path.exists(dataset_path):
            raise ValueError(
                f'[{dataset_path}] does not exist; build it with scripts/preprocess.py.'
            )
        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        self._load_additional_feat(token, dataset_path)

    def _load_inter_feat(self, token, dataset_path):
        if self.benchmark_filename_list is None:
            inter_feat_path = os.path.join(dataset_path, f'{token}.inter')
            if not os.path.isfile(inter_feat_path):
                raise ValueError(f'File {inter_feat_path} not exist.')

            inter_feat = self._load_feat(inter_feat_path, FeatureSource.INTERACTION)
            self.logger.debug(f'Interaction feature loaded successfully from [{inter_feat_path}].')
            self.inter_feat = inter_feat
        else:
            sub_inter_lens = []
            sub_inter_feats = []
            overall_field2seqlen = defaultdict(int)
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, f'{token}.{filename}.inter')
                if os.path.isfile(file_path):
                    temp = self._load_feat(file_path, FeatureSource.INTERACTION)
                    sub_inter_feats.append(temp)
                    sub_inter_lens.append(len(temp))
                    for field in self.field2seqlen:
                        overall_field2seqlen[field] = max(overall_field2seqlen[field], self.field2seqlen[field])
                else:
                    raise ValueError(f'File {file_path} not exist.')
            inter_feat = pd.concat(sub_inter_feats, ignore_index=True)
            self.inter_feat, self.file_size_list = inter_feat, sub_inter_lens
            self.field2seqlen = overall_field2seqlen

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        feat_path = os.path.join(dataset_path, f'{token}.{source.value}')
        if os.path.isfile(feat_path):
            feat = self._load_feat(feat_path, source)
            self.logger.debug(f'[{source.value}] feature loaded successfully from [{feat_path}].')
        else:
            feat = None
            self.logger.debug(f'[{feat_path}] not found, [{source.value}] features are not loaded.')

        field = getattr(self, field_name, None)
        if feat is not None and field is None:
            raise ValueError(f'{field_name} must be exist if {source.value}_feat exist.')
        if feat is not None and field not in feat:
            raise ValueError(f'{field_name} must be loaded if {source.value}_feat is loaded.')

        if field in self.field2source:
            self.field2source[field] = FeatureSource(source.value + '_id')
        return feat

    def _load_additional_feat(self, token, dataset_path):
        if self.config['additional_feat_suffix'] is None:
            return
        for suf in self.config['additional_feat_suffix']:
            if hasattr(self, f'{suf}_feat'):
                raise ValueError(f'{suf}_feat already exist.')
            feat_path = os.path.join(dataset_path, f'{token}.{suf}')
            if os.path.isfile(feat_path):
                feat = self._load_feat(feat_path, suf)
            else:
                raise ValueError(f'Additional feature file [{feat_path}] not found.')
            setattr(self, f'{suf}_feat', feat)

    def _get_load_and_unload_col(self, source):
        if isinstance(source, FeatureSource):
            source = source.value
        if self.config['load_col'] is None:
            load_col = None
        elif source not in self.config['load_col']:
            load_col = set()
        elif self.config['load_col'][source] == '*':
            load_col = None
        else:
            load_col = set(self.config['load_col'][source])

        if self.config['unload_col'] is not None and source in self.config['unload_col']:
            unload_col = set(self.config['unload_col'][source])
        else:
            unload_col = None

        if load_col and unload_col:
            raise ValueError(f'load_col [{load_col}] and unload_col [{unload_col}] can not be set the same time.')

        self.logger.debug(set_color(f'[{source}]: ', 'pink'))
        self.logger.debug(set_color('\t load_col', 'blue') + f': [{load_col}]')
        self.logger.debug(set_color('\t unload_col', 'blue') + f': [{unload_col}]')
        return load_col, unload_col

    def _load_feat(self, filepath, source):
        self.logger.debug(set_color(f'Loading feature from [{filepath}] (source: [{source}]).', 'green'))

        load_col, unload_col = self._get_load_and_unload_col(source)
        if load_col == set():
            return None

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config['encoding']
        with open(filepath, 'r', encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')
            if load_col is not None and field not in load_col:
                continue
            if unload_col is not None and field in unload_col:
                continue
            if isinstance(source, FeatureSource) or source != 'link':
                self.field2source[field] = source
                self.field2type[field] = ftype
                if not ftype.value.endswith('seq'):
                    self.field2seqlen[field] = 1
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        df = pd.read_csv(
            filepath, delimiter=self.config['field_separator'], usecols=usecols, dtype=dtype, encoding=encoding
        )
        df.columns = columns

        seq_separator = self.config['seq_separator']
        for field in columns:
            ftype = self.field2type[field]
            if not ftype.value.endswith('seq'):
                continue
            df[field].fillna(value='', inplace=True)
            if ftype == FeatureType.TOKEN_SEQ:
                df[field] = [np.array(list(filter(None, _.split(seq_separator)))) for _ in df[field].values]
            elif ftype == FeatureType.FLOAT_SEQ:
                df[field] = [np.array(list(map(float, filter(None, _.split(seq_separator))))) for _ in df[field].values]
            self.field2seqlen[field] = max(map(len, df[field].values))
        return df

    def _set_alias(self, alias_name, default_value):
        alias = self.config[f'alias_of_{alias_name}'] or []
        alias = np.array(list(filter(None, default_value)) + alias)
        _, idx = np.unique(alias, return_index=True)
        self.alias[alias_name] = alias[np.sort(idx)]

    def _init_alias(self):
        self._set_alias('user_id', [self.uid_field])
        self._set_alias('item_id', [self.iid_field])

        for alias_name_1, alias_1 in self.alias.items():
            for alias_name_2, alias_2 in self.alias.items():
                if alias_name_1 != alias_name_2:
                    intersect = np.intersect1d(alias_1, alias_2, assume_unique=True)
                    if len(intersect) > 0:
                        raise ValueError(
                            f'`alias_of_{alias_name_1}` and `alias_of_{alias_name_2}` '
                            f'should not have the same field {list(intersect)}.'
                        )

        self._rest_fields = self.token_like_fields
        for alias_name, alias in self.alias.items():
            isin = np.isin(alias, self._rest_fields, assume_unique=True)
            if isin.all() is False:
                raise ValueError(
                    f'`alias_of_{alias_name}` should not contain '
                    f'non-token-like field {list(alias[~isin])}.'
                )
            self._rest_fields = np.setdiff1d(self._rest_fields, alias, assume_unique=True)

    def _user_item_feat_preparation(self):
        if self.user_feat is not None:
            new_user_df = pd.DataFrame({self.uid_field: np.arange(self.user_num)})
            self.user_feat = pd.merge(new_user_df, self.user_feat, on=self.uid_field, how='left')
            self.logger.debug(set_color('ordering user features by user id.', 'green'))
        if self.item_feat is not None:
            new_item_df = pd.DataFrame({self.iid_field: np.arange(self.item_num)})
            self.item_feat = pd.merge(new_item_df, self.item_feat, on=self.iid_field, how='left')
            self.logger.debug(set_color('ordering item features by item id.', 'green'))

    def _preload_weight_matrix(self):
        preload_fields = self.config['preload_weight']
        if preload_fields is None:
            return

        self.logger.debug(f'Preload weight matrix for {preload_fields}.')

        for preload_id_field, preload_value_field in preload_fields.items():
            if preload_id_field not in self.field2source:
                raise ValueError(f'Preload id field [{preload_id_field}] not exist.')
            if preload_value_field not in self.field2source:
                raise ValueError(f'Preload value field [{preload_value_field}] not exist.')
            pid_source = self.field2source[preload_id_field]
            pv_source = self.field2source[preload_value_field]
            if pid_source != pv_source:
                raise ValueError(
                    f'Preload id field [{preload_id_field}] is from source [{pid_source}],'
                    f'while preload value field [{preload_value_field}] is from source [{pv_source}], '
                    f'which should be the same.'
                )

            id_ftype = self.field2type[preload_id_field]
            value_ftype = self.field2type[preload_value_field]
            if id_ftype != FeatureType.TOKEN:
                raise ValueError(f'Preload id field [{preload_id_field}] should be type token, but is [{id_ftype}].')
            if value_ftype not in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                self.logger.warning(
                    f'Field [{preload_value_field}] with type [{value_ftype}] is not `float` or `float_seq`, '
                    f'which will not be handled by preload matrix.'
                )
                continue

            token_num = self.num(preload_id_field)
            feat = self.field2feats(preload_id_field)[0]
            if value_ftype == FeatureType.FLOAT:
                matrix = np.zeros(token_num)
                matrix[feat[preload_id_field]] = feat[preload_value_field]
            else:
                max_len = self.field2seqlen[preload_value_field]
                matrix = np.zeros((token_num, max_len))
                preload_ids = feat[preload_id_field].values
                preload_values = feat[preload_value_field].to_list()
                for pid, prow in zip(preload_ids, preload_values):
                    length = len(prow)
                    if length <= max_len:
                        matrix[pid, :length] = prow
                    else:
                        matrix[pid] = prow[:max_len]
            self._preloaded_weight[preload_id_field] = matrix

    def _fill_nan(self):
        self.logger.debug(set_color('Filling nan', 'green'))

        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            for field in feat:
                ftype = self.field2type[field]
                if ftype == FeatureType.TOKEN:
                    feat[field].fillna(value=0, inplace=True)
                elif ftype == FeatureType.FLOAT:
                    feat[field].fillna(value=feat[field].mean(), inplace=True)
                else:
                    dtype = np.int64 if ftype == FeatureType.TOKEN_SEQ else np.float
                    feat[field] = feat[field].apply(lambda x: np.array([], dtype=dtype) if isinstance(x, float) else x)

    def _normalize(self):
        if self.config['normalize_field'] is not None and self.config['normalize_all'] is True:
            raise ValueError('Normalize_field and normalize_all can\'t be set at the same time.')

        if self.config['normalize_field']:
            fields = self.config['normalize_field']
            for field in fields:
                if field not in self.field2type:
                    raise ValueError(f'Field [{field}] does not exist.')
                ftype = self.field2type[field]
                if ftype != FeatureType.FLOAT and ftype != FeatureType.FLOAT_SEQ:
                    self.logger.warning(f'{field} is not a FLOAT/FLOAT_SEQ feat, which will not be normalized.')
        elif self.config['normalize_all']:
            fields = self.float_like_fields
        else:
            return

        self.logger.debug(set_color('Normalized fields', 'blue') + f': {fields}')

        for field in fields:
            for feat in self.field2feats(field):
                def norm(arr):
                    mx, mn = max(arr), min(arr)
                    if mx == mn:
                        self.logger.warning(f'All the same value in [{field}] from [{feat}_feat].')
                        arr = 1.0
                    else:
                        arr = (arr - mn) / (mx - mn)
                    return arr

                ftype = self.field2type[field]
                if ftype == FeatureType.FLOAT:
                    feat[field] = norm(feat[field].values)
                elif ftype == FeatureType.FLOAT_SEQ:
                    split_point = np.cumsum(feat[field].agg(len))[:-1]
                    feat[field] = np.split(norm(feat[field].agg(np.concatenate)), split_point)

    def _filter_nan_user_or_item(self):
        for field, name in zip([self.uid_field, self.iid_field], ['user', 'item']):
            feat = getattr(self, name + '_feat')
            if feat is not None:
                dropped_feat = feat.index[feat[field].isnull()]
                if len(dropped_feat):
                    self.logger.warning(
                        f'In {name}_feat, line {list(dropped_feat + 2)}, {field} do not exist, so they will be removed.'
                    )
                    feat.drop(feat.index[dropped_feat], inplace=True)
            if field is not None:
                dropped_inter = self.inter_feat.index[self.inter_feat[field].isnull()]
                if len(dropped_inter):
                    self.logger.warning(
                        f'In inter_feat, line {list(dropped_inter + 2)}, {field} do not exist, so they will be removed.'
                    )
                    self.inter_feat.drop(self.inter_feat.index[dropped_inter], inplace=True)

    def _remove_duplication(self):
        keep = self.config['rm_dup_inter']
        if keep is None:
            return
        self._check_field('uid_field', 'iid_field')

        if self.time_field in self.inter_feat:
            self.inter_feat.sort_values(by=[self.time_field], ascending=True, inplace=True)
            self.logger.info(
                f'Records in original dataset have been sorted by value of [{self.time_field}] in ascending order.'
            )
        else:
            self.logger.warning(
                f'Timestamp field has not been loaded or specified, '
                f'thus strategy [{keep}] of duplication removal may be meaningless.'
            )
        self.inter_feat.drop_duplicates(subset=[self.uid_field, self.iid_field], keep=keep, inplace=True)

    def _filter_by_inter_num(self):
        if self.uid_field is None or self.iid_field is None:
            return

        user_inter_num_interval = self._parse_intervals_str(self.config['user_inter_num_interval'])
        item_inter_num_interval = self._parse_intervals_str(self.config['item_inter_num_interval'])

        user_inter_num = Counter(self.inter_feat[self.uid_field].values) if user_inter_num_interval else Counter()
        item_inter_num = Counter(self.inter_feat[self.iid_field].values) if item_inter_num_interval else Counter()

        while True:
            ban_users = self._get_illegal_ids_by_inter_num(
                field=self.uid_field,
                feat=self.user_feat,
                inter_num=user_inter_num,
                inter_interval=user_inter_num_interval
            )
            ban_items = self._get_illegal_ids_by_inter_num(
                field=self.iid_field,
                feat=self.item_feat,
                inter_num=item_inter_num,
                inter_interval=item_inter_num_interval
            )

            if len(ban_users) == 0 and len(ban_items) == 0:
                break

            if self.user_feat is not None:
                dropped_user = self.user_feat[self.uid_field].isin(ban_users)
                self.user_feat.drop(self.user_feat.index[dropped_user], inplace=True)

            if self.item_feat is not None:
                dropped_item = self.item_feat[self.iid_field].isin(ban_items)
                self.item_feat.drop(self.item_feat.index[dropped_item], inplace=True)

            dropped_inter = pd.Series(False, index=self.inter_feat.index)
            user_inter = self.inter_feat[self.uid_field]
            item_inter = self.inter_feat[self.iid_field]
            dropped_inter |= user_inter.isin(ban_users)
            dropped_inter |= item_inter.isin(ban_items)

            user_inter_num -= Counter(user_inter[dropped_inter].values)
            item_inter_num -= Counter(item_inter[dropped_inter].values)

            dropped_index = self.inter_feat.index[dropped_inter]
            self.logger.debug(f'[{len(dropped_index)}] dropped interactions.')
            self.inter_feat.drop(dropped_index, inplace=True)

    def _get_illegal_ids_by_inter_num(self, field, feat, inter_num, inter_interval=None):
        self.logger.debug(
            set_color('get_illegal_ids_by_inter_num', 'blue') + f': field=[{field}], inter_interval=[{inter_interval}]'
        )

        if inter_interval is not None:
            if len(inter_interval) > 1:
                self.logger.warning(f'More than one interval of interaction number are given!')

        ids = {id_ for id_ in inter_num if not self._within_intervals(inter_num[id_], inter_interval)}

        if feat is not None:
            min_num = inter_interval[0][1] if inter_interval else -1
            for id_ in feat[field].values:
                if inter_num[id_] < min_num:
                    ids.add(id_)
        self.logger.debug(f'[{len(ids)}] illegal_ids_by_inter_num, field=[{field}]')
        return ids

    def _parse_intervals_str(self, intervals_str):
        if intervals_str is None:
            return None

        endpoints = []
        for endpoint_pair_str in str(intervals_str).split(';'):
            endpoint_pair_str = endpoint_pair_str.strip()
            left_bracket, right_bracket = endpoint_pair_str[0], endpoint_pair_str[-1]
            endpoint_pair = endpoint_pair_str[1:-1].split(',')
            if not (len(endpoint_pair) == 2 and left_bracket in ['(', '['] and right_bracket in [')', ']']):
                self.logger.warning(f'{endpoint_pair_str} is an illegal interval!')
                continue

            left_point, right_point = float(endpoint_pair[0]), float(endpoint_pair[1])
            if left_point > right_point:
                self.logger.warning(f'{endpoint_pair_str} is an illegal interval!')

            endpoints.append((left_bracket, left_point, right_point, right_bracket))
        return endpoints

    def _within_intervals(self, num, intervals):
        result = True
        for i, (left_bracket, left_point, right_point, right_bracket) in enumerate(intervals):
            temp_result = num >= left_point if left_bracket == '[' else num > left_point
            temp_result &= num <= right_point if right_bracket == ']' else num < right_point
            result = temp_result if i == 0 else result | temp_result
        return result

    def _filter_by_field_value(self):
        val_intervals = [] if self.config['val_interval'] is None else self.config['val_interval']
        self.logger.debug(set_color('drop_by_value', 'blue') + f': val={val_intervals}')

        for field in val_intervals:
            if field not in self.field2type:
                raise ValueError(f'Field [{field}] not defined in dataset.')
            if self.field2type[field] not in {FeatureType.FLOAT, FeatureType.FLOAT_SEQ}:
                raise ValueError(f'Field [{field}] is not float-like field in dataset, which can\'t be filter.')

            field_val_interval = self._parse_intervals_str(val_intervals[field])
            for feat in self.field2feats(field):
                feat.drop(feat.index[~self._within_intervals(feat[field].values, field_val_interval)], inplace=True)

    def _reset_index(self):
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            if feat.empty:
                raise ValueError('Some feat is empty, please check the filtering settings.')
            feat.reset_index(drop=True, inplace=True)

    def _del_col(self, feat, field):
        self.logger.debug(f'Delete column [{field}].')
        if isinstance(feat, Interaction):
            feat.drop(column=field)
        else:
            feat.drop(columns=field, inplace=True)
        for dct in [self.field2id_token, self.field2token_id, self.field2seqlen, self.field2source, self.field2type]:
            if field in dct:
                del dct[field]

    def _filter_inter_by_user_or_item(self):
        if self.config['filter_inter_by_user_or_item'] is not True:
            return

        remained_inter = pd.Series(True, index=self.inter_feat.index)

        if self.user_feat is not None:
            remained_uids = self.user_feat[self.uid_field].values
            remained_inter &= self.inter_feat[self.uid_field].isin(remained_uids)

        if self.item_feat is not None:
            remained_iids = self.item_feat[self.iid_field].values
            remained_inter &= self.inter_feat[self.iid_field].isin(remained_iids)

        self.inter_feat.drop(self.inter_feat.index[~remained_inter], inplace=True)

    def _set_label_by_threshold(self):
        threshold = self.config['threshold']
        if threshold is None:
            return

        self.logger.debug(f'Set label by {threshold}.')

        if len(threshold) != 1:
            raise ValueError('Threshold length should be 1.')

        self.set_field_property(self.label_field, FeatureType.FLOAT, FeatureSource.INTERACTION, 1)
        for field, value in threshold.items():
            if field in self.inter_feat:
                self.inter_feat[self.label_field] = (self.inter_feat[field] >= value).astype(int)
            else:
                raise ValueError(f'Field [{field}] not in inter_feat.')
            if field != self.label_field:
                self._del_col(self.inter_feat, field)

    def _get_remap_list(self, field_list):
        remap_list = []
        for field in field_list:
            ftype = self.field2type[field]
            for feat in self.field2feats(field):
                remap_list.append((feat, field, ftype))
        return remap_list

    def _remap_ID_all(self):
        for alias in self.alias.values():
            remap_list = self._get_remap_list(alias)
            self._remap(remap_list)

        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def _concat_remaped_tokens(self, remap_list):
        tokens = []
        for feat, field, ftype in remap_list:
            if ftype == FeatureType.TOKEN:
                tokens.append(feat[field].values)
            elif ftype == FeatureType.TOKEN_SEQ:
                tokens.append(feat[field].agg(np.concatenate))
        split_point = np.cumsum(list(map(len, tokens)))[:-1]
        tokens = np.concatenate(tokens)
        return tokens, split_point

    def _remap(self, remap_list):
        if len(remap_list) == 0:
            return
        tokens, split_point = self._concat_remaped_tokens(remap_list)
        new_ids_list, mp = pd.factorize(tokens)
        new_ids_list = np.split(new_ids_list + 1, split_point)
        mp = np.array(['[PAD]'] + list(mp))
        token_id = {t: i for i, t in enumerate(mp)}

        for (feat, field, ftype), new_ids in zip(remap_list, new_ids_list):
            if field not in self.field2id_token:
                self.field2id_token[field] = mp
                self.field2token_id[field] = token_id
            if ftype == FeatureType.TOKEN:
                feat[field] = new_ids
            elif ftype == FeatureType.TOKEN_SEQ:
                split_point = np.cumsum(feat[field].agg(len))[:-1]
                feat[field] = np.split(new_ids, split_point)

    def _change_feat_format(self):
        for feat_name in self.feat_name_list:
            feat = getattr(self, feat_name)
            setattr(self, feat_name, self._dataframe_to_interaction(feat))

    def num(self, field):
        if field not in self.field2type:
            raise ValueError(f'Field [{field}] not defined in dataset.')
        if self.field2type[field] not in {FeatureType.TOKEN, FeatureType.TOKEN_SEQ}:
            return self.field2seqlen[field]
        else:
            return len(self.field2id_token[field])

    def fields(self, ftype=None, source=None):
        ftype = set(ftype) if ftype is not None else set(FeatureType)
        source = set(source) if source is not None else set(FeatureSource)
        ret = []
        for field in self.field2type:
            tp = self.field2type[field]
            src = self.field2source[field]
            if tp in ftype and src in source:
                ret.append(field)
        return ret

    @property
    def float_like_fields(self):
        return self.fields(ftype=[FeatureType.FLOAT, FeatureType.FLOAT_SEQ])

    @property
    def token_like_fields(self):
        return self.fields(ftype=[FeatureType.TOKEN, FeatureType.TOKEN_SEQ])

    @property
    def seq_fields(self):
        return self.fields(ftype=[FeatureType.FLOAT_SEQ, FeatureType.TOKEN_SEQ])

    @property
    def non_seq_fields(self):
        return self.fields(ftype=[FeatureType.FLOAT, FeatureType.TOKEN])

    def set_field_property(self, field, field_type, field_source, field_seqlen):
        self.field2type[field] = field_type
        self.field2source[field] = field_source
        self.field2seqlen[field] = field_seqlen

    def copy_field_property(self, dest_field, source_field):
        self.field2type[dest_field] = self.field2type[source_field]
        self.field2source[dest_field] = self.field2source[source_field]
        self.field2seqlen[dest_field] = self.field2seqlen[source_field]

    def field2feats(self, field):
        if field not in self.field2source:
            raise ValueError(f'Field [{field}] not defined in dataset.')
        if field == self.uid_field:
            feats = [self.inter_feat]
            if self.user_feat is not None:
                feats.append(self.user_feat)
        elif field == self.iid_field:
            feats = [self.inter_feat]
            if self.item_feat is not None:
                feats.append(self.item_feat)
        else:
            source = self.field2source[field]
            if not isinstance(source, str):
                source = source.value
            feats = [getattr(self, f'{source}_feat')]
        return feats

    def token2id(self, field, tokens):
        if isinstance(tokens, str):
            if tokens in self.field2token_id[field]:
                return self.field2token_id[field][tokens]
            else:
                raise ValueError(f'token [{tokens}] is not existed in {field}')
        elif isinstance(tokens, (list, np.ndarray)):
            return np.array([self.token2id(field, token) for token in tokens])
        else:
            raise TypeError(f'The type of tokens [{tokens}] is not supported')

    def id2token(self, field, ids):
        try:
            return self.field2id_token[field][ids]
        except IndexError:
            if isinstance(ids, list):
                raise ValueError(f'[{ids}] is not a one-dimensional list.')
            else:
                raise ValueError(f'[{ids}] is not a valid ids.')

    def counter(self, field):
        if field not in self.inter_feat:
            raise ValueError(f'Field [{field}] is not defined in ``inter_feat``.')
        if self.field2type[field] == FeatureType.TOKEN:
            if isinstance(self.inter_feat, pd.DataFrame):
                return Counter(self.inter_feat[field].values)
            else:
                return Counter(self.inter_feat[field].numpy())
        else:
            raise ValueError(f'Field [{field}] is not a token field.')

    @property
    def user_counter(self):
        self._check_field('uid_field')
        return self.counter(self.uid_field)

    @property
    def item_counter(self):
        self._check_field('iid_field')
        return self.counter(self.iid_field)

    @property
    def user_num(self):
        self._check_field('uid_field')
        return self.num(self.uid_field)

    @property
    def item_num(self):
        self._check_field('iid_field')
        return self.num(self.iid_field)

    @property
    def inter_num(self):
        return len(self.inter_feat)

    @property
    def avg_actions_of_users(self):
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.uid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.uid_field].numpy()).values()))

    @property
    def avg_actions_of_items(self):
        if isinstance(self.inter_feat, pd.DataFrame):
            return np.mean(self.inter_feat.groupby(self.iid_field).size())
        else:
            return np.mean(list(Counter(self.inter_feat[self.iid_field].numpy()).values()))

    @property
    def sparsity(self):
        return 1 - self.inter_num / self.user_num / self.item_num

    def _check_field(self, *field_names):
        for field_name in field_names:
            if getattr(self, field_name, None) is None:
                raise ValueError(f'{field_name} isn\'t set.')

    def join(self, df):
        if self.user_feat is not None and self.uid_field in df:
            df.update(self.user_feat[df[self.uid_field]])
        if self.item_feat is not None and self.iid_field in df:
            df.update(self.item_feat[df[self.iid_field]])
        return df

    def __getitem__(self, index, join=True):
        df = self.inter_feat[index]
        return self.join(df) if join else df

    def __len__(self):
        return len(self.inter_feat)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink')]
        if self.uid_field:
            info.extend([
                set_color('The number of users', 'blue') + f': {self.user_num}',
                set_color('Average actions of users', 'blue') + f': {self.avg_actions_of_users}'
            ])
        if self.iid_field:
            info.extend([
                set_color('The number of items', 'blue') + f': {self.item_num}',
                set_color('Average actions of items', 'blue') + f': {self.avg_actions_of_items}'
            ])
        info.append(set_color('The number of inters', 'blue') + f': {self.inter_num}')
        if self.uid_field and self.iid_field:
            info.append(set_color('The sparsity of the dataset', 'blue') + f': {self.sparsity * 100}%')
        info.append(set_color('Remain Fields', 'blue') + f': {list(self.field2type)}')
        return '\n'.join(info)

    def copy(self, new_inter_feat):
        nxt = copy.copy(self)
        nxt.inter_feat = new_inter_feat
        return nxt

    def _drop_unused_col(self):
        unused_col = self.config['unused_col']
        if unused_col is None:
            return

        for feat_name, unused_fields in unused_col.items():
            feat = getattr(self, feat_name + '_feat')
            for field in unused_fields:
                if field not in feat:
                    self.logger.warning(
                        f'Field [{field}] is not in [{feat_name}_feat], which can not be set in `unused_col`.'
                    )
                    continue
                self._del_col(feat, field)

    def _grouped_index(self, group_by_list):
        index = {}
        for i, key in enumerate(group_by_list):
            if key not in index:
                index[key] = [i]
            else:
                index[key].append(i)
        return index.values()

    def _calcu_split_ids(self, tot, ratios):
        cnt = [int(ratios[i] * tot) for i in range(len(ratios))]
        cnt[0] = tot - sum(cnt[1:])
        for i in range(1, len(ratios)):
            if cnt[0] <= 1:
                break
            if 0 < ratios[-i] * tot < 1:
                cnt[-i] += 1
                cnt[0] -= 1
        split_ids = np.cumsum(cnt)[:-1]
        return list(split_ids)

    def split_by_ratio(self, ratios, group_by=None):
        self.logger.debug(f'split by ratios [{ratios}], group_by=[{group_by}]')
        tot_ratio = sum(ratios)
        ratios = [_ / tot_ratio for _ in ratios]

        if group_by is None:
            tot_cnt = self.__len__()
            split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
            next_index = [range(start, end) for start, end in zip([0] + split_ids, split_ids + [tot_cnt])]
        else:
            grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
            next_index = [[] for _ in range(len(ratios))]
            for grouped_index in grouped_inter_feat_index:
                tot_cnt = len(grouped_index)
                split_ids = self._calcu_split_ids(tot=tot_cnt, ratios=ratios)
                for index, start, end in zip(next_index, [0] + split_ids, split_ids + [tot_cnt]):
                    index.extend(grouped_index[start:end])

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def _split_index_by_leave_one_out(self, grouped_index, leave_one_num):
        next_index = [[] for _ in range(leave_one_num + 1)]
        for index in grouped_index:
            index = list(index)
            tot_cnt = len(index)
            legal_leave_one_num = min(leave_one_num, tot_cnt - 1)
            pr = tot_cnt - legal_leave_one_num
            next_index[0].extend(index[:pr])
            for i in range(legal_leave_one_num):
                next_index[-legal_leave_one_num + i].append(index[pr])
                pr += 1
        return next_index

    def leave_one_out(self, group_by, leave_one_mode):
        self.logger.debug(f'leave one out, group_by=[{group_by}], leave_one_mode=[{leave_one_mode}]')
        if group_by is None:
            raise ValueError('leave one out strategy require a group field')

        grouped_inter_feat_index = self._grouped_index(self.inter_feat[group_by].numpy())
        if leave_one_mode == 'valid_and_test':
            next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num=2)
        elif leave_one_mode == 'valid_only':
            next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num=1)
            next_index.append([])
        elif leave_one_mode == 'test_only':
            next_index = self._split_index_by_leave_one_out(grouped_inter_feat_index, leave_one_num=1)
            next_index = [next_index[0], [], next_index[1]]
        else:
            raise NotImplementedError(f'The leave_one_mode [{leave_one_mode}] has not been implemented.')

        self._drop_unused_col()
        next_df = [self.inter_feat[index] for index in next_index]
        next_ds = [self.copy(_) for _ in next_df]
        return next_ds

    def shuffle(self):
        self.inter_feat.shuffle()

    def sort(self, by, ascending=True):
        self.inter_feat.sort(by=by, ascending=ascending)

    def build(self):
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':
            self.shuffle()
        elif ordering_args == 'TO':
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

        split_args = self.config['eval_args']['split']
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if not isinstance(split_args, dict):
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config['eval_args']['group_by']
        if split_mode == 'RS':
            if not isinstance(split_args['RS'], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_ratio(split_args['RS'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        elif split_mode == 'LS':
            datasets = self.leave_one_out(group_by=self.uid_field, leave_one_mode=split_args['LS'])
        else:
            raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        return datasets

    def save(self):
        file = os.path.join(self.config['checkpoint_dir'], f'{self.config["dataset"]}-dataset.pth')
        self.logger.info(set_color('Saving filtered dataset into ', 'pink') + f'[{file}]')
        with open(file, 'wb') as f:
            pickle.dump(self, f)

    def get_user_feature(self):
        if self.user_feat is None:
            self._check_field('uid_field')
            return Interaction({self.uid_field: torch.arange(self.user_num)})
        else:
            return self.user_feat

    def get_item_feature(self):
        if self.item_feat is None:
            self._check_field('iid_field')
            return Interaction({self.iid_field: torch.arange(self.item_num)})
        else:
            return self.item_feat

    def _create_sparse_matrix(self, df_feat, source_field, target_field, form='coo', value_field=None):
        src = df_feat[source_field]
        tgt = df_feat[target_field]
        if value_field is None:
            data = np.ones(len(df_feat))
        else:
            if value_field not in df_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
            data = df_feat[value_field]
        mat = coo_matrix((data, (src, tgt)), shape=(self.num(source_field), self.num(target_field)))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def _create_graph(self, tensor_feat, source_field, target_field, form='dgl', value_field=None):
        src = tensor_feat[source_field]
        tgt = tensor_feat[target_field]

        if form == 'dgl':
            import dgl
            graph = dgl.graph((src, tgt))
            if value_field is not None:
                if isinstance(value_field, str):
                    value_field = {value_field}
                for k in value_field:
                    graph.edata[k] = tensor_feat[k]
            return graph
        elif form == 'pyg':
            from torch_geometric.data import Data
            edge_attr = tensor_feat[value_field] if value_field else None
            graph = Data(edge_index=torch.stack([src, tgt]), edge_attr=edge_attr)
            return graph
        else:
            raise NotImplementedError(f'Graph format [{form}] has not been implemented.')

    def inter_matrix(self, form='coo', value_field=None):
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')
        return self._create_sparse_matrix(self.inter_feat, self.uid_field, self.iid_field, form, value_field)

    def _history_matrix(self, row, value_field=None):
        self._check_field('uid_field', 'iid_field')

        user_ids, item_ids = self.inter_feat[self.uid_field].numpy(), self.inter_feat[self.iid_field].numpy()
        if value_field is None:
            values = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `inter_feat`\'s features.')
            values = self.inter_feat[value_field].numpy()

        if row == 'user':
            row_num, max_col_num = self.user_num, self.item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = self.item_num, self.user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        if col_num > max_col_num * 0.2:
            self.logger.warning(
                f'Max value of {row}\'s history interaction records has reached '
                f'{col_num / max_col_num * 100}% of the total.'
            )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)

    def history_item_matrix(self, value_field=None):
        return self._history_matrix(row='user', value_field=value_field)

    def history_user_matrix(self, value_field=None):
        return self._history_matrix(row='item', value_field=value_field)

    def get_preload_weight(self, field):
        if field not in self._preloaded_weight:
            raise ValueError(f'Field [{field}] not in preload_weight')
        return self._preloaded_weight[field]

    def _dataframe_to_interaction(self, data):
        new_data = {}
        for k in data:
            value = data[k].values
            ftype = self.field2type[k]
            if ftype == FeatureType.TOKEN:
                new_data[k] = torch.LongTensor(value)
            elif ftype == FeatureType.FLOAT:
                new_data[k] = torch.FloatTensor(value)
            elif ftype == FeatureType.TOKEN_SEQ:
                seq_data = [torch.LongTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
            elif ftype == FeatureType.FLOAT_SEQ:
                seq_data = [torch.FloatTensor(d[:self.field2seqlen[k]]) for d in value]
                new_data[k] = rnn_utils.pad_sequence(seq_data, batch_first=True)
        return Interaction(new_data)
