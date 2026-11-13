import numpy as np
import pandas as pd
import torch
import torch.nn.utils.rnn as rnn_utils


def _convert_to_tensor(data):
    elem = data[0]
    if isinstance(elem, (float, int, np.float, np.int64)):
        new_data = torch.as_tensor(data)
    elif isinstance(elem, (list, tuple, pd.Series, np.ndarray, torch.Tensor)):
        seq_data = [torch.as_tensor(d) for d in data]
        new_data = rnn_utils.pad_sequence(seq_data, batch_first=True)
    else:
        raise ValueError(f'[{type(elem)}] is not supported!')
    if new_data.dtype == torch.float64:
        new_data = new_data.float()
    return new_data


class Interaction(object):
    def __init__(self, interaction):
        self.interaction = dict()
        if isinstance(interaction, dict):
            for key, value in interaction.items():
                if isinstance(value, (list, np.ndarray)):
                    self.interaction[key] = _convert_to_tensor(value)
                elif isinstance(value, torch.Tensor):
                    self.interaction[key] = value
                else:
                    raise ValueError(f'The type of {key}[{type(value)}] is not supported!')
        elif isinstance(interaction, pd.DataFrame):
            for key in interaction:
                value = interaction[key].values
                self.interaction[key] = _convert_to_tensor(value)
        else:
            raise ValueError(f'[{type(interaction)}] is not supported for initialize `Interaction`!')
        self.length = -1
        for k in self.interaction:
            self.length = max(self.length, self.interaction[k].shape[0])

    def __iter__(self):
        return self.interaction.__iter__()

    def __getattr__(self, item):
        if 'interaction' not in self.__dict__:
            raise AttributeError(f"'Interaction' object has no attribute 'interaction'")
        if item in self.interaction:
            return self.interaction[item]
        raise AttributeError(f"'Interaction' object has no attribute '{item}'")

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.interaction[index]
        else:
            ret = {}
            for k in self.interaction:
                ret[k] = self.interaction[k][index]
            return Interaction(ret)

    def __contains__(self, item):
        return item in self.interaction

    def __len__(self):
        return self.length

    def __str__(self):
        info = [f'The batch_size of interaction: {self.length}']
        for k in self.interaction:
            inter = self.interaction[k]
            temp_str = f"    {k}, {inter.shape}, {inter.device.type}, {inter.dtype}"
            info.append(temp_str)
        info.append('\n')
        return '\n'.join(info)

    def __repr__(self):
        return self.__str__()

    @property
    def columns(self):
        return list(self.interaction.keys())

    def to(self, device, selected_field=None):
        ret = {}
        if isinstance(selected_field, str):
            selected_field = [selected_field]

        if selected_field is not None:
            selected_field = set(selected_field)
            for k in self.interaction:
                if k in selected_field:
                    ret[k] = self.interaction[k].to(device)
                else:
                    ret[k] = self.interaction[k]
        else:
            for k in self.interaction:
                ret[k] = self.interaction[k].to(device)
        return Interaction(ret)

    def cpu(self):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].cpu()
        return Interaction(ret)

    def numpy(self):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].numpy()
        return ret

    def repeat(self, sizes):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat([sizes] + [1] * (len(self.interaction[k].shape) - 1))
        return Interaction(ret)

    def repeat_interleave(self, repeats, dim=0):
        ret = {}
        for k in self.interaction:
            ret[k] = self.interaction[k].repeat_interleave(repeats, dim=dim)
        return Interaction(ret)

    def update(self, new_inter):
        for k in new_inter.interaction:
            self.interaction[k] = new_inter.interaction[k]

    def drop(self, column):
        if column not in self.interaction:
            raise ValueError(f'Column [{column}] is not in [{self}].')
        del self.interaction[column]

    def _reindex(self, index):
        for k in self.interaction:
            self.interaction[k] = self.interaction[k][index]

    def shuffle(self):
        index = torch.randperm(self.length)
        self._reindex(index)

    def sort(self, by, ascending=True):
        if isinstance(by, str):
            if by not in self.interaction:
                raise ValueError(f'[{by}] is not exist in interaction [{self}].')
            by = [by]
        elif isinstance(by, (list, tuple)):
            for b in by:
                if b not in self.interaction:
                    raise ValueError(f'[{b}] is not exist in interaction [{self}].')
        else:
            raise TypeError(f'Wrong type of by [{by}].')

        if isinstance(ascending, bool):
            ascending = [ascending]
        elif isinstance(ascending, (list, tuple)):
            for a in ascending:
                if not isinstance(a, bool):
                    raise TypeError(f'Wrong type of ascending [{ascending}].')
        else:
            raise TypeError(f'Wrong type of ascending [{ascending}].')

        if len(by) != len(ascending):
            if len(ascending) == 1:
                ascending = ascending * len(by)
            else:
                raise ValueError(f'by [{by}] and ascending [{ascending}] should have same length.')

        for b, a in zip(by[::-1], ascending[::-1]):
            index = np.argsort(self.interaction[b], kind='stable')
            if not a:
                index = index[::-1]
            self._reindex(index)

    def add_prefix(self, prefix):
        self.interaction = {prefix + key: value for key, value in self.interaction.items()}


def cat_interactions(interactions):
    if not isinstance(interactions, (list, tuple)):
        raise TypeError(f'Interactions [{interactions}] should be list or tuple.')
    if len(interactions) == 0:
        raise ValueError(f'Interactions [{interactions}] should have some interactions.')

    columns_set = set(interactions[0].columns)
    for inter in interactions:
        if columns_set != set(inter.columns):
            raise ValueError(f'Interactions [{interactions}] should have some interactions.')

    new_inter = {col: torch.cat([inter[col] for inter in interactions]) for col in columns_set}
    return Interaction(new_inter)
