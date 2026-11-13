from recbole.evaluator.register import Register
import torch
import copy


class DataStruct(object):
    def __init__(self):
        self._data_dict = {}

    def __getitem__(self, name: str):
        return self._data_dict[name]

    def __setitem__(self, name: str, value):
        self._data_dict[name] = value

    def __delitem__(self, name: str):
        self._data_dict.pop(name)

    def __contains__(self, key: str):
        return key in self._data_dict

    def get(self, name: str):
        if name not in self._data_dict:
            raise IndexError("Can not load the data without registration !")
        return self[name]

    def set(self, name: str, value):
        self._data_dict[name] = value

    def update_tensor(self, name: str, value: torch.Tensor):
        if name not in self._data_dict:
            self._data_dict[name] = value.cpu().clone().detach()
        else:
            if not isinstance(self._data_dict[name], torch.Tensor):
                raise ValueError("{} is not a tensor.".format(name))
            self._data_dict[name] = torch.cat((self._data_dict[name], value.cpu().clone().detach()), dim=0)

    def __str__(self):
        data_info = '\nContaining:\n'
        for data_key in self._data_dict.keys():
            data_info += data_key + '\n'
        return data_info


class Collector(object):
    def __init__(self, config):
        self.config = config
        self.data_struct = DataStruct()
        self.register = Register(config)
        self.full = ('full' in config['eval_args']['mode'])
        self.topk = self.config['topk']
        self.device = self.config['device']

    def data_collect(self, train_data):
        if self.register.need('data.num_items'):
            item_id = self.config['ITEM_ID_FIELD']
            self.data_struct.set('data.num_items', train_data.dataset.num(item_id))
        if self.register.need('data.num_users'):
            user_id = self.config['USER_ID_FIELD']
            self.data_struct.set('data.num_users', train_data.dataset.num(user_id))
        if self.register.need('data.count_items'):
            self.data_struct.set('data.count_items', train_data.dataset.item_counter)
        if self.register.need('data.count_users'):
            self.data_struct.set('data.count_items', train_data.dataset.user_counter)

    def _average_rank(self, scores):
        length, width = scores.shape
        true_tensor = torch.full((length, 1), True, dtype=torch.bool, device=self.device)

        obs = torch.cat([true_tensor, scores[:, 1:] != scores[:, :-1]], dim=1)
        bias = torch.arange(0, length, device=self.device).repeat(width).reshape(width, -1). \
            transpose(1, 0).reshape(-1)
        dense = obs.view(-1).cumsum(0) + bias

        count = torch.where(torch.cat([obs, true_tensor], dim=1))[1]
        avg_rank = .5 * (count[dense] + count[dense - 1] + 1).view(length, -1)

        return avg_rank

    def eval_batch_collect(
        self, scores_tensor: torch.Tensor, interaction, positive_u: torch.Tensor, positive_i: torch.Tensor
    ):
        if self.register.need('rec.items'):
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)
            self.data_struct.update_tensor('rec.items', topk_idx)

        if self.register.need('rec.topk'):
            _, topk_idx = torch.topk(scores_tensor, max(self.topk), dim=-1)
            pos_matrix = torch.zeros_like(scores_tensor, dtype=torch.int)
            pos_matrix[positive_u, positive_i] = 1
            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            pos_idx = torch.gather(pos_matrix, dim=1, index=topk_idx)
            result = torch.cat((pos_idx, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.topk', result)

        if self.register.need('rec.meanrank'):
            desc_scores, desc_index = torch.sort(scores_tensor, dim=-1, descending=True)

            pos_matrix = torch.zeros_like(scores_tensor)
            pos_matrix[positive_u, positive_i] = 1
            pos_index = torch.gather(pos_matrix, dim=1, index=desc_index)

            avg_rank = self._average_rank(desc_scores)
            pos_rank_sum = torch.where(pos_index == 1, avg_rank, torch.zeros_like(avg_rank)).sum(dim=-1, keepdim=True)

            pos_len_list = pos_matrix.sum(dim=1, keepdim=True)
            user_len_list = desc_scores.argmin(dim=1, keepdim=True)
            result = torch.cat((pos_rank_sum, user_len_list, pos_len_list), dim=1)
            self.data_struct.update_tensor('rec.meanrank', result)

        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', scores_tensor)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', interaction[self.label_field].to(self.device))

    def model_collect(self, model: torch.nn.Module):
        pass

    def eval_collect(self, eval_pred: torch.Tensor, data_label: torch.Tensor):
        if self.register.need('rec.score'):
            self.data_struct.update_tensor('rec.score', eval_pred)

        if self.register.need('data.label'):
            self.label_field = self.config['LABEL_FIELD']
            self.data_struct.update_tensor('data.label', data_label.to(self.device))

    def get_data_struct(self):
        returned_struct = copy.deepcopy(self.data_struct)
        for key in ['rec.topk', 'rec.meanrank', 'rec.score', 'rec.items', 'data.label']:
            if key in self.data_struct:
                del self.data_struct[key]
        return returned_struct
