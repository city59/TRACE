import torch
from recbole.utils import EvaluatorType


class AbstractMetric(object):
    smaller = False

    def __init__(self, config):
        self.decimal_place = config['metric_decimal_place']

    def calculate_metric(self, dataobject):
        raise NotImplementedError('Method [calculate_metric] should be implemented.')


class TopkMetric(AbstractMetric):
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.topk']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        rec_mat = dataobject.get('rec.topk')
        topk_idx, pos_len_list = torch.split(rec_mat, [max(self.topk), 1], dim=1)
        return rec_mat.to(torch.bool).numpy(), pos_len_list.squeeze(-1).numpy()

    def topk_result(self, metric, value):
        metric_dict = {}
        avg_result = value.mean(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        raise NotImplementedError('Method [metric_info] of top-k metric should be implemented.')


class LossMetric(AbstractMetric):
    metric_type = EvaluatorType.VALUE
    metric_need = ['rec.score', 'data.label']

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        preds = dataobject.get('rec.score')
        trues = dataobject.get('data.label')

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        raise NotImplementedError('Method [metric_info] of loss-based metric should be implemented.')
