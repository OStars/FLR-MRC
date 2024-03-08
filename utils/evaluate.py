from numpy.core.numeric import outer
import torch
import numpy as np
from utils.common import LabelSpan, LabelSpanWithScore
from collections import namedtuple, Counter
from torch import tensor


class MetricsCalculator4Ner():
    def __init__(self, args, processor) -> None:
        self.model_name = args.model_name
        self.processor = processor
        self.reset()

    def update(self, infer_starts, infer_ends, golden_labels, seq_lens, match_label_ids=None, tokens=None, match_pattern="default", is_logits=True):
        decoded_infers = self.processor.decode_label(infer_starts, infer_ends, seq_lens,
                                                     is_logits=is_logits, batch_match_label_ids=match_label_ids)[0]
        self.goldens.extend(golden_labels)
        self.infers.extend(decoded_infers)
        self.corrects.extend(
            [list(set(seq_infers) & set(seq_goldens)) for seq_infers,
                seq_goldens in zip(decoded_infers, golden_labels)]
        )
        if tokens != None:
            self.tokens.extend(tokens)

    def reset(self):
        self.goldens = []
        self.infers = []
        self.corrects = []
        self.tokens = []

    def get_metrics(self, metrics_type='micro_f1', info_type="default"):
        assert len(self.goldens) == len(self.infers) == len(self.corrects)
        infer_num = sum([len(items) for items in self.infers])
        golden_num = sum([len(items) for items in self.goldens])
        correct_num = sum([len(items) for items in self.corrects])
        precision, recall, f1 = self.calculate_f1(
            golden_num, infer_num, correct_num)
        metrics_info = {}
        metrics_info['general'] = {"precision": precision,
                                   "recall": recall, "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        if info_type == "detail":
            infer_counter = Counter(
                [item.label_id for items in self.infers for item in items])
            golden_counter = Counter(
                [item.label_id for items in self.goldens for item in items])
            correct_counter = Counter(
                [item.label_id for items in self.corrects for item in items])
            for label_id, golden_num in golden_counter.items():
                infer_num = infer_counter.get(label_id, 0)
                correct_num = correct_counter.get(label_id, 0)
                precision, recall, f1 = self.calculate_f1(
                    golden_num, infer_num, correct_num)
                metrics_info[self.processor.id2label[label_id]] = {"precision": precision, "recall": recall,
                                                                   "f1": f1, "infer_num": infer_num, "golden_num": golden_num, "correct_num": correct_num}
        return metrics_info

    def get_results(self, output_diff=True):
        assert len(self.tokens) == len(self.infers)
        return self.processor.build_output_results(self.tokens, self.infers, self.goldens if output_diff else None)

    def calculate_f1(self, label_num, infer_num, correct_num):
        """calcuate f1, precision, recall"""
        if infer_num == 0:
            precision = 0.0
        else:
            precision = correct_num * 1.0 / infer_num
        if label_num == 0:
            recall = 0.0
        else:
            recall = correct_num * 1.0 / label_num
        if correct_num == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1