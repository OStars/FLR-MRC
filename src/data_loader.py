from torch.utils.data import DataLoader, Dataset
import csv
import json
import copy
import torch
import random
from transformers import BertTokenizer
import numpy as np
import pandas as pd
from tqdm import tqdm
from random import choice
import utils.common as common
from utils.tokenization import basic_tokenize, convert_to_unicode
from multiprocessing import Pool, cpu_count, Manager


class DataProcessor(object):
    """Base class for data converters for token classification data sets."""

    def get_train_examples(self, input_file):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, input_file):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_text(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        lines.append({"words": words, "labels": labels})
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                lines.append({"words": words, "labels": labels})
        return lines

    @classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines


class NerDataProcessor(DataProcessor):
    """Processor for the named entity recognition data set."""

    def __init__(self, args, tokenizer) -> None:
        super(NerDataProcessor, self).__init__()
        self.max_seq_length = args.max_seq_length
        self.model_name = args.model_name
        self.label_file = args.first_label_file
        self.tokenizer = tokenizer
        self.padding_to_max = args.padding_to_max
        self.label_str_file = args.label_str_file

        self.id2label, self.label2id = self.load_labels()
        self.class_num = len(self.id2label)
        self.is_chinese = args.is_chinese

        self.use_random_label_emb = args.use_random_label_emb
        self.use_label_embedding = args.use_label_embedding
        if self.use_label_embedding:
            self.label_ann_word_id_list_file = args.label_ann_word_id_list_file
            args.label_ann_vocab_size = len(self.load_json(args.label_ann_vocab_file))
        self.use_label_encoding = args.use_label_encoding
        self.label_list = args.label_list
        self.token_ids = None
        self.input_mask = None
        self.token_type_ids = None

    def get_train_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "train")

    def get_dev_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "dev")

    def get_test_examples(self, input_file):
        """See base class."""
        return self._create_examples(self._read_json(input_file), "test")

    def load_json(self, input_file):
        with open(input_file, 'r') as fr:
            loaded_data = json.load(fr)
        return loaded_data

    def load_labels(self):
        """See base class."""
        with open(self.label_file, 'r') as fr:
            id2label_, label2id_ = json.load(fr)
        id2label, label2id = {}, {}
        for key, value in id2label_.items():
            id2label[int(key)] = str(value)
        for key, value in label2id_.items():
            label2id[str(key)] = int(value)
        return id2label, label2id

    def get_tokenizer(self):
        return self.tokenizer

    def get_class_num(self):
        return self.class_num

    def get_label_data(self, device, rebuild=False):
        if rebuild:
            self.token_ids = None
            self.input_mask = None
            self.token_type_ids = None
        if self.token_ids is None:
            if self.use_label_embedding:
                token_ids, input_mask = [], []
                max_len = 0
                with open(self.label_ann_word_id_list_file, 'r') as fr:
                    for line in fr.readlines():
                        if line != '\n':
                            token_ids.append([int(item) for item in line.strip().split(' ')])
                            max_len = max(max_len, len(token_ids[-1]))
                            input_mask.append([1] * len(token_ids[-1]))
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
            else:
                if self.use_label_encoding:
                    with open(self.label_list, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                else:
                    with open(self.label_str_file, 'r', encoding='utf-8') as fr:
                        label_str_list = [line.strip() for line in fr.readlines()]
                token_ids, input_mask, max_len = [], [], 0
                for label_str in label_str_list:
                    encoded_results = self.tokenizer.encode_plus(label_str, add_special_tokens=True)
                    token_id = encoded_results['input_ids']
                    input_mask.append(encoded_results['attention_mask'])
                    max_len = max(max_len, len(token_id))
                    token_ids.append(token_id)
                assert max_len <= self.max_seq_length and len(token_ids) == self.class_num
                for idx in range(len(token_ids)):
                    padding_length = max_len - len(token_ids[idx])
                    token_ids[idx] += [0] * padding_length
                    input_mask[idx] += [0] * padding_length
                self.token_ids = torch.tensor(token_ids, dtype=torch.long).to(device)
                self.input_mask = torch.tensor(input_mask, dtype=torch.float32).to(device)
                self.token_type_ids = torch.zeros_like(self.token_ids, dtype=torch.long).to(device)
        return self.token_ids, self.token_type_ids, self.input_mask


    def encode_labels(self, entities, seq_len, offset_dict, tokens):
        first_label_start_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        first_label_end_ids = torch.zeros((seq_len, self.class_num), dtype=torch.float32)
        golden_labels, entity_cnt = [], 0

        for label in entities:
            label_start_offset = label["start_offset"]
            label_end_offset = label["end_offset"]
            try:
                start_idx = offset_dict[label_start_offset]
                end_idx = offset_dict[label_end_offset]
            except:
                # logger.warn(tokens)
                # logger.warn("{},{},{}".format(
                #     text[label_start_offset:label_end_offset+1], label_start_offset, label_end_offset))
                errmsg = "first_label '{}' doesn't exist in '{}'\n".format(label['text'], ' '.join(tokens))
                common.logger.warn(errmsg)
                continue
            if end_idx >= seq_len:
                continue
            if not self.is_chinese:
                assert ''.join(tokens[start_idx:end_idx+1]).replace("##",
                                                                    "").lower() == label['text'].lower().replace(" ", ""), "[error] {}\n{}\n".format(''.join(tokens[start_idx:end_idx+1]).replace("##", "").lower(), label['text'].lower().replace(" ", ""))
            entity_cnt += 1
            label_id = self.label2id[label['label']]
            golden_labels.append(common.LabelSpan(start_idx=start_idx, end_idx=end_idx, label_id=label_id))

            first_label_start_ids[start_idx][label_id] = 1
            first_label_end_ids[end_idx][label_id] = 1

        results = {
            'entity_starts': first_label_start_ids,
            'entity_ends': first_label_end_ids,
            'entity_cnt': entity_cnt,
            'golden_labels': golden_labels
        }

        return results
    
    def decode_label(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        return self.decode_label4span(batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids, is_logits)

    def decode_label4span(self, batch_start_ids, batch_end_ids, batch_seq_lens, batch_match_label_ids=None, is_logits=True):
        return self._extract_span(batch_start_ids, batch_end_ids, batch_seq_lens, is_logits=is_logits)
    
    def _extract_span(self, starts, ends, seqlens=None, position_dict=None, scores=None, is_logits=False, s_limit=0.5, e_limit=0.5, return_cnt=False, return_span_score=False):
        assert len(starts.shape) == 3 and len(ends.shape) == 3, "shape of 'starts' is {}, shape of 'ends' is {}".format(
            starts.shape, ends.shape)
        # print(seqlens)
        if is_logits:
            starts = torch.sigmoid(starts)
            ends = torch.sigmoid(ends)
        if return_span_score:
            assert scores is not None
            span_score_list = [[] for _ in range(starts.shape[0])]
        if seqlens is not None:
            assert starts.shape[0] == len(seqlens)
        if return_cnt:
            span_cnt = 0
        label_num = starts.shape[-1]
        span_list = [[] for _ in range(starts.shape[0])]

        for batch_idx in range(starts.shape[0]):
            for label_idx in range(label_num):

                cur_spans = []

                seq_start_labels = starts[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                                   ] if seqlens is not None else starts[batch_idx, :, label_idx]
                seq_end_labels = ends[batch_idx, :, label_idx][:seqlens[batch_idx]
                                                               ] if seqlens is not None else ends[batch_idx, :, label_idx]

                start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1
                for token_idx, (token_start_prob, token_end_prob) in enumerate(zip(seq_start_labels, seq_end_labels)):
                    if token_start_prob >= s_limit:
                        if end_idx != -1:  # build span
                            if return_span_score:
                                cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                           end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                            else:
                                cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                                  end_idx=end_idx, label_id=label_idx))
                            start_prob, start_idx, end_prob, end_idx, = -1, -1, -1, -1  # reset state
                        if token_start_prob > start_prob:  # new start, if pre prob is lower, drop it
                            start_prob = token_start_prob
                            start_idx = token_idx
                    if token_end_prob > e_limit and start_prob > s_limit:  # end
                        if token_end_prob > end_prob:
                            end_prob = token_end_prob
                            end_idx = token_idx
                if end_idx != -1:
                    if return_span_score:
                        cur_spans.append(common.LabelSpanWithScore(start_idx=start_idx,
                                                                   end_idx=end_idx, label_id=label_idx, start_score=scores[batch_idx, start_idx, label_idx], end_score=scores[batch_idx, end_idx, label_idx]))
                    else:
                        cur_spans.append(common.LabelSpan(start_idx=start_idx,
                                                          end_idx=end_idx, label_id=label_idx))
                cur_spans = list(set(cur_spans))
                if return_cnt:
                    span_cnt += len(cur_spans)
                if return_span_score:
                    span_score_list[batch_idx].extend(
                        [(item.start_score, item.end_score) for item in cur_spans])
                    span_list[batch_idx].extend([common.LabelSpan(
                        start_idx=item.start_idx, end_idx=item.end_idx, label_id=item.label_id) for item in cur_spans])
                else:
                    span_list[batch_idx].extend(cur_spans)
        output = (span_list,)
        if return_cnt:
            output += (span_cnt,)
        if return_span_score:
            output += (span_score_list,)
        return output

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line['id'] = guid
            examples.append(line)
        return examples

    @ classmethod
    def _read_json(self, input_file):
        lines = []
        with open(input_file, 'r') as f:
            for line in f.readlines():
                lines.append(json.loads(line.strip()))
        return lines

    def load_examples(self, input_file, data_type):
        examples = self._create_examples(self._read_json(input_file), data_type)
        return examples

    def build_offset_mapping(self, offsets, tokens):
        offset_dict = {}
        for token_idx in range(len(tokens)):
            # skip [cls] and [sep]
            if token_idx == 0 or token_idx == (len(tokens) - 1):
                continue
            token_start, token_end = offsets[token_idx]
            offset_dict[token_start] = token_idx
            offset_dict[token_end] = token_idx
        return offset_dict

    def convert_examples_to_feature(self, input_file, data_type):
        features = []
        stat_info = {'entity_cnt': 0, 'max_token_len': 0}
        examples = self.load_examples(input_file, data_type)
        sep_id = self.tokenizer.convert_tokens_to_ids('[SEP]')
        for example_idx, example in enumerate(examples):
            encoded_results = self.tokenizer.encode_plus(example['text'], add_special_tokens=True, return_offsets_mapping=True)
            token_ids = encoded_results['input_ids']
            token_type_ids = encoded_results['token_type_ids']
            input_mask = encoded_results['attention_mask']
            offsets = encoded_results['offset_mapping']
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            offset_dict = self.build_offset_mapping(offsets, tokens)

            stat_info['max_token_len'] = max(len(token_ids)-2, stat_info['max_token_len'])

            token_ids = token_ids[:self.max_seq_length]
            token_type_ids = token_type_ids[:self.max_seq_length]
            input_mask = input_mask[:self.max_seq_length]

            if token_ids[-1] != sep_id:
                assert len(token_ids) == self.max_seq_length
                token_ids[-1] = sep_id
            seq_len = len(token_ids)

            results = self.encode_labels(example['entities'], seq_len, offset_dict, tokens)
            stat_info['entity_cnt'] += results['entity_cnt']

            token_ids = torch.tensor(token_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            input_mask = torch.tensor(input_mask, dtype=torch.float32)

            assert len(token_ids) == len(input_mask) == len(token_type_ids) == len(results['entity_starts']) == len(results['entity_ends'])

            features.append(NerFeatures(example_id=example['id'],
                                        tokens_ids=token_ids,
                                        input_mask=input_mask,
                                        seq_len=seq_len,
                                        token_type_ids=token_type_ids,
                                        first_label_start_ids=results['entity_starts'],
                                        first_label_end_ids=results['entity_ends'],
                                        golden_label=results['golden_labels'],
                                        match_label=None))
        
        return {'features': features, "stat_info": stat_info}

    def _generate_batch(self, batch):
        batch_size, class_num = len(
            batch), batch[0].first_label_start_ids.shape[-1]

        batch_seq_len = torch.tensor([f.seq_len for f in batch], dtype=torch.long)
        ids = [f.example_id for f in batch]
        batch_golden_label = [f.golden_label for f in batch]
        max_len = int(max(batch_seq_len))

        batch_tokens_ids, batch_token_type_ids, batch_input_mask = torch.zeros((batch_size, max_len), dtype=torch.long), torch.zeros(
            (batch_size, max_len), dtype=torch.long), torch.zeros((batch_size, max_len), dtype=torch.float32)

        batch_first_label_start_ids, batch_first_label_end_ids = torch.zeros(
            (batch_size, max_len, class_num), dtype=torch.float32), torch.zeros((batch_size, max_len, class_num), dtype=torch.float32)

        for batch_idx in range(batch_size):
            batch_tokens_ids[batch_idx][:batch[batch_idx].tokens_ids.shape[0]
                                        ] = batch[batch_idx].tokens_ids
            batch_token_type_ids[batch_idx][:batch[batch_idx].token_type_ids.shape[0]
                                            ] = batch[batch_idx].token_type_ids
            batch_input_mask[batch_idx][:batch[batch_idx].input_mask.shape[0]
                                        ] = batch[batch_idx].input_mask
            batch_first_label_start_ids[batch_idx][:
                                                   batch[batch_idx].first_label_start_ids.shape[0]] = batch[batch_idx].first_label_start_ids
            batch_first_label_end_ids[batch_idx][:
                                                 batch[batch_idx].first_label_end_ids.shape[0]] = batch[batch_idx].first_label_end_ids

        results = {'token_ids': batch_tokens_ids,
                   'token_type_ids': batch_token_type_ids,
                   'input_mask': batch_input_mask,
                   'seq_len': batch_seq_len,
                   'first_starts': batch_first_label_start_ids,
                   'first_ends': batch_first_label_end_ids,
                   'ids': ids,
                   'golden_label': batch_golden_label,
                   }

        return results

    def generate_batch_data(self):
        return self._generate_batch


class NerFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, tokens_ids, input_mask, seq_len, token_type_ids,  first_label_start_ids=None, first_label_end_ids=None,
                 golden_label=None, match_label=None):
        self.example_id = example_id
        self.tokens_ids = tokens_ids
        self.input_mask = input_mask
        self.seq_len = seq_len
        self.token_type_ids = token_type_ids
        self.first_label_start_ids = first_label_start_ids
        self.first_label_end_ids = first_label_end_ids
        self.golden_label = golden_label
        self.match_label = match_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"