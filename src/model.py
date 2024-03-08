import sys
sys.path.insert(0, 'utils/')

import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

import numpy as np
import utils.model_utils as model_utils
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        B, N = h.size()[0], h.size()[1]

        a_input = torch.cat([h.repeat(1, 1, N).view(B, N * N, -1), h.repeat(1, N, 1)], dim=2).view(B, N, -1,
                                                                                                   2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))

        # print("adj:", adj)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads, nlayers=2):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.nlayers = nlayers
        self.nheads = nheads
        self.attentions = [GraphAttentionLayer(
            nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                for j in range(self.nheads):
                    self.add_module('attention_{}_{}'.format(i + 1, j),
                                    GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=True))

        self.out_att = GraphAttentionLayer(
            nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        input = x
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        if self.nlayers > 2:
            for i in range(self.nlayers - 2):
                temp = []
                x = F.dropout(x, self.dropout, training=self.training)
                cur_input = x
                for j in range(self.nheads):
                    temp.append(self.__getattr__(
                        'attention_{}_{}'.format(i + 1, j))(x, adj))
                x = torch.cat(temp, dim=2) + cur_input
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x + input


def normalize(A, symmetric=True):
    # A = A+I
    A = A + torch.eye(A.size(0)).cuda()
    # 所有节点的度
    d = A.sum(1)
    if symmetric:
        # D = D^-1/2
        D = torch.diag(torch.pow(d, -0.5))
        return D.mm(A).mm(D)
    else:
        # D=D^-1
        D = torch.diag(torch.pow(d, -1))
        return D.mm(A)


class Class_Fusion(nn.Module):
    def __init__(self, hidden_size):
        super(Class_Fusion, self).__init__()
        self.gat = GAT(hidden_size, hidden_size,
                           hidden_size, 0.5, 0.2, 8, 2)
        self.linear_layer = nn.Linear(hidden_size*3, hidden_size)

    def normalize_adj(self, mx):
        """
        Row-normalize matrix  D^{-1}A
        torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
        """
        mx = mx.float()
        rowsum = mx.sum(2)
        r_inv = torch.pow(rowsum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag_embed(r_inv, 0)
        mx = r_mat_inv.matmul(mx)
        return mx

    def gene_adj(self):
        adj = torch.eye(33).cuda()
        for j in range(len(self.attmap.attmap)):
            for k in range(len(self.attmap.attmap[j])):
                if self.attmap.attmap[j][k] > 0:
                    adj[j, k] = 1.
        return adj

    def forward(self, label_embs_raw, label_mask, a_weight, b_weight):
        class_num, label_seq, hid = label_embs_raw.size()
        label_embs = label_embs_raw[:, 0]
        
        adj = torch.ones(class_num, class_num).unsqueeze(0)
        adj = self.normalize_adj(adj).cuda()

        graph_out_i = self.gat(label_embs.unsqueeze(0), adj)
        return graph_out_i.transpose(0, 1).repeat(1, label_seq, 1)


class FLRMRC(nn.Module):
    def __init__(self, args):
        super(FLRMRC, self).__init__()
        self.encoder_config = BertConfig.get_config_dict(
            args.model_name_or_path)[0]
        # self.encoder_config = BertConfig.from_pretrained(args.model_name_or_path)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.label_num = args.first_label_num
        self.use_attn = args.use_attn
        self.use_random_label_emb = args.use_random_label_emb
        self.average_pooling = args.average_pooling
        self.do_add = args.do_add
        self.use_label_embedding = args.use_label_embedding
        self.hidden_size = self.encoder_config['hidden_size']
        self.gradient_checkpointing = args.gradient_checkpointing

        # 模型内容
        self.tokenizer = BertTokenizerFast.from_pretrained(
            args.model_name_or_path, do_lower_case=args.do_lower_case)
        self.bert = BertModel.from_pretrained(args.model_name_or_path)
        self.entity_start_classifier = model_utils.Classifier_Layer(
            self.label_num, self.hidden_size)
        self.entity_end_classifier = model_utils.Classifier_Layer(
            self.label_num, self.hidden_size)
        self.label_fusing_layer = model_utils.Label_Fusion_Layer_for_Token(
            self.encoder_config["hidden_size"], self.label_num, 200 if self.use_label_embedding else self.hidden_size)
        if self.use_label_embedding:  # False
            self.label_ann_vocab_size = args.label_ann_vocab_size
            self.label_embedding_layer = nn.Embedding(
                args.label_ann_vocab_size, 200)
            glove_embs = torch.from_numpy(
                np.load(args.glove_label_emb_file, allow_pickle=True)).to(args.device)
            self.label_embedding_layer.weight.data.copy_(glove_embs)

        self.class_fuse = Class_Fusion(self.hidden_size)

    def forward(self, data, label_token_ids, label_token_type_ids, label_input_mask, add_label_info=True, return_score=False, mode='train', return_bert_attention=False):
        results = self.bert(input_ids=data['token_ids'], token_type_ids=data['token_type_ids'],
                            attention_mask=data['input_mask'], output_attentions=return_bert_attention)
        encoded_text = results[0]
        encoded_text = self.dropout(encoded_text)

        # batch_size, seq_len = encoded_text.shape[:2]

        # for i in label_token_ids.detach().cpu():
        #     print(self.tokenizer.convert_ids_to_tokens(i))
        #     print(self.tokenizer.decode(i))

        if self.use_label_embedding:  # False
            label_embs = self.label_embedding_layer(label_token_ids)
        elif self.use_random_label_emb:
            label_embs = data['random_label_emb']
        else:
            label_embs = self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[
                0] if self.use_attn else self.bert(input_ids=label_token_ids, token_type_ids=label_token_type_ids, attention_mask=label_input_mask)[1]

        if not add_label_info:
            # only stop gradient of current step. It will update according to history if adam used.
            label_embs = label_embs.detach()

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, return_scores=return_score, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)
            return custom_forward

        label_embs_new = self.class_fuse(label_embs, label_input_mask,
                                         self.entity_start_classifier.weight,
                                         self.entity_end_classifier.weight)

        if mode == 'train' and self.gradient_checkpointing:
            fused_results = torch.utils.checkpoint.checkpoint(create_custom_forward(
                self.label_fusing_layer), encoded_text, label_embs_new, data['input_mask'], label_input_mask)
        else:
            fused_results = self.label_fusing_layer(
                encoded_text, label_embs_new, data['input_mask'], label_input_mask=label_input_mask, return_scores=return_score, use_attn=self.use_attn, do_add=self.do_add, average_pooling=self.average_pooling)

        fused_feature = fused_results[0]
        # #self-att
        # fused_feature = self.class_fuse(fused_feature, data['input_mask'], label_embs)
        if mode == 'train' and self.gradient_checkpointing:
            # logits_start = torch.utils.checkpoint.checkpoint(
            #     self.entity_start_classifier, fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            # logits_end = torch.utils.checkpoint.checkpoint(
            #     self.entity_end_classifier, fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            logits_start = torch.utils.checkpoint.checkpoint(
                self.entity_start_classifier, fused_feature)
            logits_end = torch.utils.checkpoint.checkpoint(
                self.entity_end_classifier, fused_feature)
        else:
            # logits_start = self.entity_start_classifier(
            #     fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            # logits_end = self.entity_end_classifier(
            #     fused_feature).contiguous().view(batch_size, seq_len, self.label_num)
            logits_start = self.entity_start_classifier(fused_feature)
            logits_end = self.entity_end_classifier(fused_feature)
        # infer_start = torch.sigmoid(logits_start)
        # infer_end = torch.sigmoid(logits_end)
        output = (logits_start, logits_end)
        if return_score:
            output += (fused_results[-1],)
        if return_bert_attention:
            output += (results[-1],)

        return output