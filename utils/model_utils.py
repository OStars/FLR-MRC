import torch
import math
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Classifier_Layer(nn.Module):
    def __init__(self, class_num, out_features, bias=True):
        super(Classifier_Layer, self).__init__()
        self.class_num = class_num
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(class_num, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(class_num))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # batch_size, seq_len = input.shape[:2]
        # input = input.contiguous().view(batch_size * seq_len,
        #                                 self.class_num, self.out_features)
        # x = input * self.weight  # [-1, class_num, dim]
        # print(input.shape)
        # print(self.weight.shape)
        x = torch.mul(input, self.weight)
        # (class_num, 1)
        x = torch.sum(x, -1)  # [-1, class_num]
        if self.bias is not None:
            x = x + self.bias
        # x = x.contiguous().view(batch_size, seq_len,
        #                         self.class_num)
        return x

    def extra_repr(self):
        return 'class_num={}, out_features={}, bias={}'.format(
            self.class_num, self.out_features, self.bias is not None)


class Label_Fusion_Layer_for_Token(nn.Module):
    def __init__(self, hidden_size, label_num, label_emb_size=300):
        super(Label_Fusion_Layer_for_Token, self).__init__()
        self.label_num = label_num
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)
        self.fc_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc_2 = nn.Linear(label_emb_size, self.hidden_size, bias=False)
        self.fc_5 = nn.Linear(self.hidden_size * 4, self.hidden_size)

    def forward(self, token_embs, label_embs, input_mask, label_input_mask=None, return_scores=False, use_attn=False, do_add=False, average_pooling=False):
        # [bs, seq_len, hidden_size]; [bs, label_num, 300]
        if use_attn:
            if average_pooling:
                return self.get_fused_feature_with_average_pooling(token_embs, label_embs, input_mask, label_input_mask)
            else:
                return self.get_fused_feature_with_biattn(token_embs, label_embs, input_mask, label_input_mask, return_scores=return_scores)
        elif do_add:
            return self.get_fused_feature_with_add(token_embs, label_embs, input_mask)
        else:
            return self.get_fused_feature_with_cos_weight(token_embs, label_embs, input_mask, return_scores=return_scores)

    def get_fused_feature_with_add(self, token_feature, label_feature, input_mask, return_scores=False):
        batch_size, seq_len = token_feature.shape[:2]
        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        label_feature = label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        token_feature = token_feature.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)

        fused_feature = token_feature + label_feature

        output = (fused_feature,)

        return output

    def get_fused_feature_with_cos_weight(self, token_feature, label_feature, input_mask, return_scores=False):
        batch_size, seq_len = token_feature.shape[:2]
        if len(token_feature.shape) != len(input_mask.shape):
            input_mask = input_mask.unsqueeze(-1)

        token_feature = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        token_feature_masked = token_feature * input_mask
        token_feature_norm = nn.functional.normalize(
            token_feature_masked, p=2, dim=-1)

        label_feature_t = label_feature.permute(
            1, 0)  # [hidden_dim, class_num]
        # label_feature_t = label_feature.transpose(0, 1)
        label_feature_t_norm = nn.functional.normalize(
            label_feature_t, p=2, dim=0)
        # [bs, seq_len, clas_num]
        scores = torch.matmul(token_feature_norm,
                              label_feature_t_norm).unsqueeze(-1)  # cosine-sim

        label_feature = label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, seq_len, 1, 1)
        token_feature = token_feature.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)
        weighted_label_feature = scores * label_feature
        fused_feature = token_feature + weighted_label_feature

        output = (fused_feature,)

        if return_scores:
            output += (scores.squeeze(-1),)
        return output

    def get_fused_feature_with_biattn(self, token_feature, label_feature, input_mask, label_input_mask,
                                    return_scores=False):
        """
            token_feature: [batch_size, context_seq_len, hidden_dim]
            label_feature: [class_num, label_seq_len, hidden_dim]
        """
        batch_size, context_seq_len = token_feature.shape[:2]
        class_num, class_seq_len = label_feature.shape[:2]
        token_feature_fc = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        # [hidden_dim, class_num, label_seq_len]
        label_feature_t = label_feature.permute(
            2, 0, 1).view(self.hidden_size, -1)

        scores = torch.matmul(token_feature_fc, label_feature_t).view(
            batch_size, context_seq_len, self.label_num, -1)  # [batch_size, context_seq_len, class_num, label_seq_len]

        extended_mask = label_input_mask[None, None, :, :]
        extended_mask = (1.0 - extended_mask) * -10000.0
        scores = scores + extended_mask
        a = torch.softmax(scores, dim=-1)

        # [bs, class_num, context_seq_len, label_seq_len] * (batch, class_num, label_seq_len, hidden_size) -> (batch, class_num, context_seq_len, hidden_size)
        c2q_att = torch.matmul(a.permute(0, 2, 1, 3), label_feature.unsqueeze(
            0).repeat(batch_size, 1, 1, 1))

        # [bs, class_num, context_seq_len]
        b_in = torch.max(scores.permute(0, 2, 1, 3), dim=3)[0]
        b_mask = input_mask[:, None, :]
        b_mask = (1.0 - b_mask) * -10000.0
        b_in = b_in + b_mask
        b = torch.softmax(b_in, dim=2).unsqueeze(2)
        # [bs, class_num, 1, context_seq_len] * (batch, calss_num, context_seq_len, hidden_size) -> (batch, class_num, 1, hidden_size)
        q2c_att = torch.matmul(b, token_feature_fc.unsqueeze(
            1).repeat(1, self.label_num, 1, 1))
        # (batch, class_num, context_seq_len, hidden_size)
        q2c_att = q2c_att.expand(-1, -1, context_seq_len, -1)

        # Origin-GAT: [bs, context_seq_len, class_num, hidden_dim * 4]
        fused_feature = torch.cat([token_feature_fc.unsqueeze(2).repeat(1, 1, self.label_num, 1),
                                   c2q_att.permute(0, 2, 1, 3),
                                   token_feature_fc.unsqueeze(2).repeat(1, 1, self.label_num, 1) * c2q_att.permute(0, 2, 1, 3),
                                   token_feature_fc.unsqueeze(2).repeat(1, 1, self.label_num, 1) * q2c_att.permute(0, 2, 1, 3)], dim=-1)

        fused_feature = torch.tanh(self.fc_5(fused_feature))
        output = (fused_feature,)

        if return_scores:
            # print(scores.shape)
            output += (scores.squeeze(-1),)
        return output


    def get_fused_feature_with_average_pooling(self, token_feature, label_feature, input_mask, label_input_mask, return_scores=False):
        """
            token_feature: [batch_size, context_seq_len, hidden_dim]
            label_feature: [class_num, label_seq_len, hidden_dim]
        """
        batch_size, context_seq_len = token_feature.shape[:2]

        token_feature_fc = self.fc_1(token_feature)
        label_feature = self.fc_2(label_feature)

        # [class_num, hidden_dim]
        averaged_label_feature = label_feature.mean(dim=1)

        # [bs, context_seq_len, class_num, hidden_dim]
        weighted_label_feature = averaged_label_feature.unsqueeze(
            0).unsqueeze(0).repeat(batch_size, context_seq_len, 1, 1)

        token_feature_fc = token_feature_fc.unsqueeze(
            2).repeat(1, 1, self.label_num, 1)

        # [bs, context_seq_len, class_num, hidden_dim]
        fused_feature = token_feature_fc + weighted_label_feature

        fused_feature = torch.tanh(self.fc_5(fused_feature))
        output = (fused_feature,)

        return output