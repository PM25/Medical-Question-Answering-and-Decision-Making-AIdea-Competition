import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def get_pooler(input_size, config):
    module_name = config.pop("name")
    module_cfg = config[module_name]
    model = eval(module_name)(input_size=input_size, **module_cfg)
    return model


class BLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=256,
        num_layers=2,
        dropout=0.2,
        bidirectional=True,
        **kwargs
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    @property
    def output_size(self):
        return self.hidden_size

    def forward(self, x, x_len):
        packed_x = pack_padded_sequence(
            x, x_len.cpu(), batch_first=True, enforce_sorted=False
        )
        output, (h_n, c_n) = self.lstm(packed_x)
        diag_embeding = h_n.view(
            -1, self.num_layers, 2 if self.bidirectional else 1, self.hidden_size
        )
        diag_embeding = diag_embeding.mean(1).mean(1)
        return diag_embeding


class AttentivePooling(nn.Module):
    def __init__(self, input_size, activation="ReLU", **kwargs):
        super(AttentivePooling, self).__init__()
        self.input_size = input_size
        self.sap_layer = AttentivePoolingModule(input_size, activation)

    @property
    def output_size(self):
        return self.input_size

    def forward(self, feature_BxTxH, features_len):
        device = feature_BxTxH.device
        len_masks = torch.lt(
            torch.arange(features_len.max()).unsqueeze(0).to(device),
            features_len.unsqueeze(1),
        )
        sap_vec, _ = self.sap_layer(feature_BxTxH, len_masks)
        return sap_vec


class AttentivePoolingModule(nn.Module):
    def __init__(self, input_size, activation="ReLU", **kwargs):
        super(AttentivePoolingModule, self).__init__()
        self.W_a = nn.Linear(input_size, input_size)
        self.W = nn.Linear(input_size, 1)
        self.act_fn = getattr(nn, activation)()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep, att_w
