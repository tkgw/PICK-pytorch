# @Author: Wenwen Yu
# @Created Time: 7/8/2020 9:21 PM

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ..utils.class_utils import iob_labels_vocab_cls, keys_vocab_cls
from .crf import ConditionalRandomField

logger = logging.getLogger('PICK')


class MLPLayer(nn.Module):
    out_dim: int
    mlp: nn.Sequential

    def __init__(self,
                 in_dim: int,
                 out_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None,
                 layer_norm: bool = False,
                 dropout: Optional[float] = 0.0,
                 activation: Optional[str] = 'relu') -> None:
        """transform output of LSTM layer to logits, as input of crf layers
        :param in_dim:
        :param out_dim:
        :param hidden_dims:
        :param layer_norm:
        :param dropout:
        :param activation:
        """
        super().__init__()
        layers: List[nn.Module] = []
        activation_layer = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU
        }

        if hidden_dims:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(activation_layer.get(activation, nn.Identity()))
                logger.warning(
                    'Activation function {} is not supported, and replace with Identity layer.'.format(activation))
                if layer_norm:
                    layers.append(nn.LayerNorm(dim))
                if dropout:
                    layers.append(nn.Dropout(dropout))
                in_dim = dim

        if not out_dim:
            layers.append(nn.Identity())
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        self.mlp = nn.Sequential(*layers)
        self.out_dim = out_dim if out_dim else hidden_dims[-1]

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat(input, 1))


class BiLSTMLayer(nn.Module):

    def __init__(self, lstm_kwargs: Dict[str, Any], mlp_kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.lstm = nn.LSTM(**lstm_kwargs)
        self.mlp = MLPLayer(**mlp_kwargs)

    @staticmethod
    def sort_tensor(x: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sorted_length, sorted_order = torch.sort(length, descending=True)
        _, invert_order = sorted_order.sort(0, descending=False)
        return x[sorted_order], sorted_length, invert_order

    def forward(self, x_seq: torch.Tensor,
                lengths: torch.Tensor,
                initial: Tuple[torch.Tensor, torch.Tensor]) -> Tensor:
        """

        :param x_seq: (B, N*T, D)
        :param lengths: (B,)
        :param initial: (num_layers * directions, batch, D)
        :return: (B, N*T, out_dim)
        """

        # B*N, T, hidden_size
        x_seq, sorted_lengths, invert_order = self.sort_tensor(x_seq, lengths)
        packed_x = nn.utils.rnn.pack_padded_sequence(x_seq, lengths=sorted_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()  # ?
        output, _ = self.lstm(packed_x)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True,
                                                     padding_value=keys_vocab_cls.stoi['<pad>'])
        # total_length=documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN
        output = output[invert_order]
        logits = self.mlp(output)
        # (B, N*T, out_dim)
        return logits


class UnionLayer(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags: Tensor) -> Tuple:
        """
        For a document, we merge all non-paddding (valid) x and x_gcn value together in a document-level format,
        then feed it into crf layer.
        :param x: set of nodes, the output of encoder, (B, N, T, D)
        :param x_gcn: node embedding, the output of graph module, (B, N, D)
        :param mask: whether is non-padding (valid) value at i-th position of segments, (B, N, T)
        :param length: the length of every segments (boxes) of documents, (B, N)
        :param tags: IBO label for every segments of documents, (B, N, T)
        :return:
                new_x, (B, max_doc_seq_len, D)
                new_mask, (B, max_doc_seq_len)
                doc_seq_len, (B,)
                new_tag, (B, max_doc_seq_len)
        """
        B, N, T, D = x.shape
        x = x.reshape(B, N * T, D)
        mask = mask.reshape(B, N * T)

        # combine x and x_gcn firstly
        # (B, N, T, D)
        x_gcn = x_gcn.unsqueeze(2).expand(B, N, T, -1)
        # (B, N*T, D)
        x_gcn = x_gcn.reshape(B, N * T, D)
        # (B, N*T, D)
        x = x_gcn + x

        # (B, )
        doc_seq_len = length.sum(dim=-1)

        # dynamic calculate max document's union sequences length, only used to one gpus training mode.
        max_doc_seq_len = doc_seq_len.max()
        # static calculate max_doc_seq_len
        # max_doc_seq_len = documents.MAX_BOXES_NUM * documents.MAX_TRANSCRIPT_LEN

        # init x, mask, tags value
        # (B, N*T, D)
        new_x = torch.zeros_like(x, device=x.device)
        # (B, N*T)
        new_mask = torch.zeros_like(mask, device=x.device)
        if self.training:
            # (B, N*T)
            tags = tags.reshape(B, N * T)
            new_tag = torch.full_like(tags, iob_labels_vocab_cls.stoi['<pad>'], device=x.device)
            new_tag = new_tag[:, :max_doc_seq_len]

        # merge all non-padding value together in document-level
        for i in range(B):  # enumerate every document
            doc_x = x[i]  # (N*T, D)
            doc_mask = mask[i]  # (N*T,)
            valid_doc_x = doc_x[doc_mask == 1]  # (num_valid, D)
            num_valid = valid_doc_x.size(0)
            new_x[i, :num_valid] = valid_doc_x  # (B, N*T, D)
            new_mask[i, :doc_seq_len[i]] = 1  # (B, N*T)

            if self.training:
                valid_tag = tags[i][doc_mask == 1]
                new_tag[i, :num_valid] = valid_tag

        # (B, max_doc_seq_len, D)
        new_x = new_x[:, :max_doc_seq_len, :]
        # (B, max_doc_seq_len)
        new_mask = new_mask[:, :max_doc_seq_len]

        if self.training:
            return new_x, new_mask, doc_seq_len, new_tag
        else:
            return new_x, new_mask, doc_seq_len, None


class Decoder(nn.Module):

    def __init__(self, bilstm_kwargs: dict, mlp_kwargs: dict, crf_kwargs: dict) -> None:
        super().__init__()
        self.union_layer = UnionLayer()
        self.bilstm_layer = BiLSTMLayer(bilstm_kwargs, mlp_kwargs)
        self.crf_layer = ConditionalRandomField(**crf_kwargs)

    def forward(self, x: Tensor, x_gcn: Tensor, mask: Tensor, length: Tensor, tags: Tensor) -> Tuple:
        """

        :param x: set of nodes, the output of encoder, (B, N, T, D)
        :param x_gcn: node embedding, the output of graph module, (B, N, D)
        :param mask: whether is non-padding (valid) value at i-th position of segments, (B, N, T)
        :param length: the length of every segments (boxes) of documents, (B, N)
        :param tags: IBO label for every segments of documents, (B, N, T)
        :return:
        """
        # new_x: (B, max_doc_seq_len, D), new_mask: (B, max_doc_seq_len),
        # doc_seq_len: (B, ), new_tag: (B, N*T)
        new_x, new_mask, doc_seq_len, new_tag = self.union_layer(x, x_gcn, mask, length, tags)

        # (B, N*T, out_dim)
        logits = self.bilstm_layer(new_x, doc_seq_len, (None, None))

        log_likelihood = None
        if self.training:
            # (B,)
            log_likelihood = self.crf_layer(logits,
                                            new_tag,
                                            mask=new_mask,
                                            input_batch_first=True,
                                            keepdim=True)

        return logits, new_mask, log_likelihood
