# @Author: Wenwen Yu
# @Created Time: 7/7/2020 5:54 PM

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, roi_pool

from . import resnet


class Encoder(nn.Module):
    position_embedding: torch.Tensor

    def __init__(self,
                 char_embedding_dim: int = 512,
                 out_dim: int = 512,
                 image_feature_dim: int = 512,
                 nheaders: int = 4,
                 nlayers: int = 3,
                 feedforward_dim: int = 1024,
                 dropout: float = 0.1,
                 max_len: int = 100,
                 image_encoder: str = 'resnet50',
                 roi_pooling_mode: str = 'roi_align',
                 roi_pooling_size: Tuple[int, int] = (7, 7)):
        """
        convert image segments and text segments to node embedding.
        :param char_embedding_dim:
        :param out_dim:
        :param image_feature_dim:
        :param nheaders:
        :param nlayers:
        :param feedforward_dim:
        :param dropout:
        :param max_len:
        :param image_encoder:
        :param roi_pooling_mode:
        :param roi_pooling_size:
        """
        super().__init__()

        self.dropout = dropout
        assert roi_pooling_mode in {'roi_align', 'roi_pool'}, 'roi pooling model: {} not support.'.format(
            roi_pooling_mode)
        self.roi_pooling_mode = roi_pooling_mode
        assert roi_pooling_size and len(roi_pooling_size) == 2, 'roi_pooling_size not be set properly.'
        self.roi_pooling_size = (roi_pooling_size[0], roi_pooling_size[1])  # (h, w)

        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=char_embedding_dim,
                                                               nhead=nheaders,
                                                               dim_feedforward=feedforward_dim,
                                                               dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layers=nlayers)

        if image_encoder == 'resnet18':
            self.cnn = resnet.resnet18(output_channels=image_feature_dim)
        elif image_encoder == 'resnet34':
            self.cnn = resnet.resnet34(output_channels=image_feature_dim)
        elif image_encoder == 'resnet50':
            self.cnn = resnet.resnet50(output_channels=image_feature_dim)
        elif image_encoder == 'resnet101':
            self.cnn = resnet.resnet101(output_channels=image_feature_dim)
        elif image_encoder == 'resnet152':
            self.cnn = resnet.resnet152(output_channels=image_feature_dim)
        else:
            raise NotImplementedError()

        self.conv = nn.Conv2d(image_feature_dim, out_dim, self.roi_pooling_size)
        self.bn = nn.BatchNorm2d(out_dim)

        self.projection = nn.Linear(2 * out_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        # Compute the positional encodings once in log space.
        position_embedding = torch.zeros(max_len, char_embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, char_embedding_dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / char_embedding_dim))
        position_embedding[:, 0::2] = torch.sin(position * div_term)
        position_embedding[:, 1::2] = torch.cos(position * div_term)
        position_embedding = position_embedding.unsqueeze(0).unsqueeze(0)  # 1, 1, max_len, char_embedding_dim
        self.register_buffer('position_embedding', position_embedding)

        self.pe_dropout = nn.Dropout(self.dropout)
        self.output_dropout = nn.Dropout(self.dropout)

    def forward(self, images: torch.Tensor, boxes_coordinate: torch.Tensor, transcripts: torch.Tensor,
                src_key_padding_mask: torch.Tensor) -> torch.Tensor:
        """

        :param images: whole_images, shape is (B, C, H, W), where B is batch size, C is channel of images (default is 3),
                H is height of image, W is width of image.
        :param boxes_coordinate: boxes coordinate, shape is (B, N, 8),
                where 8 is coordinates (x1, y1, x2, y2, x3, y3, x4, y4).
        :param transcripts: text segments, shape is (B, N, T, D), where T is the max length of transcripts,
                                D is dimension of model.
        :param src_key_padding_mask: text padding mask, shape is (B * N, T), True for padding value.
            if provided, specified padding elements in the key will be ignored by the attention.
            This is an binary mask. When the value is True, the corresponding value on the attention layer of Transformer
            will be filled with -inf.
        :return: set of nodes X, shape is (B*N, T, D)
        """

        B, N, T, D = transcripts.shape

        # get image embedding using cnn
        # (B, 3, H, W)
        _, _, origin_H, _ = images.shape

        # image embedding: (B, C, H/16, W/16)
        images = self.cnn(images)
        _, _, H, _ = images.shape

        # generate rois for roi pooling, rois shape is (B, N, 5), 5 means (batch_index, x0, y0, x1, y1)
        rois_batch = torch.zeros(B, N, 5, device=images.device)
        for i in range(B):  # (B, N, 8)
            rois_batch[i, :, 0] = i
        rois_batch[:, :, 1:3] = boxes_coordinate[:, :, 0:2]
        rois_batch[:, :, 3:5] = boxes_coordinate[:, :, 4:6]

        spatial_scale = float(H / origin_H)
        # use roi pooling get image segments
        # (B*N, C, roi_pooling_size, roi_pooling_size)
        if self.roi_pooling_mode == 'roi_align':
            image_segments = roi_align(images, rois_batch.view(B * N, 5), self.roi_pooling_size, spatial_scale)
        else:
            image_segments = roi_pool(images, rois_batch.view(B * N, 5), self.roi_pooling_size, spatial_scale)

        # (B*N, D, 1, 1)
        image_segments = F.relu(self.bn(self.conv(image_segments)))
        # (B*N, D)
        image_segments = image_segments.squeeze()

        # (B*N, 1, D)
        image_segments = image_segments.unsqueeze(dim=1)

        # add positional embedding
        transcripts_segments = self.pe_dropout(transcripts + self.position_embedding[:, :, :T, :])
        # (B*N, T ,D)
        transcripts_segments = transcripts_segments.reshape(B * N, T, D)

        # (B*N, T, D)
        image_segments = image_segments.expand_as(transcripts_segments)

        # here we first add image embedding and text embedding together,
        # then as the input of transformer to get a non-local fusion features, different from paper process.
        out = image_segments + transcripts_segments

        # (T, B*N, D)
        out = out.transpose(0, 1).contiguous()

        # (T, B*N, D)
        out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)

        # (B*N, T, D)
        out = out.transpose(0, 1).contiguous()
        out = self.norm(out)
        out = self.output_dropout(out)

        return out
