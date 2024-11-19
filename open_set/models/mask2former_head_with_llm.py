# A Mask2FormerHeadOpen module with an LLM
import copy
from typing import OrderedDict, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Dict, List, Optional, Set, Tuple, Union

import transformers
import clip

import mmcv
from mmcv.cnn import Conv2d, build_plugin_layer, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.ops import point_sample
from mmcv.runner import ModuleList, force_fp32, get_dist_info

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils import preprocess_panoptic_gt
from mmdet.models.utils import get_uncertain_point_coords_with_randomness
from mmdet.models.builder import HEADS, build_loss, build_head
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from .mask2former_head import Mask2FormerHeadOpen
from ..utils.eval.inference import beam_search, get_ids_embedding
from .utils.bert_embeddings import BertEmbeddings
from open_set.models.utils.bert_embeddings import BERT_MODEL_BY_EMBEDDING_TYPES
from .transformers.object_captioner import ObjectCaptioner


BOS_TOKEN = 101
EOS_TOKEN = 102

@HEADS.register_module()
class Mask2FormerHeadOpenWithLLM(Mask2FormerHeadOpen):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
        void_loss (str): Masks not matched to gts are seen as void masks.
            Defaults to 'void-background'.
            'void-background': train label of void masks as background.
            'void-thing': void as thing.
            'void-suppression': negative supervision to the void regions predicted as stuff.
        loss_caption_generation: A dictionary that defines the loss for caption generation.
    """

    def __init__(self,
                 in_channels,
                 feat_channels,
                 out_channels,
                 num_things_classes=80, # num_known_classes
                 num_stuff_classes=53,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 pixel_decoder=None,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 v2l_head=None,
                 caption_generator: Optional[Dict[str, Any]]=None,
                 loss_cls=None,
                 loss_cls_emb=None,
                 loss_grounding=None,
                 loss_caption_generation: Optional[Dict[str, Any]]=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels,
            num_things_classes=num_things_classes, # num_known_classes
            num_stuff_classes=num_stuff_classes,
            num_queries=num_queries,
            num_transformer_feat_level=num_transformer_feat_level,
            pixel_decoder=pixel_decoder,
            enforce_decoder_input_project=enforce_decoder_input_project,
            transformer_decoder=transformer_decoder,
            positional_encoding=positional_encoding,
            v2l_head=v2l_head,
            caption_generator=caption_generator,
            loss_cls=loss_cls,
            loss_cls_emb=loss_cls_emb,
            loss_grounding=loss_grounding,
            loss_caption_generation=loss_caption_generation,
            loss_mask=loss_mask,
            loss_dice=loss_dice,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            **kwargs
        )