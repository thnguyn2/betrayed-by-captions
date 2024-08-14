# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import OrderedDict

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
from mmdet.models.dense_heads.maskformer_head import MaskFormerHead

from ..utils.eval.inference import beam_search, get_ids_embedding
from .utils.bert_embeddings import BertEmbeddings
from open_set.models.utils.bert_embeddings import BERT_MODEL_BY_EMBEDDING_TYPES


BOS_TOKEN = 101
EOS_TOKEN = 102

@HEADS.register_module()
class Mask2FormerHeadOpen(MaskFormerHead):
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
                 caption_generator=None,
                 loss_cls=None,
                 loss_cls_emb=None,
                 loss_grounding=None,
                 loss_caption_generation=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.\
            attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.transformerlayers.\
            attn_cfgs.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)
        self.pixel_decoder = build_plugin_layer(pixel_decoder_)[1]
        self.transformer_decoder = build_transformer_layer_sequence(
            transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = build_positional_encoding(
            positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))
        self.feat_channels = feat_channels

        self.v2l_head_cfg = v2l_head
        self.caption_generator_cfg = caption_generator

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self._loss_cls = build_loss(loss_cls)
        if loss_cls_emb is not None:
            self._loss_cls_emb = build_loss(loss_cls_emb)
        if loss_grounding is not None:
            self._loss_grounding = build_loss(loss_grounding)
        if loss_caption_generation is not None:
            self._loss_caption_generation = build_loss(loss_caption_generation)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)

        self.init_kwargs(**kwargs)

    def init_kwargs(self, **kwargs):
        self.kwargs = kwargs
        self.class_agnostic = kwargs.get('class_agnostic', False)
        self.use_class_emb = kwargs.get('use_class_emb', False)
        self._use_caption = kwargs.get('use_caption', False)
        self.use_caption_generation = kwargs.get('use_caption_generation', False)
        self.use_caption_align = kwargs.get('use_caption_align', False)
        self.known_file = kwargs.get('known_file', None)
        self.unknown_file = kwargs.get('unknown_file', None)
        self.softmax_temperature = kwargs.get('softmax_temperature', 10.0)
        self.learnable_temperature = kwargs.get('learnable_temperature', False)
        self.pred_emb_norm = kwargs.get('pred_emb_norm', False)
        self.text_emb_norm = kwargs.get('text_emb_norm', True)
        self.freeze_pretrained = kwargs.get('freeze_pretrained', False)
        self.freeze_v2l = kwargs.get('freeze_v2l', False)
        self.loss_only_last = kwargs.get('loss_only_last', False)
        self.loss_aux_weight = kwargs.get('loss_aux_weight', 1.0)
        self.gen_only_obj_nouns = kwargs.get('gen_only_obj_nouns', False)
        self.gen_mask_obj_nouns = kwargs.get('gen_mask_obj_nouns', False)
        self.gen_replace_obj_nouns = kwargs.get('gen_replace_obj_nouns', False)

        if self.known_file is not None:
            file_client = mmcv.FileClient()
            known_cat_names = set(file_client.get_text(self.known_file).split('\n'))
        else:
            known_cat_names = set()
            
        if self.unknown_file is not None:
            file_client = mmcv.FileClient()
            self._unknown_cat_names = set(file_client.get_text(self.unknown_file).split('\n'))
        else:
            self._unknown_cat_names = set()
        if self.use_class_emb:
            class_to_emb = {class_dict['name']: torch.FloatTensor(class_dict['emb']) for class_dict in mmcv.load(kwargs['class_to_emb_file'])}
            emb_dim = len(list(class_to_emb.values())[0])
            
            class_embs = torch.zeros((self.num_classes + 1, emb_dim), dtype=torch.float)
            known_class_idx = 0
            for name, emb in class_to_emb.items():
                if name not in known_cat_names:   # Only store embeddings of the non-classes
                    continue 
                if name in self._unknown_cat_names:
                    continue
                class_embs[known_class_idx, :] = emb
                known_class_idx += 1
                
            # automatically to cuda
            self.register_buffer('class_embs', class_embs)
            self.v2l_transform = nn.Linear(self.feat_channels, emb_dim)
        self.bert_embeddings = self.clip = None
        if self._use_caption:
            self.caption_emb_type = kwargs.get('caption_emb_type', 'clip')
            self._build_text_encoders(self.caption_emb_type)
        if self.use_caption_generation:
            self.caption_gen_emb_type = kwargs.get('caption_gen_emb_type', 'bert')
            self.caption_generator = build_head(self.caption_generator_cfg)
            self._build_text_encoders(self.caption_gen_emb_type, normalize_word_embeddings=self.text_emb_norm)
        if self.learnable_temperature:
            self.softmax_temperature = nn.Parameter(torch.tensor([self.softmax_temperature]), requires_grad=True)
            
        self.assigner.softmax_temperature = self.softmax_temperature

    @property
    def unknown_cat_names(self) -> Set[str]:
        return self._unknown_cat_names
    
    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

        if self.freeze_v2l:
            for p in self.v2l_transform.parameters():
                p.requires_grad = False
        
        if self.freeze_pretrained:
            self.freeze_params()

    def _build_text_encoders(self, emb_type: str, normalize_word_embeddings: bool=True) -> None:
        """Builds a text encoder.
        
        Args:
            emb_type: The type of embedding to use.
        """
        if emb_type in ('pubmed-bert', 'bert') and self.bert_embeddings is None:
            self.bert_embeddings = BertEmbeddings(
                bert_model=transformers.AutoModel.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[emb_type]).eval(),
            )
            for param in self.bert_embeddings.parameters():
                param.requires_grad = False
                
        if emb_type == 'clip' and self.clip is None:
            # clip_model, _ = clip.load('ViT-B/32')
            clip_model, _ = clip.load('RN50')
            self.clip = clip_model.eval()
            for param in self.clip.parameters():
                param.requires_grad = False

    def freeze_params(self):
        self.decoder_input_projs.eval()
        self.pixel_decoder.eval()
        self.transformer_decoder.eval()
        for p in self.decoder_input_projs.parameters():
            p.requires_grad = False
        for p in self.pixel_decoder.parameters():
            p.requires_grad = False
        for p in self.transformer_decoder.parameters():
            p.requires_grad = False

    def _get_targets_all_images_single_layer(
        self, 
        cls_scores_list: List[torch.Tensor], 
        cls_emb_preds_list: List[torch.Tensor], 
        mask_preds_list: List[torch.Tensor],
        gt_labels_list: List[torch.Tensor], 
        gt_masks_list: List[torch.Tensor], 
        caption_noun_emb_list: List[torch.Tensor],
        img_metas: List[Dict],
        ) -> Tuple[Union[torch.Tensor, int]]:
        """Computes classification and mask targets for all images for a decoder layer.

        Args:
            cls_scores_list: Mask score logits from a single decoder layer for all images. Each with shape (num_queries, cls_out_channels).
            cls_emb_preds_list: A  list of class embedding predictions for a single decoder layer for all images with shape (batch_size, num_queries, d_l).
                d_l is the dimension of embeddings.
            mask_preds_list: Mask logits from a single decoder layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list: Ground truth class indices for all images. Each with shape (n, ), n is the sum of number of stuff type and number of instance in a image.
                The index in this list ranges from [0, 1, ..., #known class - 1] that stores the label index from the original data. There is no background class here!
            gt_masks_list: Ground truth mask for each image, each with shape (n, h, w).
            unmasked_noun_emb_list: A list of unmasked noun embeddings. 1 item is for 1 sample, each item has a shape of (N_caption_nouns, embed_dim)
            img_metas: List of image meta information.

        Returns:
            labels_list: Labels of all images, shape (batch_size, num_queries).
            label_weights_list: Label weights of all images, shape (batch_size, num_queries).
            mask_targets_list: Mask targets of all images, shape (batch_size, num_queries, h, w).
            mask_weights_list: Mask weights of all images, shape (batch_size, num_queries).
                We use items in this list to manipulates the importance of the queries. Queries assigned by the Hungarian matching 
                with the groundtruth masks have the weights of 1. Queries assigned by the Hungarian matching to novel noun embeddings
                without masks have weights of 0. This behavior is important to avoid the downstream class embedding loss to modify the representation
                power of queries matched to novel noun embeddings.
            num_total_pos: Number of positive samples in all images, shape.
            num_total_neg: Number of negative samples in all images, shape.
        """ 
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single_image, 
            cls_scores_list, 
            cls_emb_preds_list, 
            mask_preds_list, 
            gt_labels_list, 
            gt_masks_list, 
            caption_noun_emb_list, 
            img_metas
        )
        return (
            torch.stack(labels_list, dim=0), 
            torch.stack(label_weights_list, dim=0), 
            torch.cat(mask_targets_list, dim=0),
            torch.stack(mask_weights_list, dim=0), 
            sum((inds.numel() for inds in pos_inds_list)),
            sum((inds.numel() for inds in neg_inds_list)),
        )

    def _get_target_single_image(self, cls_score: torch.Tensor, cls_emb_pred: torch.Tensor, mask_pred: torch.Tensor,
                        gt_labels: torch.Tensor, gt_masks: torch.Tensor, caption_noun_emb: torch.Tensor, img_metas: Dict) -> Tuple[torch.Tensor]:
        """Compute classification and mask targets for one image.

        Args:
            cls_score: Mask score logits from a single decoder layer for one image. Shape (num_queries, cls_out_channels).
            cls_emb_pred: Embedding prediction of all queries for a single decoder layer for one image with shape (num_queries, embd_dim).
            mask_pred: Mask logits for a single decoder layer for one image. Shape (num_queries, h, w).
            gt_labels: Ground truth class indices for one image with shape (num_gts, ).
            gt_masks: Ground truth mask for each image, each with shape (num_gts, h, w).  
            caption_noun_emb: The caption noun embeddings of shape (num caption nouns, embd_dim)
            img_metas: Image information.

        Returns:
            labels: Labels of each image of shape (num_queries, ).
            label_weights: Label weights of each image of shape (num_queries,).  We use items in this list to manipulates the importance of the queries. 
                Queries assigned by the Hungarian matching with the to foreground classes have the weights of 1. Otherwise, it is assigned to 0. 
                This is important to tell the model not to adjust the embeddings of queries that were matched to a novel caption noun emb when optimizing the 
                class embedding loss of known classes. The behavior is
                    Assign 1 to queries that were matched to a groundtruth class or a background class
                    Assign 0 to queries that were matched to a novel class in the noun embeddings.
            mask_targets: Mask targets of each image of shape (num_queries, h, w).
            mask_weights: Mask weights of each image of shape (num_queries, ). We use items in this list to manipulates the importance of the queries. 
                Queries assigned by the Hungarian matching with the groundtruth masks have the weights of 1. Queries assigned by the Hungarian matching to 
                novel noun embeddings without masks have weights of 0. This behavior is important to avoid the downstream class embedding loss to modify the 
                representation power of queries matched to novel noun embeddings.
            pos_inds: Sampled positive indices for each image.
            neg_inds: Sampled negative indices for each image.
        """
        
        # sample points
        num_queries = cls_score.shape[0]
        point_coords = torch.rand((1, self.num_points, 2), device=cls_score.device)
        
        # assign and sample    
        sampling_result = self.sampler.sample(
            self.assigner.assign(
                cls_score, 
                cls_emb_pred=cls_emb_pred,
                known_cls_emb=self.class_embs,
                mask_pred=point_sample(mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1)).squeeze(1),  # shape (num_queries, num_points)
                gt_labels=gt_labels,
                gt_mask=point_sample(gt_masks.unsqueeze(1).float(), point_coords.repeat(gt_labels.shape[0], 1, 1)).squeeze(1),  # shape (num_gts, num_points)
                img_meta=img_metas,
            ), 
            mask_pred, 
            gt_masks
        )
        
        pos_inds = sampling_result.pos_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = 1.0

        label_weights = gt_labels.new_ones((self.num_queries,))

        # Avoid the class embedding loss/dice to update the the queries matched to novel classes by setting the label weights to 0.
        novel_caption_noun_query_inds = self.assigner.find_novel_class_query_indices(
            cls_emb_pred=cls_emb_pred,
            known_cls_emb=self.class_embs,
            caption_noun_emb=caption_noun_emb,
            query_indices_to_avoid=pos_inds,
        )
        
        if novel_caption_noun_query_inds is not None:
            label_weights[novel_caption_noun_query_inds] = 0.0
    
        
        return (
            labels, 
            label_weights, 
            gt_masks[sampling_result.pos_assigned_gt_inds], 
            mask_weights, 
            pos_inds, 
            sampling_result.neg_inds)

    @force_fp32(apply_to=('all_layer_cls_scores', 'all_layer_mask_preds'))
    def loss(
        self, 
        all_layer_cls_scores: List[torch.Tensor], 
        all_cls_emb_preds: List[torch.Tensor], 
        all_layer_mask_preds: List[torch.Tensor], 
        gt_labels_list: List[Optional[torch.Tensor]], 
        gt_masks_list: List[Optional[torch.Tensor]], 
        gt_caption_ids_list: List[torch.Tensor],
        gt_caption_embs_list: List[torch.Tensor], 
        gt_caption_mask_list: List[torch.Tensor],
        gt_caption_nouns_ids_list: List[torch.Tensor], 
        gt_caption_avg_pooled_nouns_embs_list: List[torch.Tensor], 
        gt_caption_avg_pooled_nouns_mask_list: List[torch.Tensor], 
        img_metas,
        ) -> Dict[str, float]:
        """Loss function.

        Args:
            all_layer_cls_scores: A list of classification scores for all decoder
                layers. Each is a tensor with shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes background.
            all_cls_emb_preds: A list of class embedding prediction for all decoder
                layers. Each is a tensor with shape (batch_size, num_queries, d_l). d_l is the dimension of embeddings.
            all_layer_mask_preds: A list of mask prediction logits for all decoder layers with
                shape (batch_size, num_queries, h, w).
            gt_labels_list: A list of ground truth class indice arrays for each sample in the minibatch. Each array has shape (n, ) with
                n is the sum of number of stuff type and number of instance in a image.
            gt_masks_list: A list of groundtruth mask arrays for different samples in the minibatch. Each ground truth mask array has shape (n, h, w).
            gt_caption_ids_list: A list of caption id arrays for different samples in the minibatch. Each array has length of (max_token,).
            gt_caption_embs_list: A list of caption embedding arrays for different samples in the minibatch. Each array has shape (max_token, d_l) 
            gt_caption_mask_list: A list of caption id mask arrays for different samples in the minibatch. Each array has a length of (max_token,).
            gt_caption_nouns_ids_list: A list of caption now ids.
            gt_caption_avg_pooled_nouns_embs_list:
            gt_caption_avg_pooled_nouns_mask_list: 
            img_metas (list[dict]): List of image meta information.

        Returns:
            A dictionary of loss components.
        """
        num_dec_layers = len(all_layer_cls_scores)

        losses_cls, losses_cls_emb, losses_grounding, losses_caption_generation, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_layer_cls_scores, all_cls_emb_preds, all_layer_mask_preds,
            [gt_labels_list for _ in range(num_dec_layers)], 
            [gt_masks_list for _ in range(num_dec_layers)], 
            [gt_caption_ids_list for _ in range(num_dec_layers)],
            [gt_caption_embs_list for _ in range(num_dec_layers)], 
            [gt_caption_mask_list for _ in range(num_dec_layers)],
            [gt_caption_nouns_ids_list for _ in range(num_dec_layers)], 
            [gt_caption_avg_pooled_nouns_embs_list for _ in range(num_dec_layers)], 
            [gt_caption_avg_pooled_nouns_mask_list for _ in range(num_dec_layers)], 
            [img_metas for _ in range(num_dec_layers)]
        )

        loss_dict = {
            'loss_cls': losses_cls[-1],
            'loss_cls_emb': losses_cls_emb[-1],
            'loss_grounding': losses_grounding[-1],
            'loss_caption_generation': losses_caption_generation[-1],
            'loss_mask': losses_mask[-1],
            'loss_dice': losses_dice[-1],
        }
        
        if self.loss_only_last:
            return loss_dict
        
        # loss from other decoder layers
        for decoder_layer_idx, (loss_cls_i, loss_cls_emb_i, loss_grounding_i, losses_caption_generation_i, loss_mask_i, loss_dice_i) in enumerate(zip(
                losses_cls[:-1], losses_cls_emb[:-1], losses_grounding[:-1], losses_caption_generation[:-1], losses_mask[:-1], losses_dice[:-1])):
            loss_dict[f'd{decoder_layer_idx}.loss_cls'] = loss_cls_i * self.loss_aux_weight
            loss_dict[f'd{decoder_layer_idx}.loss_cls_emb'] = loss_cls_emb_i * self.loss_aux_weight
            loss_dict[f'd{decoder_layer_idx}.loss_grounding'] = loss_grounding_i * self.loss_aux_weight
            loss_dict[f'd{decoder_layer_idx}.loss_caption_generation'] = losses_caption_generation_i * self.loss_aux_weight
            loss_dict[f'd{decoder_layer_idx}.loss_mask'] = loss_mask_i * self.loss_aux_weight
            loss_dict[f'd{decoder_layer_idx}.loss_dice'] = loss_dice_i * self.loss_aux_weight
        return loss_dict
    
    def loss_single(
        self, 
        cls_scores: torch.Tensor, 
        cls_emb_preds: torch.Tensor, 
        mask_preds: torch.Tensor,
        gt_labels_list: List[torch.Tensor], 
        gt_masks_list: List[torch.Tensor],
        gt_caption_ids_list: List[torch.Tensor], 
        gt_caption_embs_list: List[torch.Tensor], 
        gt_caption_mask_list: List[torch.Tensor],
        gt_caption_nouns_ids_list: List[torch.Tensor], 
        gt_caption_avg_pooled_nouns_embs_list: List[torch.Tensor], 
        gt_caption_avg_pooled_nouns_mask_list: List[torch.Tensor], 
        img_metas: List[Dict],
        ) -> Tuple[torch.Tensor]:
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores: Mask score logits from a single decoderlayer for all images. Shape (batch_size, num_queries, cls_out_channels). 
                Note `cls_out_channels` should includes background.
            cls_emb_preds: Embedding prediction for a single decoder layer for all images with shape (batch_size, num_queries, d_l).
                d_l is the dimension of embeddings.
            mask_preds: Mask logits for a pixel decoder for all images. Shape (batch_size, num_queries, h, w).
            gt_labels_list: Ground truth class indices for each image, each with shape (num_gts, ).
            gt_masks_list: Ground truth mask for each image, each with shape (num_gts, h, w).
            gt_caption_ids_list: (max_token,)
            gt_caption_avg_pooled_nouns_embs_list: (max_token, d_l)
            gt_caption_avg_pooled_nouns_mask_list: (max_token,)
            img_metas: List of image meta information.

        Returns:
            Loss components for outputs from a single decoder layer.
        """
        # Debug only
        # all_noun_token_ids = torch.cat(gt_caption_nouns_ids_list)
        # if 10777 in all_noun_token_ids:
        #     print("Here!")
        
        
        (labels, label_weights, mask_targets, mask_weights, num_total_pos, _) = self._get_targets_all_images_single_layer(
            [x for x in cls_scores], 
            [x for x in cls_emb_preds] if self.use_class_emb else [None] * cls_scores.size(0),
            [x for x in mask_preds],
            gt_labels_list=gt_labels_list, 
            gt_masks_list=gt_masks_list,
            caption_noun_emb_list=[noun_emb[mask.bool()] for noun_emb, mask in zip(gt_caption_avg_pooled_nouns_embs_list, gt_caption_avg_pooled_nouns_mask_list)],
            img_metas=img_metas)
        
        # Classfication loss
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)  # (batch_size * num_queries, )
        label_weights = label_weights.flatten(0, 1)  # (batch_size * num_queries, )
        class_weight = cls_scores.new_tensor(self.class_weight)  # (batch_size * num_queries, )
        loss_cls = self._loss_cls(cls_scores, labels, label_weights, avg_factor=class_weight[labels].sum())

        # Embedding prediction loss
        loss_cls_emb = loss_cls.new_tensor(0.0)
        if self.use_class_emb:
            loss_cls_emb = self._loss_cls_emb(self._get_cls_emb_logits_from_cls_emb_pred(cls_emb_preds).flatten(0, 1), labels, label_weights.float(), avg_factor=class_weight[labels].sum())

        # Caption grounding loss
        loss_grounding = loss_cls.new_tensor(0.0)
        if self._use_caption:
            all_gt_caption_nouns_embs, all_gt_caption_nouns_mask, all_cls_emb_preds = \
                self._gather_captions_and_preds(gt_caption_avg_pooled_nouns_embs_list, gt_caption_avg_pooled_nouns_mask_list, cls_emb_preds)
            loss_grounding = self._loss_grounding(all_cls_emb_preds, all_gt_caption_nouns_embs, all_gt_caption_nouns_mask, self.softmax_temperature)

        # Caption generation loss
        loss_caption_generation = loss_cls.new_tensor(0.0)
        if self.use_caption_generation:
            caption_logits = self.caption_generator(
                tgt=torch.stack(gt_caption_embs_list, dim=0)[:, :-1, :], 
                memory=cls_emb_preds,
                tgt_key_padding_mask=torch.logical_not(torch.stack(gt_caption_mask_list, dim=0).bool()[:, :-1]))[1]
            
            caption_logits = caption_logits.flatten(0, 1)  #(batch_size * (max_tokens - 1), vocab_size)
            for i in range(len(gt_caption_ids_list)):
                gt_caption_ids =  gt_caption_ids_list[i]
                gt_caption_nouns_ids = gt_caption_nouns_ids_list[i].cpu().numpy().tolist()
                for j in range(len(gt_caption_ids)):
                    if int(gt_caption_ids[j]) not in gt_caption_nouns_ids:
                        if self.gen_only_obj_nouns:
                            gt_caption_ids[j] = 0  # set gt to 0 except for obj nouns
                    elif int(gt_caption_ids[j]) in gt_caption_nouns_ids:
                        if self.gen_mask_obj_nouns:
                            gt_caption_ids[j] = 0  # set 0 to one object noun (the first one seen in the caption)
                            break
                        if self.gen_replace_obj_nouns:
                            gt_caption_ids[j] = 4874    # 'object'
            gt_caption_ids = torch.stack(gt_caption_ids_list, dim=0)[:, 1:].flatten(0, 1)  # (batch_size * (max_tokens - 1))
            loss_caption_generation = self._loss_caption_generation(caption_logits, gt_caption_ids)
            
        num_total_masks = max(reduce_mean(cls_scores.new_tensor([num_total_pos])), 1)

        # extract positive ones
        mask_preds = mask_preds[mask_weights > 0]  # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)

        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_cls_emb, loss_grounding, loss_caption_generation, loss_mask, loss_dice

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(mask_preds.unsqueeze(1), None, self.num_points, self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(mask_targets.unsqueeze(1).float(), points_coords).squeeze(1)  # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
        
        mask_point_preds = point_sample(mask_preds.unsqueeze(1), points_coords).squeeze(1)  # shape (num_queries, h, w) -> (num_queries, num_points)
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, avg_factor=num_total_masks)

        # mask loss
        mask_point_preds = mask_point_preds.reshape(-1)  # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)  # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        loss_mask = self.loss_mask(mask_point_preds, mask_point_targets, avg_factor=num_total_masks * self.num_points)

        return loss_cls, loss_cls_emb, loss_grounding, loss_caption_generation, loss_mask, loss_dice 
        
    def _get_cls_emb_logits_from_cls_emb_pred(self, cls_emb_preds: torch.Tensor) -> torch.Tensor:
        """Compute prediction logits for embedding predicion head. 

        The output will be <cls_emb_preds, self.class_embs> / temperature.
        Args:
            cls_emb_preds: A tensor of (batch_size, num_queries, d_l) that stores class embedding prediction for a single decoder for all images.
            
        Returns:
            cls_emb_logits: Embedding predicion scores with shape of (batch_size, num_queries, self.num_classes + 1).
        """
        # (batch_size, num_queries, d_l) * (d_l, self.num_classes) -> (batch_size, num_queries, self.num_classes)
        return torch.matmul(cls_emb_preds, self.class_embs.t()) / self.softmax_temperature

    def _gather_captions_and_preds(self, gt_caption_embs_list: List[torch.Tensor], gt_caption_mask_list: List[torch.Tensor], cls_emb_preds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather all caption annotations from the whole batch using dist.all_gather. 

        Args:
            gt_caption_embs_list: (max_token, d_l)
            gt_caption_mask_list: (max_token)
            cls_emb_preds: (batch_size, num_queries, d_l).
            
        Returns:
            all_gt_caption_embs: (batch_size * world_size, max_tokens, d_l)
            all_gt_caption_mask: (batch_size * world_size, max_tokens).
            all_cls_emb_preds: (batch_size * world_size, num_queries, d_l): The predicted class embeddings from all the workers.     
        """
        batch_size = len(gt_caption_embs_list)
        rank, world_size = get_dist_info()
        if world_size > 1:
            gt_caption_embs = torch.stack(gt_caption_embs_list, dim=0)
            gt_caption_mask = torch.stack(gt_caption_mask_list, dim=0)
            emb_tmp_list = [gt_caption_embs.new_zeros(size=gt_caption_embs.size()) for i in range(world_size)]
            mask_tmp_list = [gt_caption_mask.new_zeros(size=gt_caption_mask.size()) for i in range(world_size)]
            pred_tmp_list = [cls_emb_preds.new_zeros(size=cls_emb_preds.size()) for i in range(world_size)]
        
            dist.all_gather(emb_tmp_list, gt_caption_embs)
            dist.all_gather(mask_tmp_list, gt_caption_mask)
            dist.all_gather(pred_tmp_list, cls_emb_preds)

            all_cls_emb_preds = torch.cat(pred_tmp_list, dim=0)
            all_cls_emb_preds[rank * batch_size : (rank + 1) * batch_size] = cls_emb_preds
            return torch.cat(emb_tmp_list, dim=0), torch.cat(mask_tmp_list, dim=0), all_cls_emb_preds
        else:
            return torch.stack(gt_caption_embs_list, dim=0), torch.stack(gt_caption_mask_list, dim=0), cls_emb_preds

    def _extract_word_embeddings(self, ids_list: List[torch.Tensor], mask_list: List[torch.Tensor], emb_type: str='bert') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Extract caption words' embeddings and masks
        
        Args:
            ids_list: A list of token ids for all captions in the minibatch. Each item of the list is for 1 sample of shapes (N_tokens,).
            mask_list: A list of token masks for all captions in the minibatch. One item of the list is for 1 sample of shapes (N_tokens,).
        
        Returns:
            A list of token embeddings, one item for 1 sample of shape (N_token, Emb_dim).
            A list of token makes, one item for 1 sample of shape (N_token,).
        """
        embs_list = []
        valid_mask_list = []
        if emb_type in ('bert', 'pubmed-bert'):
            for i, ids in enumerate(ids_list):
                embs = self.bert_embeddings.calculate_word_embeddings(ids)
                embs_list.append(embs)
                valid_mask_list.append(mask_list[i])
        elif emb_type == 'clip':
            for i, ids in enumerate(ids_list):
                nouns_embs = self.clip.encode_text(ids).float()
                if self.text_emb_norm:
                    nouns_embs /= nouns_embs.norm(dim=-1, keepdim=True)
                embs = nouns_embs.new_zeros((ids.shape[1], nouns_embs.shape[-1]))
                embs[:nouns_embs.shape[0], :] = nouns_embs
                embs_list.append(embs)
                valid_mask_list.append(mask_list[i])

        return embs_list, valid_mask_list
    
    def _extract_noun_word_embeddings(self,  ids_list: List[torch.Tensor], mask_list: List[torch.Tensor], token_noun_indices: List[torch.Tensor], emb_type: str='bert') -> torch.Tensor:
        """Calculates the work embeddings of tokens ids.
        
        Args:
            ids_list: A list of token ids for all captions in the minibatch. Each item of the list is for 1 sample of shapes (N_tokens,).
            mask_list: A list of token masks for all captions in the minibatch. One item of the list is for 1 sample of shapes (N_tokens,).
            token_noun_indices: A tensor of shape (Ntokens,) that specifies the noun indices of the token.
            emb_type: The type of embeddings, defaults to `bert`.
            
        Returns:
            A tensor of shape (N_reduced_token, emb) that contains the (avg pooled) embeddings of differnt nouns.
            A tensor of shape (N_reduced_token,) that contains the (avg pooled) embedding masks of differnt nouns.
            
        Note that a noun can be converted to many tokens. Similar to a sentence.
        For example, the word `girrafe` can be tokenize into [21025, 11335, 7959].
        The embeddings saved in the file embeddings/coco_class_with_bert_emb.json are 
        average pooled across tokens of a noun. 
        Hence, we need to make sure that we need to properly account for this in the grounding loss.    
        """
        embs_list = []
        valid_mask_list = []
        if emb_type in ('bert', 'pubmed-bert'):
            for i, (ids, noun_index) in enumerate(zip(ids_list, token_noun_indices)):
                embs, mask = self.bert_embeddings.calculate_mean_pool_embeddings(ids=ids, noun_idx=noun_index.clone())
                embs_list.append(embs)
                valid_mask_list.append(mask)
        else:
            raise NotImplementedError(f"Unkown embedding type {emb_type}")
        
        return embs_list, valid_mask_list

    def forward_head(self, decoder_out: torch.Tensor, mask_feature: torch.Tensor, attn_mask_target_size: Tuple[int, int]) -> \
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out: the decoded query of shape in shape (num_queries, batch_size, c), where c is the embeding dimension.
            mask_feature: in shape (batch_size, c, h, w).
            attn_mask_target_size : target attention mask size.

        Returns:
            cls_pred: Classification scores in shape (batch_size, num_queries, cls_out_channels). Note `cls_out_channels` should includes background.
            cls_emb_pred: Embedding prediction in shape (batch_size, num_queries, d_l). d_l is the dimension of embeddings.
            mask_pred: Mask scores in shape (batch_size, num_queries,h, w).
            attn_mask: Attention mask in shape (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out).transpose(0, 1)  # (batch_size, num_queries, c)
        
        cls_pred = self.cls_embed(decoder_out) # shape (batch_size, num_queries, num_classes + 1)
    
        cls_emb_pred = cls_pred   # shape (num_queries, batch_size, d_l)
        if self.use_class_emb:
            cls_emb_pred = self.v2l_transform(decoder_out)
            if self.pred_emb_norm:
                cls_emb_pred = cls_emb_pred / cls_emb_pred.norm(dim=-1, keepdim=True)
                
        mask_pred = torch.einsum('bqc,bchw->bqhw', self.mask_embed(decoder_out), mask_feature) # shape (num_queries, batch_size, h, w)
        attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='bilinear', align_corners=False)
    
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)  # shape (num_queries, batch_size, h, w) -> (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, cls_emb_pred, mask_pred, attn_mask

    def forward(self, feats: List[torch.Tensor], img_metas: List[Dict]) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """Forward function.

        Args:
            feats: Multi scale Features from the upstream network, each is a 4D-tensor.
            img_metas: List of image information.

        Returns:
            cls_pred_list: Classification logits for each decoder layer. Each is a 3D-tensor with shape (batch_size, num_queries, cls_out_channels). 
                Note `cls_out_channels` should includes background.
            cls_emb_pred_list: Embedding prediction for each decoder layer. Each is a 3D-tensor with shape (batch_size, num_queries, d_l). 
                d_l is the dimension of embeddings.
            mask_pred_list: Mask logits for each  decoder layer. Each with shape (batch_size, num_queries, h, w).
        """
        batch_size = len(img_metas)
        mask_features, multi_scale_memorys = self.pixel_decoder(feats)
        
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            decoder_input = decoder_input.flatten(2).permute(2, 0, 1) # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_inputs.append(decoder_input + self.level_embed.weight[i].view(1, 1, -1))
            
            mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-2:], dtype=torch.bool)  # shape (batch_size, c, h, w) -> (h*w, batch_size, c)
            decoder_positional_encoding = self.decoder_positional_encoding(mask)
            decoder_positional_encodings.append(decoder_positional_encoding.flatten(2).permute(2, 0, 1))
            
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))

        cls_pred_list = []
        cls_emb_pred_list = []
        mask_pred_list = []
        cls_pred, cls_emb_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        cls_pred_list.append(cls_pred)
        cls_emb_pred_list.append(cls_emb_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # if a mask is all True(all background), then set it all False.

            query_feat = self.transformer_decoder.layers[i](
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=[attn_mask, None],  # cross_attn + self_attn
                query_key_padding_mask=None,
                key_padding_mask=None)   # here we do not apply masking on padded region
            
            cls_pred, cls_emb_pred, mask_pred, attn_mask = self.forward_head(query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:])
            cls_pred_list.append(cls_pred)
            cls_emb_pred_list.append(cls_emb_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, cls_emb_pred_list, mask_pred_list
    
    def forward_train(self,
                    feats: List[torch.Tensor],
                    img_metas: List[Dict],
                    gt_bboxes: List[torch.Tensor],
                    gt_labels: List[torch.Tensor],
                    gt_masks: List[torch.Tensor],
                    gt_semantic_seg: Optional[List[torch.Tensor]],
                    gt_caption_ids: List[torch.Tensor],
                    gt_caption_mask: List[torch.Tensor],
                    gt_caption_nouns_ids: List[torch.Tensor],
                    gt_caption_nouns_mask: List[torch.Tensor],
                    gt_bboxes_ignore: Optional[torch.Tensor]=None,
                    gt_token_noun_indices: Optional[torch.Tensor]=None,
                    enable_debug: bool = False,
                    **kwargs) -> Dict[str, torch.Tensor]:
        """Forward function for training mode.

        Args:
            feats: Multi-level features from the upstream network, each is a 4D-tensor.
            img_metas: List of image information.
            gt_bboxes: Each element is ground truth bboxes of the image, shape (num_gts, 4). Not used here.
            gt_labels: Each element is ground truth labels of each box, shape (num_gts,).
            gt_masks: A list of instance masks of shape (num_gts, h, w).
            gt_semantic_seg: Each element is the ground truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things, [num_thing_class, num_class-1] means stuff, 255 means VOID. 
                It's None when training instance segmentation.
            gt_caption_ids: Each element is the caption token ids.
            gt_caption_mask: Each element is the caption mask. 1 represents valid token.
            gt_caption_nouns_ids: The id of the noun tokens.
            gt_caption_nouns_mask : The mask of the noun tokens.
            gt_bboxes_ignore: Ground truth bboxes to be ignored. Defaults to None.
            gt_token_noun_indices: A list of noun indices for different tokens in caption. Defaults to None.
            kwargs:
                gt_cat_names (list[list[str]]): List of List of category names
                of the corresponding label in gt_labels

        Returns:
            A dictionary of loss components
        """
        assert gt_bboxes_ignore is None

        all_layers_cls_scores, all_layers_cls_emb_preds, all_layers_mask_preds = self(feats, img_metas)

        gt_labels, gt_masks = self.preprocess_gt(gt_labels, gt_masks, gt_semantic_seg, img_metas)

        
        gt_caption_embs, gt_caption_nouns_embs = None, None
        if self.use_caption_generation:
            gt_caption_embs, gt_caption_mask = self._extract_word_embeddings(gt_caption_ids, gt_caption_mask, self.caption_gen_emb_type)
        
        if self._use_caption:
            gt_caption_avg_pooled_nouns_embs, gt_caption_avg_pool_nouns_mask = self._extract_noun_word_embeddings(
                ids_list=gt_caption_nouns_ids, 
                mask_list=gt_caption_nouns_mask, 
                token_noun_indices=gt_token_noun_indices, 
                emb_type=self.caption_emb_type
            )
        
        if enable_debug:
            print_debug = False
            for gt_token_noun_index in gt_token_noun_indices:
                for idx in range(gt_token_noun_index.max() + 1):
                    if (gt_token_noun_index == idx).sum() > 1:
                        print_debug = True
                        break
            if print_debug:
                print(gt_caption_avg_pooled_nouns_embs)
        

        return self.loss(all_layers_cls_scores, all_layers_cls_emb_preds, all_layers_mask_preds,
                        gt_labels, gt_masks, gt_caption_ids, gt_caption_embs, gt_caption_mask,
                        gt_caption_nouns_ids, gt_caption_avg_pooled_nouns_embs, gt_caption_avg_pool_nouns_mask, 
                        img_metas)

    def simple_test(self, feats: List[torch.Tensor], img_metas: List[Dict], **kwargs) -> Tuple[torch.Tensor]:
        """Tests without augmentaton.
        Args:
            feats: Multi-level features from the upstream network, each is a 4D-tensor.
            img_metas: List of image information.

        Returns:
            mask_cls_results: Mask classification logits, shape (batch_size, num_queries, cls_out_channels).
                Note `cls_out_channels` should includes background.
            mask_cls_emb_results: embedding predictions, shape (batch_size, num_queries, d_l).
            mask_pred_results (Tensor): Mask logits, shape (batch_size, num_queries, h, w).
            caption_generation_results: The results of the caption generation.
        """
        all_layer_cls_scores, all_cls_emb_preds, all_layer_mask_preds = self(feats, img_metas)
        mask_cls_results = all_layer_cls_scores[-1]
        mask_cls_emb_results = all_cls_emb_preds[-1]
        mask_pred_results = all_layer_mask_preds[-1]
        assigned_labels = mask_cls_results
        if kwargs.get('gt_labels', None) is not None:
            cls_emb_logits = self._get_cls_emb_logits_from_cls_emb_pred(mask_cls_emb_results)
            gt_masks = kwargs['gt_masks'][0][0].pad(img_metas[0]['pad_shape'][:2], pad_val=0).to_tensor(dtype=torch.long, device=cls_emb_logits.device)
            # (num_queries, )
            assigned_labels = self._get_target_single(mask_cls_results[0], cls_emb_logits[0], mask_pred_results[0], kwargs['gt_labels'][0][0], gt_masks, img_metas)[0]
            
        img_shape = kwargs['img_shape'] if kwargs.get('img_shape', None) else img_metas[0]['batch_input_shape']   
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)

        caption_generation_results = None
        if kwargs.get('with_caption', False) or ('cap_results' in self.test_cfg.get('eval_types', [])):
            caption_generation_results = beam_search(self, mask_cls_emb_results, BOS_TOKEN, EOS_TOKEN, emb_type=self.caption_emb_type, max_len=35, beam_width=7, logging=kwargs.get('logging'
    , False))
        
        att = None
        if kwargs.get('with_att', False):
            att = torch.matmul(mask_cls_emb_results[0,:,:], get_ids_embedding(self, kwargs['nouns_ids']).t())

        return assigned_labels, mask_cls_emb_results, mask_pred_results, caption_generation_results, att