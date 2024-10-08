# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Optional
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import build_match_cost
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from typing import Dict, Optional
try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

@BBOX_ASSIGNERS.register_module()
class MaskHungarianAssignerOpen(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth for
    mask.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, mask focal cost and mask dice cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (:obj:`mmcv.ConfigDict` | dict): Classification cost config.
        mask_cost (:obj:`mmcv.ConfigDict` | dict): Mask cost config.
        dice_cost (:obj:`mmcv.ConfigDict` | dict): Dice cost config.
    """

    _BACKGROUND_OBJECT_INDEX = 0
    _BACKGROUND_OBJECT_LABEL = -1
    
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.0),
                 mask_cost=dict(
                     type='FocalLossCost', weight=1.0, binary_input=True),
                 dice_cost=dict(type='DiceCost', weight=1.0),
                 cls_emb_cost=dict(type='ClassficationCost', weight=1.0),
                ):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        self.cls_emb_cost = build_match_cost(cls_emb_cost)
           
    def assign(self,
               cls_pred: Optional[torch.Tensor],
               cls_emb_pred: torch.Tensor,
               known_cls_emb: torch.Tensor,
               mask_pred: Optional[torch.Tensor],
               gt_labels: torch.Tensor,
               gt_mask: torch.Tensor,
               softmax_temperature: float,
               img_meta: Dict,
               gt_bboxes_ignore=None,
               eps: Optional[float] = 1e-7) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred: Class prediction in shape (num_query, cls_out_channels) for a single layer of a single image.
            cls_emb_pred: Predicted class embedding in shape (num_query, emb_dim) from a single layer of a single image.
            known_cls_emb: The target class embeddings of shape (N_cls, emb_dim).
            mask_pred: The (sampled) predicted mask logit in shape of (num_query, num_points) for a single layer of a single image.
            gt_labels: Groundtruth label of 'gt_mask'in shape = (num_gt, ) one for each object in the image.
                Here num_gt is the number of objects in the groundtruth image.
                The labels of the mask is in the range of [0, 1, ... #known_class - 1]
            gt_mask: The (sampled) groundtruth mask in the shape of (num_queries, num_points).
            softmax_temperature: The softmax temperature.
            img_meta: Meta information for current image.
            eps: A value added to the denominator for numerical stability. Default 1e-7.

        Returns:
            The assigned result that contains the followings
                num_gt: The number of groundtruths.
                assigned_gt_inds: A tensor of shape (N_query,) that stores the index of the object that has a match with a query.
                    An index 0 is used for the background object and 1, ... num_gt is used for the for ground objects.
                assigned_labels: A tensor of shape (N_query,) that stores the labels of the matched object.
                    A label of -1 is used for the label of the background object and 0, 1, ... #known class - 1 
                    are used the label of foreground objects.
        """

        assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
        # K-Net sometimes passes cls_pred=None to this assigner.
        # So we should use the shape of mask_pred
        num_gt, num_query = gt_labels.shape[0], mask_pred.shape[0]

        
        # 1. Initialize defaults to background object
        assigned_gt_inds = mask_pred.new_full((num_query, ), self._BACKGROUND_OBJECT_INDEX, dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_query, ), self._BACKGROUND_OBJECT_LABEL, dtype=torch.long)
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            return AssignResult(
                num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification, embedding, and mask cost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        
        if self.cls_emb_cost.weight != 0:
            cls_emb_pred_logits = torch.matmul(cls_emb_pred, known_cls_emb.t()) / softmax_temperature if cls_emb_pred is not None else None
            # gt_labels has the range of [0, 1, ..., # known class - 1], no background clas.
            cls_emb_cost = self.cls_emb_cost(cls_emb_pred_logits, gt_labels)  # [N_query, num_gt]
        else:
            cls_emb_cost = 0

        if self.mask_cost.weight != 0:
            mask_cost = self.mask_cost(mask_pred, gt_mask)  # [num_query, num_gt]
        else:
            mask_cost = 0

        if self.dice_cost.weight != 0:
            dice_cost = self.dice_cost(mask_pred, gt_mask)
        else:
            dice_cost = 0

        cost = cls_cost + cls_emb_cost + mask_cost + dice_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_query_inds, matched_mask_inds = linear_sum_assignment(cost)
        matched_query_inds = torch.from_numpy(matched_query_inds).to(mask_pred.device)
        matched_mask_inds = torch.from_numpy(matched_mask_inds).to(mask_pred.device)

        # 4. assign foregrounds based on matching results
        assigned_gt_inds[matched_query_inds] = matched_mask_inds + 1
        assigned_labels[matched_query_inds] = gt_labels[matched_mask_inds]
        
        return AssignResult(
            num_gt, assigned_gt_inds, None, labels=assigned_labels)

    def find_novel_class_query_indices(
        self,
        cls_emb_pred: torch.Tensor,
        known_cls_emb: torch.Tensor,
        caption_noun_emb: torch.Tensor,
        query_indices_to_avoid: torch.Tensor,
        softmax_temperature: float,
    ) -> Optional[torch.Tensor]:
        """Computes one-to-one matching for each novel caption noun embeddings to the query.

        Args:
            cls_emb_pred: Predicted class embedding in shape (num_query, emb_dim) from a single layer of a single image.
            known_cls_emb: The target class embeddings of shape (N_cls, emb_dim) of the known classes
            caption_noun_emb: The caption noun embeddings of shape (num caption nouns, embd_dim)
            query_indices_to_avoid: The index of the query to avoid
            softmax_temperature: The softmax temperature.
            
        Returns:
            A list of query indices that were matched to novel noun embeddings
        """
        _LARGE_COST_VAL = 1e+9
        novel_noun_emb = self._calculate_novel_noun_embd(known_cls_emb=known_cls_emb, caption_noun_emb=caption_noun_emb)
        if novel_noun_emb.size(0) == 0:
            return None
        else:
            query_emb_pred_logits = torch.matmul(cls_emb_pred, novel_noun_emb.t()) / softmax_temperature if cls_emb_pred is not None else None
            cost = -query_emb_pred_logits.detach().cpu()
            cost[query_indices_to_avoid] = _LARGE_COST_VAL
            return linear_sum_assignment(cost)[0]
    
    @staticmethod
    def _calculate_novel_noun_embd(
        known_cls_emb: torch.Tensor,
        caption_noun_emb: torch.Tensor,
        ) -> torch.Tensor:
        _EQUAL_THRESH = 1e-5
        # https://stackoverflow.com/questions/62588779/pytorch-differences-between-two-tensors
        caption_noun_emb_exp = caption_noun_emb.unsqueeze(1)
        res = ((caption_noun_emb_exp - known_cls_emb).norm('fro', dim=-1) < _EQUAL_THRESH)
        novel_class_idx = ~res.any(dim=-1)
        return caption_noun_emb[novel_class_idx]
        
        