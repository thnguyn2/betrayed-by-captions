# Copyright (c) OpenMMLab. All rights reserved.
import torch
from typing import Optional
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import build_match_cost
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from typing import Dict
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

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.0),
                 mask_cost=dict(
                     type='FocalLossCost', weight=1.0, binary_input=True),
                 dice_cost=dict(type='DiceCost', weight=1.0),
                 cls_emb_cost=dict(type='ClassficationCost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.mask_cost = build_match_cost(mask_cost)
        self.dice_cost = build_match_cost(dice_cost)
        self.cls_emb_cost = build_match_cost(cls_emb_cost)
            
    def assign(self,
               cls_pred: Optional[torch.Tensor],
               cls_emb_pred: torch.Tensor,
               mask_pred: Optional[torch.Tensor],
               gt_labels: torch.Tensor,
               gt_mask: torch.Tensor,
               img_meta: Dict,
               gt_bboxes_ignore=None,
               eps: Optional[float] = 1e-7) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.

        Args:
            cls_pred: Class prediction in shape (num_query, cls_out_channels) for a single layer of a single image.
            cls_emb_pred: Predicted class embedding logits in shape (num_query, cls_out_channels) for a single layer of a single image.
            mask_pred: The (sampled) predicted mask logit in shape of (num_query, num_points) for a single layer of a single image.
            gt_labels: Groundtruth label of 'gt_mask'in shape = (num_gt, ) one for each object in the image.
            gt_mask: The (sampled) groundtruth mask in the shape of (num_queries, num_points).
            img_meta: Meta information for current image.
            eps: A value added to the denominator for numerical stability. Default 1e-7.

        Returns:
            The assigned result.
        """
        assert gt_bboxes_ignore is None, 'Only case when gt_bboxes_ignore is None is supported.'
        # K-Net sometimes passes cls_pred=None to this assigner.
        # So we should use the shape of mask_pred
        num_gt, num_query = gt_labels.shape[0], mask_pred.shape[0]

        # 1. assign -1 by default
        assigned_gt_inds = mask_pred.new_full((num_query, ),-1, dtype=torch.long)
        assigned_labels = mask_pred.new_full((num_query, ), -1, dtype=torch.long)
        if num_gt == 0 or num_query == 0:
            # No ground truth or boxes, return empty assignment
            if num_gt == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gt, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification, embedding, and mask cost.
        if self.cls_cost.weight != 0 and cls_pred is not None:
            cls_cost = self.cls_cost(cls_pred, gt_labels)
        else:
            cls_cost = 0
        
        if self.cls_emb_cost.weight != 0 and cls_emb_pred is not None:
            cls_emb_cost = self.cls_emb_cost(cls_emb_pred, gt_labels)
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

        cost = cls_cost + cls_emb_cost + mask_cost + mask_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')

        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(mask_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(mask_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gt, assigned_gt_inds, None, labels=assigned_labels)
