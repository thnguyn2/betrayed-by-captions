"""A module that implements the multi-modal Hungarian assigner."""
import torch
from typing import Dict, Optional
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs.builder import build_match_cost
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner

@BBOX_ASSIGNERS.register_module()
class MultiModelHungarianAssignerOpen(BaseAssigner):
    """Computes one-to-one matching between predictions and the ground truth

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    class embedding classification cost, mask focal cost (optional) and mask dice cost (optional). 
    The targets don't include the no_object. This class allow matching a query
    to the groundtruth with caption only or segmentation mask + mask class.
    In general, there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no matching found
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_emb_cost: Class embedding cost configuration.
        mask_cost: Mask cost configuration.
        dice_cost: Dice cost configuration.
    """
    def __init__(
        self,
        cls_emb_cost: Dict=dict(type='ClassificationCost', weight=1.0),
        mask_cost: Dict=dict(type='FocalLossCost', weight=1.0, binary_input=True),
        dice_cost=dict(type='DiceCost', weight=1.0)
    ):
        self._cls_emb_cost = build_match_cost(cls_emb_cost)
        self._mask_cost = build_match_cost(mask_cost)
        self._dice_cost = build_match_cost(dice_cost)
        
    def assign(self,
        cls_pred: Optional[torch.Tensor],       
        cls_emb_pred: torch.Tensor,
        mask_pred: Optional[torch.Tensor],
        gt_labels: torch.Tensor,
        gt_mask: torch.Tensor,
        gt_noun_emb: torch.Tensor,
        img_meta: Dict,
        gt_bboxes_ignore=None,
        eps: Optional[float] = 1e-7,
    ) -> AssignResult:
        """Computes one-to-one matching between the prediction and the groundtruth.
        
        Args:
            cls_pred: The predicted class logits in the shape of (num_query, cls_out_channels) for the queries.
            cls_emb_pred: Predicted class embedding logits in shape (num_query, d_l) for a single layer of a single image.  d_l is the
                embedding dimension.
            mask_pred: The (sampled) predicted mask logit in shape of (num_query, num_points) for a single layer of a single image.
                This can be None, in which case, there is no groundtruth mask to match with.
            gt_labels: A tensor of shape = (num_gt, ) that store the labels of the groundtruth mask.
            gt_mask: The (sampled) groundtruth mask in the shape of (num_queries, num_points).
                This can be None, in which case, there is no gfoundtruth mask to match with.
            gt_noun_emb: The embedding of the groundtruth, which is of shape (num_gt, dl) that contains the class embedding of the groundtruth.
            img_meta: Meta information for current image.
            eps: A value added to the denominator for numerical stability. Default 1e-7.

        Returns:
            The assigned result.
        """
        raise NotImplementedError