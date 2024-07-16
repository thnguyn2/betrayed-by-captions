import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import get_dist_info

from mmdet.models.builder import LOSSES

def grounding_loss(cls_emb_pred: torch.Tensor, gt_caption_embs: torch.Tensor, gt_caption_mask: float, temperature: float) -> float:
    """Computing grounding loss

    Args:
        cls_emb_preds: A tensor of share (batch_size, num_queries, d_l) stores the class embedding for all the queries.
        gt_caption_embs: A tensor of shape (batch_size, max_tokens, d_l) that stores the word embeddings of the caption.
        gt_caption_mask: A tensor of shape (batch_size, max_tokens) that stores the mask of the word token.
    
    Returns:
        The value of the grounding loss
    """
    batch_size, num_queries, d_l = cls_emb_pred.shape
    _, num_max_tokens = gt_caption_mask.shape
    num_tokens = gt_caption_mask.sum(dim=1)  # batchsize

    # we should compute the image-sentence distances for all image-sentence pairs 
    # in the batch, rather than only matching ones. So we replicate them BxB times.
    cls_emb_pred = cls_emb_pred[None, :, :, :].repeat(batch_size, 1, 1, 1).reshape(batch_size**2, num_queries, d_l)
    gt_caption_embs = gt_caption_embs[:, None, :, :].repeat(1, batch_size, 1, 1).reshape(batch_size**2, num_max_tokens, d_l)
    gt_caption_mask = gt_caption_mask[:, None, :].repeat(1, batch_size, 1).reshape(batch_size**2, num_max_tokens)
    num_tokens = num_tokens[:, None].repeat(1, batch_size).reshape(batch_size**2)

    local_similarity = torch.bmm(gt_caption_embs, cls_emb_pred.transpose(1,2))  # (B**2, max_tokens. Nq)
    local_similarity = local_similarity / temperature

    attention_l2v = F.softmax(local_similarity, dim=2)
    attention_l2v = attention_l2v * gt_caption_mask[:, :, None]
    
    # This sum is in equation (1)
    local_distance = -local_similarity / temperature
    global_dist_l2v = (attention_l2v * local_distance).sum(dim=2).sum(dim=1) / torch.max(num_tokens, other=torch.ones_like(num_tokens))  # [B, B]
    global_dist_l2v = torch.where(num_tokens > 0, global_dist_l2v, global_dist_l2v.max().detach() + 100.0)
    pw_cost_l2v = global_dist_l2v.reshape(batch_size, batch_size)
    
    attention_v2l = F.softmax(local_similarity, dim=1)
    global_dist_v2l = ((attention_v2l * local_distance).sum(dim=2).sum(dim=1) / num_queries)
    global_dist_v2l = torch.where(num_tokens > 0, global_dist_v2l, global_dist_v2l.max().detach() + 100.0)
    pw_cost_v2l = global_dist_v2l.reshape(batch_size, batch_size)
   
    loss = (
        torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=0)).mean() + 
        torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=1)).mean() + 
        torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=0)).mean() + 
        torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=1)).mean()
    ) / 4

    return loss

@LOSSES.register_module()
class GroundingLoss(nn.Module):

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(GroundingLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

        self.grounding_loss = grounding_loss

    def forward(self,
                cls_emb_pred: torch.Tensor,
                gt_caption_embs: torch.Tensor,
                gt_caption_mask: torch.Tensor,
                temperature: float,
                **kwargs):
        """Forward function.

        Args:
            cls_score: The prediction.
            label: The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The method used to reduce the
                loss. Options are "none", "mean" and "sum".
            ignore_index (int | None): The label index to be ignored.
                If not None, it will override the default value. Default: None.
        Returns:
            torch.Tensor: The calculated loss.
        """
        loss_grounding = self.loss_weight * self.grounding_loss(
            cls_emb_pred,
            gt_caption_embs,
            gt_caption_mask,
            temperature,
            **kwargs)
        return loss_grounding
