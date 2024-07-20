import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from mmcv.runner import get_dist_info

from mmdet.models.builder import LOSSES

@LOSSES.register_module()
class GroundingLoss(nn.Module):

    def __init__(self,
                 reduction: str='mean',
                 loss_weight: float=1.0,
                 ):
        """CrossEntropyLoss.

        Args:
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(GroundingLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        
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
        loss_grounding = self.loss_weight * self._grounding_loss(
            cls_emb_pred,
            gt_caption_embs,
            gt_caption_mask,
            temperature,
            **kwargs)
        return loss_grounding
    
    def _grounding_loss(self, cls_emb_pred: torch.Tensor, gt_caption_embs: torch.Tensor, gt_caption_mask: float, temperature: float) -> float:
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
    
        return (
            torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=0)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=1)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=0)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=1)).mean()
        ) / 4

        
@LOSSES.register_module()
class GroundingLossWithSparistyConstrain(GroundingLoss):
    """A class that implements the grounding loss with the sparsity in the attention vision-to-language.
    
    Args:
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        sparsity_loss_relative_weight (optional): The weight of the sparsity loss. This should be less than 1.
            Defaults to 0.0, meaning that no sparsity loss is being used.
    """
    def __init__(
        self,
        word_embedding_dim: int,
        reduction: str='mean',
        loss_weight: float=1.0,
        sparsity_loss_relative_weight: float = 0.0,
        sparsity_coef: Optional[float]=None,
    ) -> None:
        super().__init__(reduction=reduction, loss_weight=loss_weight)
        if sparsity_loss_relative_weight > 1.0:
            raise ValueError("sparsity_loss_relative_weight can't be larger than 1")
        self._sparsity_loss_relative_weight = sparsity_loss_relative_weight
        self._no_des_emb = nn.Embedding(num_embeddings=1, embedding_dim=word_embedding_dim)
        self._sparsity_coef = sparsity_coef
        
    def _grounding_loss(self, cls_emb_pred: torch.Tensor, gt_caption_noun_embs: torch.Tensor, gt_caption_mask: float, temperature: float) -> float:
        """Computing grounding loss

        Args:
            cls_emb_preds: A tensor of share (batch_size, num_queries, d_l) stores the class embedding for all the queries.
            gt_caption_noun_embs: A tensor of shape (batch_size, max_tokens, d_l) that stores the embeddings of the caption nouns.
            gt_caption_mask: A tensor of shape (batch_size, max_tokens) that stores the mask of the word token.
        
        Returns:
            The value of the grounding loss
        """
        batch_size, num_queries, d_l = cls_emb_pred.shape
    
        gt_caption_noun_embs_with_no_desc = torch.cat([gt_caption_noun_embs, self._no_des_emb.weight[None, :].repeat(batch_size, 1, 1)], dim=1)  # (B, max_token + 1, d_l)
        gt_caption_mask = torch.cat([gt_caption_mask, torch.ones((batch_size, 1),device=gt_caption_mask.device)], dim=1)  # None description should be attended to
        _, num_max_tokens = gt_caption_mask.shape
        num_tokens = gt_caption_mask.sum(dim=1)  # batchsize

        # we should compute the image-sentence distances for all image-sentence pairs 
        # in the batch, rather than only matching ones. So we replicate them BxB times.
        cls_emb_pred = cls_emb_pred[None, :, :, :].repeat(batch_size, 1, 1, 1).reshape(batch_size**2, num_queries, d_l)
        gt_caption_noun_embs_with_no_desc = gt_caption_noun_embs_with_no_desc[:, None, :, :].repeat(1, batch_size, 1, 1).reshape(batch_size**2, num_max_tokens, d_l)
        gt_caption_mask = gt_caption_mask[:, None, :].repeat(1, batch_size, 1).reshape(batch_size**2, num_max_tokens)
        num_tokens = num_tokens[:, None].repeat(1, batch_size).reshape(batch_size**2)

        local_similarity = torch.bmm(gt_caption_noun_embs_with_no_desc, cls_emb_pred.transpose(1,2))  # (B**2, max_tokens + 1. Nq)
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
    
        similarity_loss = (
            torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=0)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_l2v, dim=1)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=0)).mean() + 
            torch.diag(-torch.log_softmax(-pw_cost_v2l, dim=1)).mean()
        ) / 4
        
        return (1 - self._sparsity_loss_relative_weight) * similarity_loss + \
            self._sparsity_loss_relative_weight * self._v2l_attention_sparsity_loss(
                att_v2l=attention_v2l, 
                noun_mask=gt_caption_mask,
                batch_size=batch_size)
            
    def _v2l_attention_sparsity_loss(self, att_v2l: torch.Tensor, noun_mask: torch.Tensor, batch_size: int) -> float:
        """Computes the vision to language attention loss.
        
        Args: 
            att_v2l: A tensor of shape [B^2, N_tok + 1, Nq]. Normalized to unit sum over the token dimension for each query. 
            batch_size: The size of the minibatch.
            noun_mask: A tensor of shape [B^2, N_token + 1] that defines the mask of valid noun.
        Returns:
            The value the sparsity loss.
        
        Reference:
            https://docs.google.com/document/d/128JlKiWCDNssiSuv_kQxzkrVG5sQAtM3hBXfsethgTs/edit#bookmark=id.xhvki9zdtqst
        """
        if self._sparsity_coef is None:
            self._sparsity_coef = 1.0 / att_v2l.size(-1)
        inner_image_att_v2l = att_v2l[0:batch_size:]
        noun_mask = noun_mask[0:batch_size:,:,None]  # [B, N_token + 1]
        # inner_image_att_v2l = inner_image_att_v2l * noun_mask  # Do not need to mask since the query should not match to a padded embedding.
        average_rho = inner_image_att_v2l.mean(dim=0)  # [Ntok + 1, Nq] - We want this mean to be very small + similar to that of a Bernoulli variable dist.
        loss = self._sparsity_coef * torch.log(self._sparsity_coef / average_rho) + (1 - self._sparsity_coef) * torch.log((1 - self._sparsity_coef) / (1-average_rho))
        return loss.mean()