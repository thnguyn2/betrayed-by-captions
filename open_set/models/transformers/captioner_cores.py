"""A module to generate object context embeddings from the object embeddings and the text embeddings."""
import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
from open_set.models.transformers.transformers import MultiHeadSelfAttention, MultiHeadCrossAttention


class CausualDecoderLayer(nn.Module):
    """A module that implements a causual transformer layer.

    Args:
        embedding_dim: The dimension of the embeddings.
        num_attention_heads (optional): The number of attention heads. Defaults to 8.
    
    References:
        models.transformers.transformers.DecoderBlock
    """

    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self._self_atten = MultiHeadSelfAttention(
            in_dim=embedding_dim,
            nb_heads=num_attention_heads,
        )
        
        self._cross_atten = MultiHeadCrossAttention(
            in_dim=embedding_dim,
            nb_heads=num_attention_heads,
        )
        
        self._ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        
        self._self_atten_norm = nn.LayerNorm(embedding_dim)
        self._cross_atten_norm = nn.LayerNorm(embedding_dim)
        self._ffn_norm = nn.LayerNorm(embedding_dim)
        
    
    def forward(
        self,
        text_emb: torch.Tensor, 
        obj_emb: torch.Tensor, 
        text_causual_mask: torch.Tensor, 
        text_pad_mask: torch.Tensor, 
        per_obj_desc_output: bool,
        obj_mask: Optional[torch.Tensor]=None, 
        ) -> torch.Tensor:
        """Performs the following operations.
        
        1. Causual masked self-attention of tgt using the attention mask of tgt_mask and tgt_key_padding_mask to mask invalid tokens.
        2. Cross-attention of tgt to the memory embeddings using the memory_mask for cross-attention.
        3. Layer normalization and feedforward network.
        
        Args:
            text_emb: A tensor of shape [B, S, D] representing the embeddings of different words in target the captions where S is the sequence length.
                B is the batch size and D is the dimension of the embeddings.
                tgt is the embeddings of the <BOS> tgt1, tgt2, tgt3, ..., tgt[S-1] tokens.
                
            obj_emb: An embedding prediction for a single decoder layer for all images with shape (B, Nq, D).
                D is the dimension of embeddings. Nq is the number of queries.
            
            text_causual_mask: An [1, S, S] upper triangular mask used for causual masked self-attention.
            memory_mask: A mask that can be used to mask the query embeddings. This can be used to generate the caption for single objects.
            
            text_pad_mask: A tensor of shape [B, S] representing the padding mask for the captions. It will be True where the padding is present
                and False where the padding is not present.
            
            per_obj_desc_output: A flag to indicate whether the output is for the object description or the whole image description.
            obj_mask (optional): A tensor of shape [B, Nq] representing the padding mask for the queries. Defaults to None.
        
        Returns:
            A tensor of shape [B, S, D] when per_obj_desc_output is False.
            A tensor of shape [B, Nq, S, D] when per_obj_desc_output is True.
        """
        out = self._self_atten(src=text_emb, mask=text_causual_mask, key_padding_mask=text_pad_mask)
        agg = text_emb + out
        self_atten_out = self._self_atten_norm(agg)  # [B, S, D]
        
        if per_obj_desc_output:
            out = self._calculate_obj_cross_atten_output(
                self_atten_out=self_atten_out,
                obj_emb=obj_emb)
        else:
            out = self._calculate_whole_image_cross_atten_output(
                self_atten_out=self_atten_out,
                obj_emb=obj_emb, 
                obj_mask=obj_mask)

        out = self._ffn(agg)
        agg = agg + out
        agg = self._ffn_norm(agg)
        return agg
                    
    def _calculate_whole_image_cross_atten_output(self, self_atten_out: torch.Tensor, obj_emb: torch.Tensor, obj_mask: torch.Tensor):
        out = self._cross_atten(
            qry=self_atten_out,  # [B, S, D]
            key=obj_emb,  # [B, Nq, D]
            val=obj_emb,  # [B, Nq, D]
            mask=obj_mask, 
            key_padding_mask=None,
        )
        agg = self_atten_out + out    
        agg = self._cross_atten_norm(agg)
        return agg
        
    def _calculate_obj_cross_atten_output(self, self_atten_out: torch.Tensor, obj_emb: torch.Tensor):
        """Returns the 
        
        The cross-attention output of tensor `qry = self_atten_out[q, s]` to only the object embeddings `key=obj_emb[q]` is
            softmax_{over_all_keys}(qry*key^T) * key will results in a tensor obj_emb[b, q]. Hence,
                self_atten_out[q, s] += obj_emb[q] for all index S.
        Args:
            self_atten_out: A tensor of shape [B * Nq, S, D]
            obj_emb: An object embeddings of shape (B, Nq, D). 
            
        """
        D = obj_emb.size(-1)
        obj_desc_cross_atten_output = self_atten_out + obj_emb.reshape(-1, D).unsqueeze(1)  # [B * Nq, S, D]
        agg = self._cross_atten_norm(obj_desc_cross_atten_output)
        return agg
    

class SinusoidalPositionEmbedding(nn.Module):
    """A module that implements the sinusoidal position encoding.

    PE[pos, 2i] = sin(pos / 10000^(2i/d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    where pos is the position in the sequence with the value of 0 to max_seq_length - 1,
    and i is the dimension of the embeddings. 0<=2i<D
    Args:
        embd_size: The dimension of the embeddings (D)
        max_seq_length: The maximum sequence length. The maximum length of the sequence

    References:
        https://medium.com/@pranay.janupalli/understanding-sinusoidal-positional-encoding-in-transformers-26c4c161b7cc
    """

    def __init__(self, embed_size: int, max_seq_length: int):
        super().__init__()
        position = torch.arange(max_seq_length).unsqueeze(1)  # [S, 1]
        normlized_even_embedding_indices = torch.arange(0, embed_size, 2) / embed_size  # 2i/d
        div_term = torch.exp(-math.log(10000.0) * normlized_even_embedding_indices)  # [D/2]
        position_encodings = torch.zeros(max_seq_length, embed_size)  # [S, D]
        position_encodings[:, 0::2] = torch.sin(position * div_term)
        position_encodings[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('position_encodings', position_encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the positional embeddings to the input tensor x and return the new tensor.

        Args:
            x: A input tensor of shape [B, S, D], where B is the (multiple of) batch size, S is the sequence length, and D is the dimension of the embeddings.
        """
        return x + self.position_encodings[None, : x.size(-2), :]

class VLMTransformerDecoder(nn.Module):
    """A transformer module that uses Causual Mask Attention to generate the object names and the whole image description.

    Note that this class can handle flexible sequence length.
    Args:
        text_embedding_dim: The dimension of the text embeddings.
        num_layers (optional): The number of layers for the Causual LLM. Defaults to 4.

    References:
    """

    def __init__(
        self,
        text_embedding_dim: nn.Module,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self._layers = torch.nn.ModuleList(
            [
                CausualDecoderLayer(
                    embedding_dim=text_embedding_dim,
                    num_attention_heads=8,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt_im_dsc_emb: torch.Tensor, 
        obj_emb: torch.Tensor, 
        text_causual_mask=None, 
        obj_mask=None, 
        text_pad_mask=None, 
        ) -> List[torch.Tensor]:
        """Performs the following operations.
        
        1. Causual masked self-attention of tgt using the attention mask of tgt_mask and tgt_key_padding_mask to mask invalid tokens.
        2. Cross-attention of tgt to the memory embeddings using the memory_mask for cross-attention.
        3. Layer normalization and feedforward network.
        
        Args:
            tgt_im_dsc_emb: A tensor of shape [B, S, D] representing the embeddings of different words in target image description
                where S is the sequence length. B is the batch size and D is the dimension of the embeddings.
                It can be written as <BOS> tgt1, tgt2, tgt3, ..., tgt[S-1] tokens.
                
            obj_emb: An object embeddings of shape (B, Nq, D).
            
            text_causual_mask: An S x S upper triangular mask used for causual masked self-attention.
            memory_mask: A mask that can be used to mask the query embeddings. This can be used to generate the caption for single objects.
            
            text_pad_mask: A tensor of shape [B, S] representing the padding mask for the captions. It will be True where the padding is present
                and False where the padding is not present.
            
            obj_mask: A tensor of shape [B, Nq] representing the padding mask for the queries. Defaults to None.
        
        Returns:
            A list of outputs of the transformer decoder layers. Each list items has a shape of [B, S, D] containing the output of 1 transformer decoder layer.
        """        
        im_desc_layer_outputs = []
        
        for layer in self._layers:
            text_emb = layer(
                text_emb=tgt_im_dsc_emb, 
                obj_emb=obj_emb, 
                text_causual_mask=self._generate_text_causual_mask(
                    text_causual_mask=text_causual_mask,
                    tgt_im_dsc_emb=tgt_im_dsc_emb,
                ), 
                obj_mask=obj_mask, 
                text_pad_mask=text_pad_mask,
                per_obj_desc_output=False,
            )
            im_desc_layer_outputs.append(text_emb)
        return im_desc_layer_outputs
    
    def _generate_text_causual_mask(self, text_causual_mask: Optional[torch.Tensor], tgt_im_dsc_emb: torch.Tensor) -> torch.Tensor:
        """Generates a causual attention mask for the text embeddings.

        The attention mask is is an upper triangular matrix of shape [1, S, S] where S is the sequence length.
        The entry of True means that the attention should be MASKED. 
        The entry of False means that the attention should be KEPT.
        Args:
            device: The device that the mask will be stored.

        Returns:
            The causual mask tensor of shape [1, S, S].
        """
        if text_causual_mask is not None:  
            return text_causual_mask
        
        _ZERO_OUT_DIAGONAL = 1  # The zero-th token should pay attention to the zero-th input token, which is the <BOS> token.
        seq_len = tgt_im_dsc_emb.size(-2)
        return torch.triu(torch.ones((seq_len, seq_len), device=tgt_im_dsc_emb.device), diagonal=_ZERO_OUT_DIAGONAL).bool().unsqueeze(0)

    def generate_obj_desc_outputs(
        self,
        obj_dsc_emb: torch.Tensor, 
        obj_emb: torch.Tensor, 
        ) -> List[torch.Tensor]:
        """Performs the following operations.
        
        1. Causual masked self-attention of tgt using the attention mask of tgt_mask and tgt_key_padding_mask to mask invalid tokens.
        2. Cross-attention of tgt to the memory embeddings using the memory_mask for cross-attention.
        3. Layer normalization and feedforward network.
        
        Args:
            tgt_im_dsc_emb: A tensor of shape [1, S, D] representing the embeddings of the sequence [CLS]<PAD>...<PAD> where S is the sequence length.
                This sequence is used as in the input for the object description generation.
            
            obj_emb: An object embeddings of shape (B, Nq, D).
                D is the dimension of embeddings. Nq is the number of queries.
                         
        Returns:
            A list of outputs of the transformer decoder layers. Each list items has a shape of [B, Nq, S, D] containing the output of 1 transformer decoder layer.
        """        
        B, Nq = obj_emb.shape[:2]
        obj_dsc_emb = obj_dsc_emb.repeat(B * Nq, 1, 1)  # [B * Nq, S, D]
        obj_desc_layer_outputs = []
        for layer in self._layers:
            obj_dsc_emb = layer(
                text_emb=obj_dsc_emb, 
                obj_emb=obj_emb, 
                text_causual_mask=self._generate_text_causual_mask(
                    text_causual_mask=None,
                    tgt_im_dsc_emb=obj_dsc_emb,
                ), 
                obj_mask=None, 
                text_pad_mask=None,
                per_obj_desc_output=True,
            )
            obj_desc_layer_outputs.append(obj_dsc_emb)
        return obj_desc_layer_outputs
        
    def _generate_obj_desc_input_embeddings(self, num_queries: int) -> torch.Tensor:
        """Generates an input tensor embeddings of shape [B, Nq, S, Dt] for object description generation.
        
        Args:
            num_queries: The number of object queries.
        """
        return 0.0