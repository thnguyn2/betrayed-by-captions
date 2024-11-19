# A module that defines a module to generate captions for objects.
import numpy as np
import torch
import torch.nn as nn 

from .transformers import TransformerDecoder, PositionalEncoding
from transformers import AutoTokenizer, AutoModel
from .captioner_cores import ObjectContextGenerator, DescriptionGenerator
from mmdet.models.builder import HEADS
from .caption_tranformer import build_mask

@HEADS.register_module()
class ObjectCaptioner(nn.Module):
    """A simple LLM decoder module that takes in the object embeddings and outputs the prediction logits.

    This module can be used to generate captions and other types of text outputs.

    Args:
        num_queries: The number of object queries.
        object_embedding_dim: The dimension of the object embeddings.
        tokenizer_max_length (optional): The maximum length for the tokenizer to pad to. Defaults to 50.
        max_desc_length (optional): The maximum length for the description. Defaults to 256.
        obj_desc_max_length (optional): The maximum length for the object description. Defaults to 10.
        embedder_model_id (optional): The huggingface model id for the embedder model. Defaults to "michiyasunaga/BioLinkBERT-large", which is currently the SOTA model
        in the Biomedical NLP benchmark, see the leaderboard at https://microsoft.github.io/BLURB/leaderboard.html
        num_descriptor_llm_layers (optional): The number of layers for the Descriptor generator. Defaults to 3.
    References:
        https://github.com/thnguyn2/quilt-llava/tree/pathai/notebooks
    """

    def __init__(
        self,
        num_queries: int,
        object_embedding_dim: int,
        obj_desc_max_length: int = 50,
        embedder_model_id: str = "michiyasunaga/BioLinkBERT-large",
        num_descriptor_llm_layers: int = 3,
    ) -> None:
        super().__init__()
        self._num_queries = num_queries
        self._tokenizer = AutoTokenizer.from_pretrained(embedder_model_id)
        self._obj_desc_max_length = obj_desc_max_length
        self._embedder = AutoModel.from_pretrained(embedder_model_id)
        self._object_context_generator = ObjectContextGenerator(
            embedding_dim=self._embedder.config.hidden_size,
            object_embdding_dim=object_embedding_dim,
            num_context_queries=obj_desc_max_length,
            
        )
        self._desc_generator = DescriptionGenerator(
            seq_len=obj_desc_max_length,
            token_embeder=self._embedder,
            tokenizer=self._tokenizer,
            num_llm_layers=num_descriptor_llm_layers,
        )
        
        
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None
            ) -> torch.Tensor:
        """Forward pass for the ObjectCaptioner module.
        
        Args:
            tgt: A tensor of shape [B, S, D] representing the embeddings of different words in target the captions.
            memory: An embedding prediction for a single decoder layer for all images with shape (B, Nq, D).
                D is the dimension of embeddings.
            tgt_mask: An S x S tensor representing the causual mask for caption generation.
            tgt_key_padding_mask: A tensor of shape [B, S] representing the padding mask for the captions. It will be 1 where the padding is present.
            
        Returns:
            - None
            - The logits for the predicted captions with shape (B, S, V), where V is the vocabulary size.
        """
        
        (
            projected_object_embeddings,
            updated_context_embeddings,
            context_to_obj_attention,
        ) = self._object_context_generator(memory)

        # object_desc_logits = self._desc_generator.generate_object_desc_logits(
        #     object_queries=projected_object_embeddings,
        #     updated_context_embeddings=updated_context_embeddings,
        # )
        object_desc_logits = None

        global_desc_logits = self._desc_generator.generate_global_desc_logits(
            object_queries=projected_object_embeddings,
            updated_context_embeddings=updated_context_embeddings,
            context_to_query_atten_mask=context_to_obj_attention,
        )
        return object_desc_logits, global_desc_logits[:,1:,:]