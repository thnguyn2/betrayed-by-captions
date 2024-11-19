# A module that defines a module to generate captions for objects.
import torch
import torch.nn as nn 
from typing import List, Tuple

from .captioner_cores import SinusoidalPositionEmbedding
from transformers import AutoTokenizer, AutoModel
from .captioner_cores import VLMTransformerDecoder
from mmdet.models.builder import HEADS
from .caption_tranformer import build_mask


@HEADS.register_module()
class ObjectCaptioner(nn.Module):
    """A simple LLM decoder module that takes in the object embeddings and outputs the prediction logits.

    This module can be used to generate captions and other types of text outputs.

    Args:
        object_embedding_dim: The dimension of the object embeddings.
        max_seq_length (optional): The maximum sequence length for the description. This is often chosen to be the maximum
            desc length for the caption Defaults to 50.
        obj_desc_max_length (optional): The maximum length for the object description. Defaults to 8.
        embedder_model_id (optional): The huggingface model id for the embedder model. Defaults to "michiyasunaga/BioLinkBERT-large", which is currently the SOTA model
            in the Biomedical NLP benchmark, see the leaderboard at https://microsoft.github.io/BLURB/leaderboard.html
        num_decoder_layers (optional): The number of layers for the Descriptor generator. Defaults to 3.
    References:
        https://github.com/thnguyn2/quilt-llava/tree/pathai/notebooks
    """

    def __init__(
        self,
        object_embedding_dim: int,
        max_seq_length: int = 35,
        obj_desc_max_length: int = 8,
        embedder_model_id: str = "michiyasunaga/BioLinkBERT-large",
        num_decoder_layers: int = 3,
    ) -> None:
        super().__init__()
        token_embeder = AutoModel.from_pretrained(embedder_model_id)
        text_embedding_dim = token_embeder.config.hidden_size
        
        self._obj_to_text_dim_adapter = nn.Linear(object_embedding_dim, text_embedding_dim) if text_embedding_dim != object_embedding_dim else nn.Identity()
        self._position_encoder=SinusoidalPositionEmbedding(
            embed_size=object_embedding_dim, 
            max_seq_length=max_seq_length,
        )
        
        
        self._vlm_transformer_decoder = VLMTransformerDecoder(
            text_embedding_dim=text_embedding_dim,
            num_layers=num_decoder_layers,
        )

        self._tokenizer = AutoTokenizer.from_pretrained(embedder_model_id)
        self._token_embeddings = token_embeder.embeddings # [V, Dt]
        self._obj_desc_max_length = obj_desc_max_length
        self.generator = nn.Linear(text_embedding_dim, self._token_embeddings.word_embeddings.weight.size(0))
    
    
    def forward(self, 
                tgt: torch.Tensor, 
                memory: torch.Tensor, 
                tgt_mask=None, 
                memory_mask=None, 
                tgt_key_padding_mask=None, 
                memory_key_padding_mask=None
            ) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Performs next token predictions.
        
        Args:
            tgt: A tensor of shape [B, S, Dt] representing the embeddings of different words in the whole-image captions where S is the sequence length.
                B is the batch size and Dt is the dimension of the text embeddings.
                tgt is the embeddings of the <BOS> tgt1, tgt2, tgt3, ..., tgt[S-1] tokens.
                
            memory: An embedding prediction for a single decoder layer for all images with shape (B, Nq, Do).
                Do is the dimension of object embeddings. Nq is the number of queries.
            
            tgt_mask: An S x S upper triangular mask used for causual masked self-attention.
            memory_mask: A mask that can be used to mask the query embeddings. This can be used to generate the caption for single objects.
            
            tgt_key_padding_mask: A tensor of shape [B, S] representing the padding mask for the captions. It will be True where the padding is present
                and False where the padding is not present.
            
            memory_key_padding_mask: A tensor of shape [B, Nq] representing the padding mask for the queries. Defaults to None.
        
        Returns:
            - A list of length NL for all layers embeddings where NL is the number of layers in the transformer decoder. Each list items has the shape of [B, S, Dt].
            - Last layer logits for the whole-image description [B, S, V] for caption generation.
            - Last layer logits for per-object image description of shape [B, Nq, S, V] for object captioning.
        """
        proj_obj_emb = self._obj_to_text_dim_adapter(memory)  # [B, Nq, Dt]
        im_desc_layers_outputs = self._vlm_transformer_decoder(
            tgt_im_dsc_emb=self._position_encoder(tgt),  # [B, S, Dt]
            obj_emb=proj_obj_emb, 
            text_causual_mask=tgt_mask, 
            obj_mask=memory_mask, 
            text_pad_mask=tgt_key_padding_mask,
        )
        
        obj_desc_layers_outputs = self._vlm_transformer_decoder.generate_obj_desc_outputs(
            obj_dsc_emb=self._generate_obj_desc_token_embeddings(obj_desc_max_length=self._obj_desc_max_length),
            obj_emb=proj_obj_emb, 
        )
        return im_desc_layers_outputs, self.generator(im_desc_layers_outputs[-1]), self.generator(obj_desc_layers_outputs[-1])
        
    def _generate_obj_desc_token_embeddings(self, obj_desc_max_length: int) -> torch.Tensor:
        """Generates the token embeddings for the object description.
        
        Args:
            obj_desc_max_length: The maximum description length of the object

        Returns:
            A tensor of shape [1, S, Dt] representing the token embeddings of [CLS]<PAD>...<PAD> for the object description.
        """
        token_ids = torch.tensor(
            [self._tokenizer.cls_token_id] + [self._tokenizer.pad_token_id] * (obj_desc_max_length - 1), 
            device=self._token_embeddings.word_embeddings.weight.device,
        )
        return self._position_encoder(self._token_embeddings.LayerNorm(self._token_embeddings.word_embeddings(token_ids)))