"""A module to generate object context embeddings from the object embeddings and the text embeddings."""
import math
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from open_set.models.transformers.transformers import MultiHeadSelfAttention, MultiHeadCrossAttention

class ObjectContextGenerator(nn.Module):
    """A transformer-based module that takes in the object embeddings and text embeddings and generates the object
    context embeddings.

    It does the followings.
    1. Compute self-attention over the object embeddings.
    2. Compute the context prompting vectors by cross-attending to the object embeddings and the text embeddings.
        Let
            Qo = Proj_Q([Object, Text]) in R^{O x D}.
            K = Proj_K([Object, Text]) in R^{O x D}.
            V = Proj_V([Obj, Text]) in R^{O x D}.
        The attention output
            Qnext = softmax(Qo * K^T/srqt(D)) * V in R^{O x D}
        This attention allow attendings over the object embeddings dimensions.
        Here, masking will be used to make sure that the each entry in Qo can only attend to 1 entry in the object embeddings.

    Args:
        embedding_dim: The dimension of the object embeddings.
        object_embdding_dim: The dimension of the object embeddings.
        num_context_queries (optional): The number of object queries. Defaults to 50.

    References:
        Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017).
    """

    def __init__(
        self,
        embedding_dim: int,
        object_embdding_dim: int,
        num_context_queries: int = 50,
    ) -> None:
        super().__init__()
        self._object_queries = nn.Embedding(num_context_queries, embedding_dim)
        self._num_context_queries = num_context_queries
        self._num_attention_heads = 8

        # Decoder self-attention block
        self._dec_self_atten = MultiHeadSelfAttention(
            in_dim=embedding_dim,
            nb_heads=self._num_attention_heads,
        )
        self._dec_layer_norm1 = nn.LayerNorm(embedding_dim)
        self._to_lang_dom_projector = nn.Linear(object_embdding_dim, embedding_dim)

    def forward(self, object_embeddings: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Computes the context vector for the object embeddings by attending it to the prompt and the object
        embeddings.

        Args:
            object_embeddings: The object embeddings of shape [B, Nq, D] where Nq is the number of objects and D is the embedding dimension.

        Returns:
            - The projected object embeddings of shape [B, Nq, D]. Here Nctx is the number of context embeddings, which
            will be used to generate the outputs.
            - The updated context embeddings from self-attnetion of shape [B, Nctx, D].
            - The unmasked cross-attentions of shape [B, Nctx, Nq] between the context eembeddings and the object embeddings.
            It is QK^T / sqrt(D) where Q is the context embeddings and K is the object embeddings.

        References:
            https://github.com/pytorch/pytorch/issues/103668
            https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer
            https://github.com/pytorch-labs/gpt-fast/blob/main/model.py#L167
        """
        # Add a learnable object queries to the encoder of the coder.
        projected_object_embeddings = self._to_lang_dom_projector(object_embeddings)
        updated_context_embeddings = self._decoder_encoder_output(batch_size=object_embeddings.size(0))
        D = object_embeddings.size(-1)
        return (
            projected_object_embeddings,
            updated_context_embeddings,
            updated_context_embeddings @ projected_object_embeddings.transpose(1, 2) / (D**0.5),
        )

    def _decoder_encoder_output(self, batch_size: int) -> torch.Tensor:
        """Computes the ouputs of the encoder for the decocder part.

        Args:
            batch_size: The size of the training minibatch

        Returns:
            A tensor of shape [B, Nctx, D] that contains the updated context embeddings based on self-attention.
        """
        object_queries = self._object_queries.weight.unsqueeze(0).expand(batch_size, -1, -1)  # [B, Nctx, D].
        x = object_queries + self._dec_self_atten(
            src=object_queries,  # [B, O, D]
            mask=None,
        )
        return self._dec_layer_norm1(x)




class DescriptionGenerator(nn.Module):
    """A module that uses Causual Mask Attention to generate the object names and the whole image description.

    Args:
        token_embeder: The token embedder module
        tokenizer: The tokenizer module.
        seq_len (optional): The length of the output sequence. Defaults to 256
        num_llm_layers (optional): The number of layers for the Causual LLM. Defaults to 8.

    References:
    """

    def __init__(
        self,
        token_embeder: nn.Module,
        tokenizer: nn.Module,
        seq_len: int = 256,
        num_llm_layers: int = 8,
    ) -> None:
        super().__init__()
        self._seq_len = seq_len
        self._token_embeddings = token_embeder.embeddings.word_embeddings.weight.detach()
        self._tokenizer = tokenizer
        embedding_dim = token_embeder.config.hidden_size
        self._language_model = CausualLanguageModel(
            num_layers=num_llm_layers,
            embedding_dim=embedding_dim,
            vocab_size=tokenizer.vocab_size,
            embeddings_weight=self._token_embeddings,
            max_seq_length=seq_len,
        )

        self._layer_norm1 = nn.LayerNorm(embedding_dim)
        self._ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self._layer_norm2 = nn.LayerNorm(embedding_dim)

    
    def generate_global_desc_logits(
        self,
        object_queries: torch.Tensor,
        updated_context_embeddings: torch.Tensor,
        context_to_query_atten_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the global image description logits.

        Args:
            object_queries: The object query tensor of shape [B, Nq, D] where B is the batchsize,
                Nq is the number of object queries, and D is the dimension of the embeddings.
            updated_context_embeddings: The updated context embeddings of shape [B, Nctx, D] where B is the batchsize.
            context_to_query_atten_mask: A tensor of shape [B, Nctx, Nq] that stores the attention mask for the context-to-query attention.

        Returns:
            The generated whole image description logit tensor of shape [B, Nctx, D].
        """
        context_to_query_weight = context_to_query_atten_mask.softmax(dim=-1)
        updated_context_embeddings = (
            updated_context_embeddings + context_to_query_weight @ object_queries
        )  # [B, Nctx, D]
        updated_context_embeddings = self._layer_norm1(updated_context_embeddings)
        updated_context_embeddings = updated_context_embeddings + self._ffn(updated_context_embeddings)
        updated_context_embeddings = self._layer_norm2(updated_context_embeddings)
        return self._language_model(updated_context_embeddings)

    def generate_object_desc_logits(
        self,
        object_queries: torch.Tensor,
        updated_context_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Generate the object name logits for different object queries.

        Note that we have Nctx context queries, when these queries pays attentions to a single object queries k.
        The context_to_query weight in R^[Nctx x 1] will be all 1's b/c there is no other queries to attend to.
        Hence, the output of cross attention is a repeat of Qk for Nctx times.

        Qctx = Qctx + ones(Nctx, 1) * qk with Qctx in R^[Nctx x D] and qk in R^[1 x D]
        Qctx in R^[Nctx x D]

        Args:
            object_queries: A tensor of shape [B, Nq, D] for the object query.
                Nq is the number of object queries, and D is the dimension of the embeddings.
            updated_context_embeddings: The updated context embeddings of shape [B, Nctx, D] where B is the batchsize.

        Returns:
            The generated object names and the whole image description of shape [B, Nq, Nctx, D].
        """
        repeated_context_embeddings = updated_context_embeddings.unsqueeze(1)  # [B, 1, Nctx, D]
        repeated_object_queries = object_queries.unsqueeze(2)  # [B, Nq, 1, D]
        repeated_context_embeddings = repeated_context_embeddings + repeated_object_queries  # [B, Nq, Nctx, D]
        repeated_context_embeddings = self._layer_norm1(repeated_context_embeddings)
        repeated_context_embeddings += self._ffn(repeated_context_embeddings)
        repeated_context_embeddings = self._layer_norm2(repeated_context_embeddings)
        return self._language_model(repeated_context_embeddings)

class CausualLanguageModel(nn.Module):
    """A module that implements a transformer layers with masking.

    Performs self-attentions and predicts the logits of different tokens.
    Args:
        num_layers: The number of transformer layers.
        embedding_dim: The dimension of the embeddings.
        vocab_size: The size of the vocabulary.
        embeddings_weight (optional): The weight of the logit embeddings layers. Defaults to None, in which case, a learnable weight will be used.
        max_seq_length (optional): The maximum sequence length. Defaults to 256.
    """

    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        vocab_size: int,
        embeddings_weight: Optional[torch.Tensor] = None,
        max_seq_length: int = 256,
    ) -> None:
        super().__init__()
        if embeddings_weight is None:
            self._embeddings_weight = nn.Parameter(torch.randn(vocab_size, embedding_dim))
        else:
            self._embeddings_weight = embeddings_weight

        self._positional_embedder = SinusoidalPositionEmbedding(embed_size=embedding_dim, max_seq_length=max_seq_length)
        self._layers = torch.nn.ModuleList(
            [
                CausualTransfomerLayer(
                    embedding_dim=embedding_dim,
                    num_attention_heads=8,
                    max_seq_length=max_seq_length,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass prediction logits of the inputs.

        Args:
            x: The input tensor of shape [B, O, S, D], where B is the batch size, O is the number of objects in the case of object description
            generation [B, S, D] in the case of global description generation.
            S is the sequence length, and D is the dimension of the embeddings.

        Returns:
            The logits of the predicted tokens of shape [B, O, S, V], where V is the vocabulary size.
        """
        x_shape_org = x.shape
        S, D = x_shape_org[-2:]
        x = self._positional_embedder(x.reshape(-1, S, D))
        for layer in self._layers:
            x, _ = layer(x)
        return torch.matmul(x.reshape(*x_shape_org), self._embeddings_weight.T.to(x.device))


class CausualTransfomerLayer(nn.Module):
    """A module that implements a transformer layer with masking.

    Args:
        embedding_dim: The dimension of the embeddings.
        num_attention_heads (optional): The number of attention heads. Defaults to 8.
        max_seq_length (optional): The maximum sequence length. Defaults to 256.

    References:
    """

    def __init__(
        self,
        embedding_dim: int,
        num_attention_heads: int = 8,
        max_seq_length: int = 256,
    ) -> None:
        super().__init__()
        self._num_attention_heads = num_attention_heads
        self._self_atten = MultiHeadSelfAttention(
            in_dim=embedding_dim,
            nb_heads=self._num_attention_heads,
        )
        self._norm1 = nn.LayerNorm(embedding_dim)
        self._ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self._norm2 = nn.LayerNorm(embedding_dim)
        self._attention_mask = torch.triu(torch.ones((max_seq_length, max_seq_length)), diagonal=1).bool().unsqueeze(0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the forward pass of the transformer layer.

        Args:
            x: The input tensor of shape [B, S, D], where B is the batch size, S is the sequence length, and D is the dimension of the embeddings.
        Returns:
            The transformed tensor and the attention scores.
        """
        self._attention_mask = self._attention_mask.to(x.device)  # [1, S, S]
        self_atten_out = self._self_atten(
            src=x,
            mask=self._attention_mask,
        )
        x = self._norm1(x + self_atten_out)
        x = self._norm2(x + self._ffn(x))
        return x, None


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
