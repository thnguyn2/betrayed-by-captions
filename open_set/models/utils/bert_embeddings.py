import torch
from torch import nn
import transformers
from typing import Dict, Tuple

BERT_MODEL_BY_EMBEDDING_TYPES: Dict[str, str] = {
    'bert': 'bert-base-uncased',
    'pubmed-bert': 'NeuML/pubmedbert-base-embeddings',
}

class BertEmbeddings(nn.Module):
    """Load word_embeddings and LayerNorm from huggingface BERT checkpoint to
    decrease the size of saved checkpoint
    
    Args:
        bert_model: The BERT model used to calculate the word embeddings.
    """
    _TOKEN_DIM = 0
    def __init__(self, bert_model: transformers.models.bert.modeling_bert.BertModel):
        super().__init__()
        config: transformers.models.bert.configuration_bert.BertConfig = bert_model.config
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.word_embeddings.load_state_dict(bert_model.embeddings.word_embeddings.state_dict())
        self.LayerNorm.load_state_dict(bert_model.embeddings.LayerNorm.state_dict())
        
    def calculate_word_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        """Calculates the work embeddings of tokens ids.
        
        Args:
            ids: A tensor of shape (Ntoken,) that specifies the ids of the tokens.
            
        Returns:
            A tensor of shape (Ntoken, emb) that contains the embeddings of 
            
        Note that a noun can be converted to many tokens. Similar to a sentence.
        For example, the word `girrafe` can be tokenize into [21025, 11335, 7959].
        The embeddings saved in the file embeddings/coco_class_with_bert_emb.json are 
        average pooled across tokens of a noun. 
        Hence, we need to make sure that we need to properly account for this in the grounding loss.    
        """
        embs = self.word_embeddings(ids)
        embs = self.LayerNorm(embs)
        return embs
    
    def calculate_mean_pool_embeddings(self, ids: torch.Tensor, noun_idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculates the work embeddings of tokens ids.
        
        Args:
            ids: A tensor of shape (Ntoken,) that specifies the ids of the tokens.
            
        Returns:
            A tensor of shape (N_token_max, emb) that contains the averged pool embeddings of the nouns.
            A tensor of shape (N_token_max,) that contains the emebedding masks of the nouns
            
        Note that a noun can be converted to many tokens. Similar to a sentence.
        For example, the word `girrafe` can be tokenize into [21025, 11335, 7959].
        The embeddings saved in the file embeddings/coco_class_with_bert_emb.json are 
        average pooled across tokens of a noun. 
        Hence, we need to make sure that we need to properly account for this in the grounding loss.    
        """
        _PADDED_NOUN_IDX = -1
        _VALID_MASK_IDX = 1
       
        embs = self.word_embeddings(ids)
        embs = self.LayerNorm(embs)
        max_num_tokens, emb_dim = embs.shape
        num_nouns = noun_idx.max() + 1
        noun_idx[noun_idx == _PADDED_NOUN_IDX] = num_nouns
        reduced_embs = torch.zeros(max_num_tokens, emb_dim, device=ids.device)
        reduced_embs.index_reduce_(0, noun_idx, embs, "mean", include_self=False)
        
        out_mask = torch.zeros(max_num_tokens, device=ids.device)
        out_mask[:num_nouns] = _VALID_MASK_IDX
        return reduced_embs, out_mask
    