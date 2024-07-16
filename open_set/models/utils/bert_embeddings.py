import torch
from torch import nn
import transformers
from typing import Dict

BERT_MODEL_BY_EMBEDDING_TYPES: Dict[str, str] = {
    'bert': 'bert-base-uncased',
    'pubmed-bert': 'NeuML/pubmedbert-base-embeddings',
}

class BertEmbeddings(nn.Module):
    """Load word_embeddings and LayerNorm from huggingface BERT checkpoint to
    decrease the size of saved checkpoint
    
    Args:
        bert_model: The BERT model used to calculate the word embeddings.
        normalize_word_embeddings (optional): If True, the word embedding will be normalized.
    """
    def __init__(self, bert_model: transformers.models.bert.modeling_bert.BertModel, normalize_word_embeddings: bool=True):
        super().__init__()
        config: transformers.models.bert.configuration_bert.BertConfig = bert_model.config
        self._normalize_word_embeddings = normalize_word_embeddings
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.word_embeddings.load_state_dict(bert_model.embeddings.word_embeddings.state_dict())
        self.LayerNorm.load_state_dict(bert_model.embeddings.LayerNorm.state_dict())
        
    def calculate_word_embeddings(self, ids: torch.Tensor) -> torch.Tensor:
        embs = self.word_embeddings(ids)
        if self._normalize_word_embeddings:
            embs = self.LayerNorm(embs)
        return embs