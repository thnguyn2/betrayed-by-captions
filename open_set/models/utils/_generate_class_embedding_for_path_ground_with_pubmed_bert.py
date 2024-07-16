"""A helper module that generates the class embeddings for concepts in the Quilt dataset using the BioMed-BERT

This paper is SOTA for the BLURB benchmark.
Reference:
    https://huggingface.co/michiyasunaga/BioLinkBERT-base
    https://huggingface.co/michiyasunaga/BioLinkBERT-large
    
    https://discuss.huggingface.co/t/obtaining-word-embeddings-from-roberta/7735/14
"""
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch
import transformers
from tqdm import tqdm
from typing import Dict
from torch import nn

BERT_MODEL_BY_EMBEDDING_TYPES: Dict[str, str] = {
    'bert': 'bert-base-uncased',
    'pubmed-bert': 'NeuML/pubmedbert-base-embeddings',
}

from bert_embeddings import BERT_MODEL_BY_EMBEDDING_TYPES
from bert_embeddings import BertEmbeddings
_OUTPUT_CLASS_EMBEDDING_FILE = "datasets/embeddings/quilt_class_with_pubmed_bert_emb.json"
_EMBEDDING_DIM = 768


def _generate_class_embeddings_from_concepts(concept_file_path: Path) -> None:
    with open(str(concept_file_path), "r") as file:
        concepts = json.load(file)

    embedding_type = 'pubmed-bert'
    bert_embedder = BertEmbeddings(
        bert_model=transformers.AutoModel.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[embedding_type]).eval(),
        normalize_word_embeddings=False, # Perform pooling over subwords first before normalization!
    )        
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[embedding_type])
    all_class_embeddings: torch.Tensor = torch.zeros((len(concepts), _EMBEDDING_DIM), dtype=torch.float32)
    layernorm = nn.LayerNorm(normalized_shape=_EMBEDDING_DIM, eps=1e-9)
    with torch.no_grad():
        for idx, concept in tqdm(enumerate(concepts)):
            token_ids = tokenizer(concept, return_tensors="pt", truncation=False, add_special_tokens=False, padding=False)
            embs = bert_embedder.calculate_word_embeddings(token_ids['input_ids'])
            embs = _pooling_over_token_embeddings(
                outputs = embs,
                mask=token_ids['attention_mask'],
            )
            all_class_embeddings[idx] = layernorm(embs)

    json_obj = json.dumps([{"id": idx + 1, "name": concept, "emb": [x.item() for x in class_embd]} for idx, (concept, class_embd) in enumerate(zip(concepts, all_class_embeddings))] )
    with open(_OUTPUT_CLASS_EMBEDDING_FILE, "w") as out_file:
        out_file.write(json_obj)
        
def _pooling_over_token_embeddings(outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregates the embeddings of all tokens.
    
    Args:
        outputs: A tensor that has the shape of [1, N_tok, D], storing the token embeddings.
        mask: An [N_tok, 1] mask that stores the attention mask for different tokens.    
        
    Returns:
        A tensor of shape [1, D] that stored the mean-pooleed embddings from all tokens.
    """
    mask = mask.unsqueeze(-1).expand(outputs.size()).float()
    return torch.sum(outputs * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
    
if __name__ == "__main__":
    _generate_class_embeddings_from_concepts(concept_file_path=Path("open_set/datasets/utils/quilt_categories.json"))