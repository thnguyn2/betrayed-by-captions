"""A helper module that generates the class embeddings for concepts in the Quilt dataset using the BioMed-BERT

This paper is SOTA for the BLURB benchmark.
Reference:
    https://huggingface.co/michiyasunaga/BioLinkBERT-base
    https://huggingface.co/michiyasunaga/BioLinkBERT-large
    
    https://discuss.huggingface.co/t/obtaining-word-embeddings-from-roberta/7735/14
"""
import json
from pathlib import Path
from transformers import AutoTokenizer
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


def _generate_class_embeddings_from_concepts(concept_file_path: Path, normalized_pooled_token_embeddings: bool=True) -> None:
    """Generates the class embeddings for different concepts.
    
    Args:
        concept_file_path: The path to the medical concept file.
        normalized_pooled_token_embeddings (optional): If True, the pooled token embedding will be normalized.
    """
    with open(str(concept_file_path), "r") as file:
        concepts = json.load(file)

    embedding_type = 'pubmed-bert'
    bert_embedder = BertEmbeddings(
        bert_model=transformers.AutoModel.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[embedding_type]).eval(),
    )        
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[embedding_type])
    all_class_embeddings: torch.Tensor = torch.zeros((len(concepts), _EMBEDDING_DIM), dtype=torch.float32)
    layernorm = nn.LayerNorm(normalized_shape=_EMBEDDING_DIM, eps=1e-9) if normalized_pooled_token_embeddings else None
    with torch.no_grad():
        for idx, concept in tqdm(enumerate(concepts)):
            all_class_embeddings[idx] = bert_embedder.calculate_word_embeddings(
                torch.tensor(
                    tokenizer.encode(concept, add_special_tokens=False)
                )
            ).mean(dim=0)

    json_obj = json.dumps([{"id": idx + 1, "name": concept, "emb": [x.item() for x in class_embd]} for idx, (concept, class_embd) in enumerate(zip(concepts, all_class_embeddings))] )
    with open(_OUTPUT_CLASS_EMBEDDING_FILE, "w") as out_file:
        out_file.write(json_obj)
    
if __name__ == "__main__":
    _generate_class_embeddings_from_concepts(concept_file_path=Path("open_set/datasets/utils/quilt_categories.json"), normalized_pooled_token_embeddings=False)