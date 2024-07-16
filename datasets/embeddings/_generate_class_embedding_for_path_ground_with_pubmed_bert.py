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
from tqdm import tqdm

_OUTPUT_CLASS_EMBEDDING_FILE = "datasets/embeddings/quilt_class_with_pubmed_bert_emb.json"
_EMBEDDING_DIM = 768
def _generate_class_embeddings_from_concepts(concept_file_path: Path) -> None:
    with open(str(concept_file_path), "r") as file:
        concepts = json.load(file)
    
    tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings")
    model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings")
    all_class_embeddings: torch.Tensor = torch.zeros((len(concepts), _EMBEDDING_DIM), dtype=torch.float32)
    with torch.no_grad():
        for idx, concept in tqdm(enumerate(concepts)):
            token_ids = tokenizer(concept, return_tensors="pt", truncation=False, add_special_tokens=False, padding=False)
            all_class_embeddings[idx] = _pooling_over_token_embeddings(
                outputs = model(**token_ids),
                mask=token_ids['attention_mask'],
            )

    json_obj = json.dumps([{"id": idx + 1, "name": concept, "emb": [x.item() for x in class_embd]} for idx, (concept, class_embd) in enumerate(zip(concepts, all_class_embeddings))] )
    with open(_OUTPUT_CLASS_EMBEDDING_FILE, "w") as out_file:
        out_file.write(json_obj)
        
def _pooling_over_token_embeddings(outputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Aggregates the embeddings of all tokens.
    
    Args:
        outputs: An object whose 0-th output contains word embeddings for all token embeddings. This output has the following dimensions [1, N_tok, D]
        mask: An [N_tok, 1] mask that stores the attention mask for different tokens.    
        
    Returns:
        A tensor of shape [1, D] that stored the mean-pooleed embddings from all tokens.
    """
    outputs = outputs[0]  #First element of model_output contains all token embeddings
    mask = mask.unsqueeze(-1).expand(outputs.size()).float()
    return torch.sum(outputs * mask, dim=1) / torch.clamp(mask.sum(1), min=1e-9)
    
if __name__ == "__main__":
    _generate_class_embeddings_from_concepts(concept_file_path=Path("open_set/datasets/utils/quilt_categories.json"))