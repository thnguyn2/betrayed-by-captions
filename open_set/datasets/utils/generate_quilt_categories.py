
from absl import logging
import pandas as pd
from pathlib import Path
import spacy
from typing import Dict, List
from tqdm import tqdm
import json

logging.set_verbosity(logging.INFO)
def _generate_categories():
    _ROOT_FOLDER = Path("/jupyter-users-home/tan-2enguyen/datasets/pathology/quilt1m")
      
    df = pd.read_csv(_ROOT_FOLDER / "quilt_1M_lookup.csv")
    df = df[df.magnification > 1.0]
    
    all_concepts: List[Dict]= []
    for _, row in tqdm(df.med_umls_ids.items()):
        if isinstance(row, str):
            for entities in eval(row):
                for concept in entities:
                    all_concepts.append(concept['entity'].lower())
    
    all_concepts = list(set(all_concepts))
    print(f"Found {len(all_concepts)} categories!")
    
    nlp = spacy.load("en_core_web_sm")
    
    lemmatized_concepts = []
    for c in tqdm(all_concepts):
        lemmatized_concepts.append(_process_concept_name(nlp=nlp, concept=c))
    lemmatized_concepts = list(set(lemmatized_concepts))
    print(f"Found {len(lemmatized_concepts)} categories after lemmatization")
    
    json_obj = json.dumps(lemmatized_concepts)
    with open("open_set/datasets/utils/quilt_categories.json", "w") as file:
        file.write(json_obj)

def _process_concept_name(nlp, concept: str) -> str:
    """Performs tokenization, lemmatizatio (to account for morphological variations), and skip the part in the parenthesis."""
    tokens = nlp(concept)
    lemma_s = []
    for token in tokens:
        token_lemma = token.lemma_
        if token_lemma.startswith('('): 
            break
        lemma_s.append(token_lemma)
    lemma_s = ' '.join(lemma_s)
    lemma_s = lemma_s.replace(' - ','-')
    return lemma_s

if __name__ == "__main__":
    _generate_categories()