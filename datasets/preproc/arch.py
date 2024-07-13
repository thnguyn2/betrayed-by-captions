"""A helper script to preprocess the ARCH dataset so that it is consistent with the COCO format.

Reference:
    https://www.kaggle.com/code/jeanpat/minimalist-mask-to-coco-format-dataset-conversion
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
"""
from absl import logging
import json
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List
from PIL import Image

logging.set_verbosity(v=logging.INFO)

def _generate_dataset() -> None:
    _ROOT_FOLDER = Path("/jupyter-users-home/tan-2enguyen/datasets/pathology/arch")
    
    dir_by_name = _create_output_dataset_folders(target_folder_path=_ROOT_FOLDER / "arch_coco")
    image_infos = []
    annotations = []
    image_id = 1
    caption_id = 1
    
    for subset in ["books_set", "pubmed_set"]:
        _ORG_DATASET_FOLDER = _ROOT_FOLDER / subset
        IMG_FOLDER = _ORG_DATASET_FOLDER / "images"
        with open(str(_ORG_DATASET_FOLDER / "captions.json"), "r") as caption_file:
            caption_dicts = json.load(caption_file)
        
        for caption_val in tqdm(caption_dicts.values()):
            if 'letter' not in caption_val or caption_val['letter'].lower() == 'single':  # Only use images that associate with 1 caption.
                uuid = caption_val['uuid']
                image_path = IMG_FOLDER / f"{uuid}.png"
                if image_path.exists():
                    im = Image.open(image_path)
                    target_file_name = f"{image_id}.jpg"
                    im.convert('RGB').save(dir_by_name['image_dir'] / target_file_name)

                    image_infos.append({
                        "license": 1,
                        "file_name": target_file_name,
                        "coco_url": "",
                        "height": im.size[1],
                        "width": im.size[0],
                        "date_captured": "",
                        "flickr_url": "",
                        "id": image_id,
                    })
                    
                    annotations.append(
                        {
                            "image_id": image_id,
                            "id": caption_id,
                            "caption": caption_val['caption'],
                        }
                    )
                    image_id += 1
                    caption_id += 1
                else:
                    logging.warning(f"Image {uuid}.png does not exist!")
                
    dataset_meta: Dict[str, Any] = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
        "images": image_infos,  
        "annotations": annotations,
    }
    json_object = json.dumps(dataset_meta)
    
    with open(str(dir_by_name['annotation_dir'] / "captions.json"), "w") as file:
       file.write(json_object)
    print("Done converting the ARCH dataset!")


def _create_output_dataset_folders(target_folder_path: Path) -> Dict[str, Path]:
    target_folder_path.mkdir(exist_ok=True)
    annotation_dir = target_folder_path / "annotations"
    annotation_dir.mkdir(exist_ok=True)
    image_dir = target_folder_path / "images"
    image_dir.mkdir(exist_ok=True)
    return {
        'image_dir': image_dir,
        'annotation_dir': annotation_dir
    }
    
def _create_dataset_license_info() -> List[Dict[str, Any]]:
    return [
        {
            "url": "",
            "id": 1,
            "name": "Attribution-NonCommercial-ShareAlike 4.0 International"
        }
    ]
    
def _create_dataset_info() -> Dict[str, Any]:
    """Generates a dictionary that contains the dataset information."""
    return {
        "description": "ARCH Dataset",
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/arch",
        "version": "1.0",
        "year": 2021,
        "contributor": "",
        "date_created": "2024/07/12"
    }
            
if __name__ == "__main__":
    _generate_dataset()