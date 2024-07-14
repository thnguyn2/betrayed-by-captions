"""A helper script to preprocess the QUILT-1M dataset so that it is consistent with the COCO format.

Reference:
    https://www.kaggle.com/code/jeanpat/minimalist-mask-to-coco-format-dataset-conversion
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
"""
from absl import logging
import json
import math
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, List
from PIL import Image
import pandas as pd

logging.set_verbosity(v=logging.INFO)

def _generate_dataset() -> None:
    _ROOT_FOLDER = Path("/jupyter-users-home/tan-2enguyen/datasets/pathology/quilt1m")
    
    dir_by_name = _create_output_dataset_folders(target_folder_path=_ROOT_FOLDER / "quilt_coco")
    train_image_infos = []
    val_image_infos = []
    train_annotations = []
    val_annotations = []
    image_id = 1
    caption_id = 1
    
    IMG_FOLDER = _ROOT_FOLDER / "images"
    df = pd.read_csv(_ROOT_FOLDER / "quilt_1M_lookup.csv")
    
    # Keep the resolution high.
    df = df[df.magnification > 1.0]
    
    for row in tqdm(df.iterrows()):
        image_meta = row[1]
        image_name = image_meta.image_path
        full_image_path = IMG_FOLDER / image_name
        if full_image_path.exists() and not math.isnan(image_meta.magnification):
            im = Image.open(full_image_path)
            target_file_name = f"{image_id}.jpg"
            im.convert('RGB').save(dir_by_name['image_dir'] / target_file_name)
            if image_meta.split == "train":
                train_image_infos.append({
                    "license": 1,
                    "file_name": target_file_name,
                    "coco_url": "",
                    "height": im.size[1],
                    "width": im.size[0],
                    "date_captured": "",
                    "flickr_url": "",
                    "id": image_id,
                })
                train_annotations.append(
                    {
                        "image_id": image_id,
                        "id": caption_id,
                        "caption": image_meta.caption,
                    }
                )
            else:
                val_image_infos.append({
                    "license": 1,
                    "file_name": target_file_name,
                    "coco_url": "",
                    "height": im.size[1],
                    "width": im.size[0],
                    "date_captured": "",
                    "flickr_url": "",
                    "id": image_id,
                })
                val_annotations.append(
                    {
                        "image_id": image_id,
                        "id": caption_id,
                        "caption": image_meta.caption, #image_meta.corrected_text if isinstance(image_meta.corrected_text, str) else image_meta.caption,
                    }
                )
            image_id += 1
            caption_id += 1
        
    train_dataset_meta: Dict[str, Any] = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
        "images": train_image_infos,  
        "annotations": train_annotations,
    }
    train_json_object = json.dumps(train_dataset_meta)
    
    with open(str(dir_by_name['annotation_dir'] / "train_captions.json"), "w") as file:
       file.write(train_json_object)
       
    val_dataset_meta: Dict[str, Any] = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
        "images": val_image_infos,  
        "annotations": val_annotations,
    }
    val_json_object = json.dumps(val_dataset_meta)
    
    with open(str(dir_by_name['annotation_dir'] / "val_captions.json"), "w") as file:
       file.write(val_json_object)
       
    print("Done writing for the QUILT-1M dataset!")


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
            "name": "MIT License"
        }
    ]
    
def _create_dataset_info() -> Dict[str, Any]:
    """Generates a dictionary that contains the dataset information."""
    return {
        "description": "QUILT-1M Dataset",
        "url": "https://quilt1m.github.io/",
        "version": "1.0",
        "year": 2023,
        "contributor": "",
        "date_created": "2024/07/13"
    }
            
if __name__ == "__main__":
    _generate_dataset()