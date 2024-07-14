"""A helper script to preprocess the PanNuke dataset.

Reference:
    https://www.kaggle.com/code/jeanpat/minimalist-mask-to-coco-format-dataset-conversion
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
"""
from absl import logging
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Any, Dict, List
from skimage import measure
from shapely.geometry import Polygon
from PIL import Image

_CLASS_NAMES_BY_IDS = {
    1: "Neoplastic cells", 
    2: "Inflammatory", 
    3: "Connective/Soft tissue cells", 
    4: "Dead Cells", 
    5: "Epithelial", 
    6: "Background",
}

_BG_CLASS_ID = 6

logging.set_verbosity(v=logging.INFO)

def _generate_dataset() -> None:
    _ROOT_FOLDER = Path("/jupyter-users-home/tan-2enguyen/datasets/pathology/pannuke")
    TRAIN_FOLDS = ["Fold 1", "Fold 2"]
    VAL_FOLDS = ["Fold 3"]
    dir_by_name = _create_output_dataset_folders(target_folder_path=_ROOT_FOLDER / "pannuke_coco")
    
    starting_annotation_id = 1
    train_annotation_dicts = []
    val_annotation_dicts = []
    train_image_infos = []
    val_image_infos = []
    image_id = 1
    
    for fold in TRAIN_FOLDS + VAL_FOLDS:
        _ORG_DATASET_FOLDER = _ROOT_FOLDER / fold
        images = np.load(_ORG_DATASET_FOLDER/"images"/"images.npy").astype(np.uint8)
        image_types = np.load(_ORG_DATASET_FOLDER/"images"/"types.npy")
        masks = np.load(_ORG_DATASET_FOLDER/"masks"/"masks.npy")
        for image, organ, mask in tqdm(zip(images, image_types, masks)):
            im = Image.fromarray(image)
            file_name = f"{image_id}.jpg"
            im.save(dir_by_name['image_dir'] / file_name)
            if fold in TRAIN_FOLDS:
                train_image_infos.append({
                    "license": 1,
                    "file_name": file_name,
                    "coco_url": "",
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "date_captured": "",
                    "flickr_url": "",
                    "organ": organ,
                    "id": image_id,
                })
            else:
                val_image_infos.append({
                    "license": 1,
                    "file_name": file_name,
                    "coco_url": "",
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "date_captured": "",
                    "flickr_url": "",
                    "organ": organ,
                    "id": image_id,
                })
                
            for class_id in _CLASS_NAMES_BY_IDS:
                if class_id != _BG_CLASS_ID:
                    single_channel_annotation_dicts = _create_sub_mask_annotations(
                        image_id=image_id, 
                        class_id=class_id,
                        starting_annotation_id=starting_annotation_id,
                        single_class_mask=mask[:,:, _class_id_to_channel_id(class_id)],
                    )
                    if fold in TRAIN_FOLDS:
                        train_annotation_dicts.extend(single_channel_annotation_dicts)
                    else:
                        val_annotation_dicts.extend(single_channel_annotation_dicts)
                        
                    starting_annotation_id += len(single_channel_annotation_dicts)
            image_id += 1
    
    train_dataset_meta: Dict[str, Any] = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
        "images": train_image_infos,  
        "annotations": train_annotation_dicts,
        "categories": _create_category_list(class_name_by_ids=_CLASS_NAMES_BY_IDS)
    }
    train_json_object = json.dumps(train_dataset_meta)
    
    with open(str(dir_by_name['annotation_dir'] / "train_instances.json"), "w") as file:
        file.write(train_json_object)
        
    val_dataset_meta: Dict[str, Any] = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
        "images": val_image_infos,  
        "annotations": val_annotation_dicts,
        "categories": _create_category_list(class_name_by_ids=_CLASS_NAMES_BY_IDS)
    }
    val_json_object = json.dumps(val_dataset_meta)
    
    with open(str(dir_by_name['annotation_dir'] / "val_instances.json"), "w") as file:
        file.write(val_json_object)
  
def _class_id_to_channel_id(class_id: int) -> int:
    return class_id - 1

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
        "description": "PanNuke Dataset",
        "url": "https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke",
        "version": "1.0",
        "year": 2020,
        "contributor": "",
        "date_created": "2024/07/12"
    }

def _create_sub_mask_annotations(class_id: int, image_id: int, starting_annotation_id: int, single_class_mask: np.ndarray) -> List[Dict[str, Any]]:
    """Returns a list of an annotation dictionaries, one for each object in the single_class_mask.
    
    Args:
        class_id: The id of the class.
        image_id: The id of the input image that the caption is for.
        starting_annotation_id: The starting id of the annotation. One image can have multiple annotations, 1 annotation for each class.
        single_class_mask: A [H, W] array that stores the binary segmentation mask for a single class.
    
    References:
        https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    """
    _MIN_REGION_AREA_PIXELS = 5.0
    _MIN_CONTOUR_PIXS_REQUIRED = 4
    if class_id != _BG_CLASS_ID:
        object_contours = measure.find_contours(single_class_mask, 0.5)
        annotation_dicts: List[Dict[str, Any]] = []
        valid_off_set_idx = 0
        for offset_idx, object_contour in enumerate(object_contours):
            object_contour = np.array(object_contour)
            rows_pixs, cols_pixs = object_contour[:, 0] - 1, object_contour[:, 1] - 1
            if object_contour.shape[0] > _MIN_CONTOUR_PIXS_REQUIRED:
                contour_pixels = np.stack([cols_pixs, rows_pixs], axis=1)
                object_polygon = Polygon(contour_pixels).simplify(1.0, preserve_topology=False)    
                x_min, y_min, x_max, y_max = object_polygon.bounds
                if object_polygon.area > _MIN_REGION_AREA_PIXELS:
                    try:
                        segmentation_res = np.array(object_polygon.exterior.coords).ravel().tolist()
                    except:
                        logging.warning(f"Failed to extract the annotation for object {offset_idx} of image {image_id} - Ignored.")
                        continue
                        
                    annotation_dicts.append(
                        {
                            'segmentation':  [segmentation_res,],  # [[XYXY...]],
                            'is_crowd': 0,  # single object annotation
                            'image_id': image_id,
                            'category_id': class_id,
                            'id': starting_annotation_id + valid_off_set_idx,
                            'bbox': (x_min, y_min, x_max-x_min, y_max - y_min),
                            'area': object_polygon.area
                        }
                    )
                    
                    valid_off_set_idx += 1
        return annotation_dicts
    else:
        raise NotImplementedError("Segmentation mask extraction for the background is not implemented yet!")
    
def _create_category_list(class_name_by_ids: Dict[int, str]) -> List[Dict[str, str]]:
    return [
        {
            "supercategory": "cell",
            "id": class_id,
            "name": class_name,
        } 
        for class_id, class_name in class_name_by_ids.items()
    ]
            
if __name__ == "__main__":
    _generate_dataset()