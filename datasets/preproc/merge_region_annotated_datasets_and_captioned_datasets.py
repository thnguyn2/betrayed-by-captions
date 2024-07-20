

from absl import logging
from collections import defaultdict
from glob import glob
import json
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Any, Dict, List

_DATASETS_IMG_FOLDERS_BY_NAME = {
    'PanNuke': Path('/jupyter-users-home/tan-2enguyen/datasets/pathology/pannuke/pannuke_coco/images'),
    'Quilt': Path('/jupyter-users-home/tan-2enguyen/datasets/pathology/quilt1m/quilt_coco/images'),
}

ANNOTATION_SPLITS_TO_MERGES = {
    'train': {
        'PanNuke': '/jupyter-users-home/tan-2enguyen/datasets/pathology/pannuke/pannuke_coco/annotations/train_instances.json',
        'Quilt': '/jupyter-users-home/tan-2enguyen/datasets/pathology/quilt1m/quilt_coco/annotations/train_captions.json',
    },
    
    'val': {
        'PanNuke': '/jupyter-users-home/tan-2enguyen/datasets/pathology/pannuke/pannuke_coco/annotations/val_instances.json',
        'Quilt': '/jupyter-users-home/tan-2enguyen/datasets/pathology/quilt1m/quilt_coco/annotations/train_captions.json',
    }
}

logging.set_verbosity(logging.INFO)
        
def _merge_all() -> None:
    _TARGET_FOLDER = Path('/jupyter-users-home/tan-2enguyen/datasets/pathology/anno_caption_merged')
    output_folders = _create_output_dataset_folders(target_folder_path = _TARGET_FOLDER)
    old_image_path_to_new_image_id = _old_image_path_to_new_image_id(img_folders_by_name=_DATASETS_IMG_FOLDERS_BY_NAME, target_folder=output_folders['image_dir'])
    
    for split, anno_dict in ANNOTATION_SPLITS_TO_MERGES.items():
        logging.info(f"Processing {split} split")
        _generate_region_segmentation_annotation_file(
            split=split, 
            anno_dict_by_name=anno_dict, 
            image_path_to_new_id=old_image_path_to_new_image_id,
            annotation_folder=output_folders['annotation_dir'],
        )
        _generate_caption_annotation_files(
            split=split, 
            anno_dict_by_name=anno_dict, 
            image_path_to_new_id=old_image_path_to_new_image_id,
            annotation_folder=output_folders['annotation_dir'],
        )
       

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

def _old_image_path_to_new_image_id(img_folders_by_name: Dict[str, Path], target_folder: Path) -> Dict[str, int]:
    """Generates a map of old image it to new image ids"""
    num_im_processed = 1
    old_path_to_new_image_id: Dict[str, int] = {}
    for dataset_name, folder in img_folders_by_name.items():
        file_names = glob(str(folder / '*.jpg'))
        num_files = len(file_names)
        print(f'Found {num_files} files for dataset {dataset_name}')
        for file_name in tqdm(file_names):
            old_path_to_new_image_id[file_name] = num_im_processed
            full_target_file_name = str(target_folder /_image_id_to_file_name(image_id=num_im_processed))
            # shutil.copy(file_name, full_target_file_name)
            num_im_processed += 1
    return old_path_to_new_image_id
    
def _image_id_to_file_name(image_id: int) -> str:
    """Generates the file name for the new image from its id."""
    return f"{image_id}.jpg"

def _generate_region_segmentation_annotation_file(
    split: str, 
    anno_dict_by_name: Dict[str, str], 
    image_path_to_new_id: Dict[str, int],
    annotation_folder: Path,
) -> None:
    new_all_images_info = []
    new_anno_id = 1
    new_all_seg_anno_info = []
    out_metadata = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
    }
    image_ids_with_from_region_anno_but_no_anno = []
    for dataset_name, anno_file in anno_dict_by_name.items():
        region_annotation_dataset = False
        logging.info(f"Merging {dataset_name}.")
        
        with open(anno_file, "r") as file:
            current_metadata = json.load(file)
            old_id_to_new_id: Dict[int, int] = {}
            new_image_ids_in_current_set = []
            for image_info in current_metadata['images']:
                old_image_path = str(Path(_DATASETS_IMG_FOLDERS_BY_NAME[dataset_name]) / image_info['file_name'])
                new_image_id = image_path_to_new_id[old_image_path]
                old_id_to_new_id[image_info['id']] = new_image_id
                combined_image_info = image_info.copy()
                combined_image_info['id'] = new_image_id
                combined_image_info['file_name'] = _image_id_to_file_name(image_id=new_image_id)
                new_image_ids_in_current_set.append(new_image_id)
                new_all_images_info.append(combined_image_info)
    
            for anno_info in current_metadata['annotations']:
                if 'segmentation' in anno_info:
                    region_annotation_dataset = True
                    anno_info['id'] = new_anno_id
                    anno_info['image_id'] = old_id_to_new_id[anno_info['image_id']]
                    new_all_seg_anno_info.append(anno_info)
                    new_anno_id += 1    
            
            if region_annotation_dataset:
                annotated_image_ids = set(x['image_id'] for x in new_all_seg_anno_info)
                image_ids_with_from_region_anno_but_no_anno = set(new_image_ids_in_current_set).difference(annotated_image_ids)
            
            # TODO: add extra code to handle different image categories.
            if 'categories' in current_metadata:
                out_metadata['categories'] = current_metadata['categories']
            
    out_metadata['annotations'] = new_all_seg_anno_info
    # Don't save information of images without region annotations.
    out_metadata['images'] = [x for x in new_all_images_info if x['id'] not in image_ids_with_from_region_anno_but_no_anno]
    
    meta_json = json.dumps(out_metadata)
    with open(str(annotation_folder / f"{split}_instances.json"), "w") as json_file:
        json_file.write(meta_json)
            
def _create_dataset_info() -> Dict[str, Any]:
    """Generates a dictionary that contains the dataset information."""
    return {
        "description": "Merged VLM Dataset",
        "url": "",
        "version": "1.0",
        "year": 2024,
        "contributor": "",
        "date_created": "2024/07/14"
    }
    
def _create_dataset_license_info() -> List[Dict[str, Any]]:
    return [
        {
            "url": "",
            "id": 1,
            "name": "MIT License"
        }
    ]

def _generate_caption_annotation_files(
    split: str, 
    anno_dict_by_name: Dict[str, str], 
    image_path_to_new_id: Dict[str, int],
    annotation_folder: Path,
) -> None:
    all_datasets_images_info = []
    new_caption_id = 1
    all_datasets_captions_info = []
    out_metadata = {
        "info": _create_dataset_info(),
        "licenses": _create_dataset_license_info(),
    }
    for dataset_name, anno_file in anno_dict_by_name.items():
        logging.info(f"Merging {dataset_name}.")
        
        with open(anno_file, "r") as file:
            private_metadata = json.load(file)
            private_id_to_combined_id: Dict[int, int] = {}
            
            for private_image_info in private_metadata['images']:
                combined_image_id = image_path_to_new_id[str(Path(_DATASETS_IMG_FOLDERS_BY_NAME[dataset_name]) / private_image_info['file_name'])]
                private_id_to_combined_id[private_image_info['id']] = combined_image_id
                combined_image_info = {
                    'id': combined_image_id,
                    'file_name': _image_id_to_file_name(image_id=combined_image_id)
                }
                all_datasets_images_info.append(combined_image_info)
            
            if 'segmentation' in next(iter(private_metadata['annotations'])):
                synthetic_captions_by_image_id: Dict[int, List[str]] = defaultdict(list)
                # Generate synthetic captions from region class
                for private_anno_info in private_metadata['annotations']:
                    private_image_id = private_anno_info['image_id']
                    current_category = private_metadata['categories'][private_anno_info['category_id']]
                    synthetic_captions_by_image_id[private_image_id].append(f"{current_category['name'].lower()} {current_category['supercategory'].lower()}")
                
                synthetic_captions_by_image_id = {k: v for k, v in synthetic_captions_by_image_id.items() if len(v) > 0}
                
                for private_image_id, captions in synthetic_captions_by_image_id.items():    
                    combined_caption_info = {
                        'id': new_caption_id,
                        'image_id': private_id_to_combined_id[private_image_id],
                        'caption': ", ".join(f"{cap}" for cap in  captions)
                    }
                        
                    all_datasets_captions_info.append(combined_caption_info)
                    new_caption_id += 1
                    
            else:
                for private_anno_info in private_metadata['annotations']:                    
                    combined_caption_info = {
                        'id': new_caption_id,
                        'image_id': private_id_to_combined_id[private_anno_info['image_id']],
                        'caption': private_anno_info['caption'],
                    }
                    new_caption_id += 1
                    all_datasets_captions_info.append(combined_caption_info)
            
    # Filter all image with no captions
    combined_image_ids_with_captions = [cap_info['image_id'] for cap_info in all_datasets_captions_info]
    out_metadata['images'] = [x for x in all_datasets_images_info if x['id'] in combined_image_ids_with_captions]
    out_metadata['annotations'] = all_datasets_captions_info
    
    meta_json = json.dumps(out_metadata)
    with open(str(annotation_folder / f"{split}_captions.json"), "w") as json_file:
        json_file.write(meta_json)
        
if __name__ == "__main__":
    _merge_all()