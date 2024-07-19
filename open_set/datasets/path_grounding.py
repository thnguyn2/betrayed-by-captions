"""A module that defines a custom dataset for pathology grounding."""
import mmcv
import numpy as np
from mmdet.datasets.builder import DATASETS
from typing import Dict, List, Optional
from mmdet.datasets.api_wrappers import COCO
from .coco_open import CocoDatasetOpen
from open_set.models.utils.bert_embeddings import BERT_MODEL_BY_EMBEDDING_TYPES

@DATASETS.register_module()
class PathGroundOpen(CocoDatasetOpen):
    """A dataset that generates other the segmentation mask for grounding or captioning data.
    
    Args:
        ann_file: The path to the file that holds the annotation data.
        caption_ann_file: The path to the json file of caption data.
        transform_pipeline: A list of transformations to be applied to the data on the dataset.
        classes (optional): A list of classes to be specified on the dataset. Defaults to None.
            in which case ``cls.CLASSES`` will be used. 
        data_root (optional): A path to the folder that contains the data. Defaults to None.
        anno_img_prefix (optional): The path to files with annotations. Defaults to ''.
        caption_img_prefix (optional): The path to files with captions. Defaults to ''.
        seg_prefix (optional): A string that specifies the prefix for the segmentation file name.
        test_mode (optional): If set True, annotation will not be loaded. Defaults to False.
        filter_empty_gt (optional): If set true, images without bounding boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images during tests.
        file_client_args (optional): The file client.
        known_file (optional): The path to a file of known classes. Defaults to None.
        unknown_file (optional): The path to a file to store the name of unknown classes. Defaults to None.
        class_agnostic (optional): If True, train a class agnositic model. Defaults to False.
        emb_type (optional): The type of embeddings. Defaults to `bert`.
        use_reduced_size_dataset (optional): If True, use a smaller dataset for fast debugging. Defaults to False.
    Reference:
        https://www.youtube.com/watch?v=jftZBfMZj8k
    """
    CLASSES =  ('neoplastic', 'inflammatory', 'connective', 'dead', 'epithelial')
    def __init__(
        self,
        ann_file: str,
        caption_ann_file: str,
        transform_pipeline: List[mmcv.utils.config.ConfigDict],
        classes: Optional[List[str]]=None,
        data_root: Optional[str] = None,
        img_prefix='',
        seg_prefix: Optional[str] =None,
        proposal_file: Optional[str] =None,
        test_mode: bool =False,
        filter_empty_gt: bool =True,
        file_client_args: Dict = {'backend': 'disk'},
        known_file: Optional[str]=None,
        unknown_file: Optional[str]=None,
        class_agnostic: bool=False,
        eval_types: List =[],
        ann_sample_rate: float=1.0,
        max_ann_per_image: int=100,
        nouns_parser: str='med_lvis',
        emb_type='bert',
        use_reduced_size_dataset: bool=False,
    ) -> None:
        super().__init__(
            ann_file=ann_file,
            pipeline=transform_pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args,
            known_file=known_file,
            unknown_file=unknown_file,
            class_agnostic=class_agnostic,
            emb_type=emb_type,
            caption_ann_file=caption_ann_file,
            eval_types=eval_types,
            ann_sample_rate=ann_sample_rate,
            max_ann_per_image=max_ann_per_image,
            nouns_parser=nouns_parser,
            use_reduced_size_dataset=use_reduced_size_dataset,
        )
    
    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        data_info = self.data_infos[idx].copy()
        img_id = data_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        if self.coco_caption is not None:
            caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[img_id])
            caption_ann_info = self.coco_caption.load_anns(caption_ann_ids)
            # During training, randomly choose a caption as gt.
            random_idx = np.random.randint(0, len(caption_ann_info))
            caption = caption_ann_info[random_idx]["caption"]
            data_info["caption"] = caption
            data_info["caption_nouns"] = " ".join(self._extract_nouns_from_caption(caption))
        return self._parse_ann_info(data_info, ann_info)
    
    
