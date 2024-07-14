"""A module that defines a custom dataset for pathology grounding."""
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from typing import Dict, List, Optional
from mmdet.datasets.api_wrappers import COCO

@DATASETS.register_module()
class PathGroundOpen(CustomDataset):
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
    Reference:
        https://www.youtube.com/watch?v=jftZBfMZj8k
    """
    CLASSES =  ('Neoplastic cells', 'Inflammatory', 'Connective/Soft tissue cells', 'Dead Cells', 'Epithelial')
    def __init__(
        self,
        ann_file: str,
        caption_ann_file: str,
        transform_pipeline: List[mmcv.utils.config.ConfigDict],
        classes: Optional[List[str]]=None,
        data_root: Optional[str] = None,
        anno_img_prefix='',
        caption_img_prefix='',
        seg_prefix: Optional[str] =None,
        proposal_file: Optional[str] =None,
        test_mode: bool =False,
        filter_empty_gt: bool =True,
        file_client_args: Dict = {'backend': 'disk'},
        known_file: Optional[str]=None,
        unknown_file: Optional[str]=None,
        class_agnostic: bool=False,
        emb_type: str ='bert',
        eval_types: List =[],
        ann_sample_rate: float=1.0,
        max_ann_per_image: int=100,
        nouns_parser: str='lvis'
    ) -> None:
        self._file_client_args = file_client_args
        self._known_file = known_file
        self._unknown_file = unknown_file
        super().__init__(
            ann_file,
            transform_pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=anno_img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args
        )
        
    def load_annotations(self, ann_file):
        """Load annotation from the image with annotations.

        Args:
            ann_file (str): Path of annotation file of the region-annotation dataset.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self._anno_coco = COCO(ann_file)
        self._caption_coco = None
        # The order of returned `cat_ids` will not
        # change with the order of the CLASSES
        self._cat_ids = self._anno_coco.get_cat_ids(cat_names=self.CLASSES)

        # known and unknown classes
        self._file_client = mmcv.FileClient(**self._file_client_args)
        if self._known_file is not None:
            all_cat_names = self._file_client.get_text(self._known_file).split('\n')
            self._all_cat_ids = [id for id in self._cat_ids if id in self._anno_coco.get_cat_ids(cat_names=all_cat_names)]
                
        if self._unknown_file is not None:
            unknown_cat_names = self._file_client.get_text(self._unknown_file).split('\n')
            self._unknown_cat_ids = [id for id in self._cat_ids if id in self._anno_coco.get_cat_ids(cat_names=unknown_cat_names)]

        self._known_cat_ids = [id for id in self._cat_ids if id in self._cat_ids and id not in self._unknown_cat_ids]

        self._cat2label = {cat_id: i for i, cat_id in enumerate(self._known_cat_ids)}

        self._img_ids = self._anno_coco.get_img_ids()
     
        data_infos = []
        total_ann_ids = []
        for i in self._img_ids:
            info = self._anno_coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self._anno_coco.get_ann_ids(img_ids=[i])
            if self._caption_coco is not None:
                caption_ann_ids = self._caption_coco.get_ann_ids(img_ids=[i])
                assert len(caption_ann_ids) > 0, f"All anns should have a caption ann."
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
    
        return data_infos