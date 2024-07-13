"""A module that defines a custom dataset for pathology grounding."""
import mmcv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from typing import Dict, List, Optional

class PathGroundOpen(CustomDataset):
    """A dataset that generates other the segmentation mask for grounding or captioning data.
    
    Args:
        ann_file: The path to the file that holds the annotation data.
        transform_pipeline: A list of transformations to be applied to the data on the dataset.
        classes (optional): A list of classes to be specified on the dataset. Defaults to None.
            in which case ``cls.CLASSES`` will be used. 
        data_root (optional): A path to the folder that contains the data. Defaults to None.
        img_prefix (optional): A string that specifies the file name prefix, normally used to specify the path to the files. Defaults to ''.
        seg_prefix (optional): A string that specifies the prefix for the segmentation file name.
        test_mode (optional): If set True, annotation will not be loaded. Defaults to False.
        filter_empty_gt (optional): If set true, images without bounding boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images during tests.
        file_client_args (optional): The file client.
        
    Reference:
        https://www.youtube.com/watch?v=jftZBfMZj8k
    """
    def __init__(
        self,
        ann_file: str,
        transform_pipeline: List[mmcv.utils.config.ConfigDict],
        classes: Optional[List[str]]=None,
        data_root: Optional[str] = None,
        img_prefix: str ='',
        seg_prefix: Optional[str] =None,
        proposal_file: Optional[str] =None,
        test_mode: bool =False,
        filter_empty_gt: bool =True,
        file_client_args: Dict = {'backend': 'disk'},
    ) -> None:
        super().__init__(
            ann_file,
            transform_pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args
        )