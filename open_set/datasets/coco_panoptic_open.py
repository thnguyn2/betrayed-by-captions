# Copyright (c) OpenMMLab. All rights reserved.
import itertools
import os
from collections import defaultdict
import warnings

import torch
import clip

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import INSTANCE_OFFSET
from mmdet.datasets.api_wrappers import COCO
from ..utils.eval.pq_evaluation import pq_compute_multi_core
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.coco import CocoDataset
import transformers

from .utils.parser import LVISParser
from open_set.models.utils.bert_embeddings import BERT_MODEL_BY_EMBEDDING_TYPES

try:
    import panopticapi
    from panopticapi.evaluation import VOID
    from panopticapi.utils import id2rgb
except ImportError:
    panopticapi = None
    id2rgb = None
    VOID = None


class COCOPanoptic(COCO):
    """This wrapper is for loading the panoptic style annotation file.

    The format is shown in the CocoPanopticDataset class.

    Args:
        annotation_file (str): Path of annotation file.
    """

    def __init__(self, annotation_file=None):
        if panopticapi is None:
            raise RuntimeError(
                'panopticapi is not installed, please install it by: '
                'pip install git+https://github.com/cocodataset/'
                'panopticapi.git.')

        super(COCOPanoptic, self).__init__(annotation_file)

    def createIndex(self):
        # create index
        print('creating index...')
        # anns stores 'segment_id -> annotation'
        anns, cats, imgs = {}, {}, {}
        img_to_anns, cat_to_imgs = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann, img_info in zip(self.dataset['annotations'],
                                     self.dataset['images']):
                img_info['segm_file'] = ann['file_name']
                for seg_ann in ann['segments_info']:
                    # to match with instance.json
                    seg_ann['image_id'] = ann['image_id']
                    seg_ann['height'] = img_info['height']
                    seg_ann['width'] = img_info['width']
                    img_to_anns[ann['image_id']].append(seg_ann)
                    # segment_id is not unique in coco dataset orz...
                    if seg_ann['id'] in anns.keys():
                        anns[seg_ann['id']].append(seg_ann)
                    else:
                        anns[seg_ann['id']] = [seg_ann]

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for seg_ann in ann['segments_info']:
                    cat_to_imgs[seg_ann['category_id']].append(ann['image_id'])

        print('index created!')

        self.anns = anns
        self.imgToAnns = img_to_anns
        self.catToImgs = cat_to_imgs
        self.imgs = imgs
        self.cats = cats

    def load_anns(self, ids=[]):
        """Load anns with the specified ids.

        self.anns is a list of annotation lists instead of a
        list of annotations.

        Args:
            ids (int array): integer ids specifying anns

        Returns:
            anns (object array): loaded ann objects
        """
        anns = []

        if hasattr(ids, '__iter__') and hasattr(ids, '__len__'):
            # self.anns is a list of annotation lists instead of
            # a list of annotations
            for id in ids:
                anns += self.anns[id]
            return anns
        elif type(ids) == int:
            return self.anns[ids]

@DATASETS.register_module()
class CocoPanopticDatasetOpen(CocoDataset):
    """Coco dataset for Open-Set Panoptic segmentation.

    The annotation format is shown as follows. The `ann` field is optional
    for testing.

    .. code-block:: none

        [
            {
                'filename': f'{image_id:012}.png',
                'image_id':9
                'segments_info': {
                    [
                        {
                            'id': 8345037, (segment_id in panoptic png,
                                            convert from rgb)
                            'category_id': 51,
                            'iscrowd': 0,
                            'bbox': (x1, y1, w, h),
                            'area': 24315,
                            'segmentation': list,(coded mask)
                        },
                        ...
                    }
                }
            },
            ...
        ]

    Args:
        ann_file (str): Panoptic segmentation annotation file path.
        pipeline (list[dict]): Processing pipeline.
        ins_ann_file (str): Instance segmentation annotation file path.
            Defaults to None.
        classes (str | Sequence[str], optional): Specify classes to load.
            If is None, ``cls.CLASSES`` will be used. Defaults to None.
        data_root (str, optional): Data root for ``ann_file``,
            ``ins_ann_file`` ``img_prefix``, ``seg_prefix``, ``proposal_file``
            if specified. Defaults to None.
        img_prefix (str, optional): Prefix of path to images. Defaults to ''.
        seg_prefix (str, optional): Prefix of path to segmentation files.
            Defaults to None.
        proposal_file (str, optional): Path to proposal file. Defaults to None.
        test_mode (bool, optional): If set True, annotation will not be loaded.
            Defaults to False.
        filter_empty_gt (bool, optional): If set true, images without bounding
            boxes of the dataset's classes will be filtered out. This option
            only works when `test_mode=False`, i.e., we never filter images
            during tests. Defaults to True.
        file_client_args (:obj:`mmcv.ConfigDict` | dict): file client args.
            Defaults to dict(backend='disk').
    """
    CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff',
        'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light',
        'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]
    THING_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    STUFF_CLASSES = [
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain',
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield',
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile',
        'wall-wood', 'water-other', 'window-blind', 'window-other',
        'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged',
        'cabinet-merged', 'table-merged', 'floor-other-merged',
        'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged',
        'paper-merged', 'food-other-merged', 'building-other-merged',
        'rock-merged', 'wall-other-merged', 'rug-merged'
    ]

    PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
               (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
               (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
               (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
               (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
               (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
               (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
               (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
               (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
               (134, 134, 103), (145, 148, 174), (255, 208, 186),
               (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
               (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
               (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149),
               (179, 0, 194), (209, 99, 106), (5, 121, 0), (227, 255, 205),
               (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0),
               (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88),
               (95, 32, 0), (130, 114, 135), (110, 129, 133), (166, 74, 118),
               (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15),
               (127, 167, 115), (59, 105, 106), (142, 108, 45), (196, 172, 0),
               (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122),
               (191, 162, 208), (255, 255, 128), (147, 211, 203),
               (150, 100, 100), (168, 171, 172), (146, 112, 198),
               (210, 170, 100), (92, 136, 89), (218, 88, 184), (241, 129, 0),
               (217, 17, 255), (124, 74, 181), (70, 70, 70), (255, 228, 255),
               (154, 208, 0), (193, 0, 92), (76, 91, 113), (255, 180, 195),
               (106, 154, 176),
               (230, 150, 140), (60, 143, 255), (128, 64, 128), (92, 82, 55),
               (254, 212, 124), (73, 77, 174), (255, 160, 98), (255, 255, 255),
               (104, 84, 109), (169, 164, 131), (225, 199, 255), (137, 54, 74),
               (135, 158, 223), (7, 246, 231), (107, 255, 200), (58, 41, 149),
               (183, 121, 142), (255, 73, 97), (107, 142, 35), (190, 153, 153),
               (146, 139, 141),
               (70, 130, 180), (134, 199, 156), (209, 226, 140), (96, 36, 108),
               (96, 96, 96), (64, 170, 64), (152, 251, 152), (208, 229, 228),
               (206, 186, 171), (152, 161, 64), (116, 112, 0), (0, 114, 143),
               (102, 102, 156), (250, 141, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 ins_ann_file=None,
                 classes=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 filter_empty_gt=True,
                 known_file=None,
                 unknown_file=None,
                 eval_types=[],
                 file_client_args=dict(backend='disk'),
                 class_agnostic=False,
                 emb_type='bert',
                 caption_ann_file=None,
                 ann_sample_rate=1.0,
                 max_ann_per_image=100,
                 use_reduced_size_dataset: bool=False,
                 ):
        self.known_file = known_file
        self.unknown_file = unknown_file
        self.class_agnostic = class_agnostic

        self.caption_ann_file = caption_ann_file
        self.emb_type = emb_type
        self.eval_types = eval_types
        self.ann_sample_rate = ann_sample_rate
        self.max_ann_per_image = max_ann_per_image

        self.file_client_args = file_client_args
        self._use_reduced_size_dataset = use_reduced_size_dataset

        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
            file_client_args=file_client_args,
        )
        self.ins_ann_file = ins_ann_file

        if self.caption_ann_file is not None:
            self.coco_caption = COCO(self.caption_ann_file)
            self.max_tokens = 35
            self.tokenizer = transformers.BertTokenizer.from_pretrained(BERT_MODEL_BY_EMBEDDING_TYPES[emb_type])
        self.parser = LVISParser()

    def load_annotations(self, ann_file):
        """Load annotation from COCO Panoptic style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCOPanoptic(ann_file)
        self.cat_ids = self.coco.get_cat_ids()
        
        self.file_client = mmcv.FileClient(**self.file_client_args)

        # filter unknown cat_ids
        self.all_cat_ids = self.cat_ids
        self.unknown_cat_ids = []
        if self.unknown_file is not None:
            unknown_cat_names = self.file_client.get_text(self.unknown_file).split('\n')
            unknown_cat_ids = self.coco.get_cat_ids(cat_names=unknown_cat_names)
            self.unknown_cat_ids = [id for id in self.cat_ids if id in unknown_cat_ids]

        self.known_cat_ids = [id for id in self.cat_ids if id not in self.unknown_cat_ids]
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.known_cat_ids)}
        self.categories = self.coco.cats

        self.img_ids = self.coco.get_img_ids()
        if self._use_reduced_size_dataset:
            self.img_ids = self.img_ids[:500]

        data_infos = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            info['segm_file'] = info['filename'].replace('jpg', 'png')
            data_infos.append(info)
        return data_infos

    def _filter_imgs(self, min_size=32):
        """Filter images too small."""
        if self.filter_empty_gt:
            warnings.warn(
                'CustomDataset does not support filtering empty gt images.')
        valid_inds = []
        for i, img_info in enumerate(self.data_infos):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        data_info = self.data_infos[idx].copy()
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        # filter out unmatched images
        ann_info = [i for i in ann_info if i['image_id'] == img_id]

        # add coco captions
        if self.coco_caption is not None:
            caption_ann_ids = self.coco_caption.get_ann_ids(img_ids=[img_id])
            caption_ann_info = self.coco_caption.load_anns(caption_ann_ids)
            # During training, randomly choose a caption as gt.
            random_idx = np.random.randint(0, len(caption_ann_info))
            caption = caption_ann_info[random_idx]["caption"]
            unique_object_nouns = self.extract_obj(caption)
            data_info["caption"] = caption
            data_info["caption_nouns"] = " ".join(unique_object_nouns)

        return self._parse_ann_info(data_info, ann_info)

    def extract_obj(self, sentence):
        unique_nns = []
        nns, category_ids = self.parser.parse(sentence)
        unique_nns.extend(nns)
        unique_nns = list(set(unique_nns))
        return unique_nns

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and load panoptic ground truths.

        Args:
            img_info (int): Image info of an image.
            ann_info (list[dict]): Annotation info of an image.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_mask_infos = []

        for i, ann in enumerate(ann_info):
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            category_id = ann['category_id']
            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            is_unknown = category_id in self.unknown_cat_ids

            if not is_unknown:
                if self.class_agnostic:
                    # 0 for things, 1 for stuff
                    contiguous_cat_id = 1 - is_thing
                else:
                    contiguous_cat_id = self.cat2label[category_id]
                if is_thing:
                    is_crowd = ann.get('iscrowd', False)
                    if not is_crowd:
                        gt_bboxes.append(bbox)
                        gt_labels.append(contiguous_cat_id)
                    else:
                        gt_bboxes_ignore.append(bbox)
                        is_thing = False
                mask_info = {
                    'id': ann['id'],
                    'category': contiguous_cat_id,
                    'is_thing': is_thing
                }
                gt_mask_infos.append(mask_info)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        if self.coco_caption is not None:
            padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask = self.parse_caption(img_info)
        else:
            padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask = None, None, None, None

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=img_info['segm_file'],
            caption_ids=padded_ids,
            caption_mask=attention_mask,
            caption_nouns_ids=padded_nouns_ids,
            caption_nouns_mask=attention_nouns_mask
        )

        return ann

    def parse_caption(self, img_info):
        caption_str = img_info['caption']
        caption_nouns = img_info['caption_nouns']
        padded_ids = [0] * self.max_tokens
        attention_mask = [0] * self.max_tokens
        padded_nouns_ids = [0] * self.max_tokens
        attention_nouns_mask = [0] * self.max_tokens
        if self.emb_type == 'bert':
            caption_ids = self.tokenizer.encode(caption_str, add_special_tokens=True)
            caption_ids = caption_ids[:self.max_tokens]
            padded_ids[:len(caption_ids)] = caption_ids
            attention_mask[:len(caption_ids)] = [1] * len(caption_ids)
            caption_nouns_ids = self.tokenizer.encode(caption_nouns, add_special_tokens=False)
            caption_nouns_ids = caption_nouns_ids[:self.max_tokens]
            padded_nouns_ids[:len(caption_nouns_ids)] = caption_nouns_ids
            attention_nouns_mask[:len(caption_nouns_ids)] = [1] * len(caption_nouns_ids)
        elif self.emb_type == 'clip':
            padded_ids = clip.tokenize(caption_str).numpy()
            attention_mask = padded_ids > 0
            padded_nouns_ids = torch.cat([clip.tokenize(f'A photo of a {noun}') for noun in caption_nouns.split(' ')], dim=0).numpy()
            attention_nouns_mask = [0] * padded_nouns_ids.shape[1]
            attention_nouns_mask[:padded_nouns_ids.shape[0]] = [1] * padded_nouns_ids.shape[0]
        elif self.emb_type == 'bert-clip':
            caption_ids = self.tokenizer.encode(caption_str, add_special_tokens=True)
            caption_ids = caption_ids[:self.max_tokens]
            padded_ids[:len(caption_ids)] = caption_ids
            attention_mask[:len(caption_ids)] = [1] * len(caption_ids)
            padded_nouns_ids = torch.cat([clip.tokenize(f'A photo of a {noun}') for noun in caption_nouns.split(' ')], dim=0).numpy()
            attention_nouns_mask = [0] * padded_nouns_ids.shape[1]
            attention_nouns_mask[:padded_nouns_ids.shape[0]] = [1] * padded_nouns_ids.shape[0]

        return padded_ids, attention_mask, padded_nouns_ids, attention_nouns_mask


    def _pan2json(self, results, outfile_prefix):
        pred_annotations = []
        outdir = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            segm_file = self.data_infos[idx]['segm_file']
            pan = results[idx]

            pan_labels = np.unique(pan)
            segm_info = []
            for pan_label in pan_labels:
                sem_label = pan_label % INSTANCE_OFFSET
                # exclude VOID label
                if sem_label == len(self.all_cat_ids):
                    continue
                # convert sem_label to json label
                cat_id = self.all_cat_ids[sem_label]
                is_thing = self.categories[cat_id]['isthing']
                mask = pan == pan_label
                area = mask.sum()
                segm_info.append({
                    'id': int(pan_label),
                    'category_id': cat_id,
                    'isthing': is_thing,
                    'area': int(area)
                })
            # evaluation script uses 0 for VOID label.
            pan[pan % INSTANCE_OFFSET == len(self.all_cat_ids)] = VOID
            pan = id2rgb(pan).astype(np.uint8)
            mmcv.imwrite(pan[:, :, ::-1], os.path.join(outdir, segm_file))
            record = {
                'image_id': img_id,
                'segments_info': segm_info,
                'file_name': segm_file
            }
            pred_annotations.append(record)
        pan_json_results = dict(annotations=pred_annotations)
        return pan_json_results

    def results2json(self, results, outfile_prefix):
        result_files = dict()
        pan_json_results = self._pan2json(results, outfile_prefix)
        result_files['panoptic'] = f'{outfile_prefix}.panoptic.json'
        mmcv.dump(pan_json_results, result_files['panoptic'])

        return result_files

    def evaluate_pan_json(self,
                          result_files,
                          outfile_prefix,
                          logger=None,
                          classwise=False,
                          nproc=32):
        """Evaluate PQ according to the panoptic results json file."""
        imgs = self.coco.imgs
        gt_json = self.coco.img_ann_map  # image to annotations
        gt_json = [{
            'image_id': k,
            'segments_info': v,
            'file_name': imgs[k]['segm_file']
        } for k, v in gt_json.items()]
        pred_json = mmcv.load(result_files['panoptic'])
        pred_json = dict(
            (el['image_id'], el) for el in pred_json['annotations'])

        # match the gt_anns and pred_anns in the same image
        matched_annotations_list = []
        for gt_ann in gt_json:
            img_id = gt_ann['image_id']
            if img_id not in pred_json.keys():
                raise Exception('no prediction for the image'
                                ' with id: {}'.format(img_id))
            matched_annotations_list.append((gt_ann, pred_json[img_id]))

        gt_folder = self.seg_prefix
        pred_folder = os.path.join(os.path.dirname(outfile_prefix), 'panoptic')

        pq_stat = pq_compute_multi_core(
            matched_annotations_list,
            gt_folder,
            pred_folder)

        metrics = [("All", None, None), ("Known Things", True, False), ("Unknown Things", True, True), ("Stuff", False, None)]
        pq_results = {}

        for name, isthing, isunknown in metrics:
            pq_results[name], classwise_results = \
                pq_stat.pq_average(self.categories, isthing=isthing, isunknown=isunknown, unknown_cat_ids=self.unknown_cat_ids)
            if name == 'All':
                pq_results['classwise'] = classwise_results

        classwise_results = None
        if classwise:
            OPEN_CLASSES = []
            for name in self.CLASSES:
                cat_id = self.coco.get_cat_ids(cat_names=[name])[0]
                if cat_id in self.unknown_cat_ids:
                    name = '*' + name
                OPEN_CLASSES.append(name)
            classwise_results = {
                k: v
                for k, v in zip(OPEN_CLASSES, pq_results['classwise'].values())
            }
        print_panoptic_table(pq_results, classwise_results, logger=logger)
        pq_results = parse_pq_results(pq_results)
        pq_results['PQ_copypaste'] = (
            f'{pq_results["PQ"]:.3f} {pq_results["SQ"]:.3f} '
            f'{pq_results["RQ"]:.3f} '
            f'{pq_results["PQ_kth"]:.3f} {pq_results["SQ_kth"]:.3f} '
            f'{pq_results["RQ_kth"]:.3f} '
            f'{pq_results["PQ_ukth"]:.3f} {pq_results["SQ_ukth"]:.3f} '
            f'{pq_results["RQ_ukth"]:.3f} '
            f'{pq_results["PQ_st"]:.3f} {pq_results["SQ_st"]:.3f} '
            f'{pq_results["RQ_st"]:.3f}')

        return pq_results

    def evaluate(self,
                 results,
                 metric='PQ',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 nproc=32,
                 **kwargs):
        """Evaluation in COCO Panoptic protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. 'PQ', 'bbox',
                'segm', 'proposal' are supported. 'pq' will be regarded as 'PQ.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to print classwise evaluation results.
                Default: False.
            nproc (int): Number of processes for panoptic quality computing.
                Defaults to 32. When `nproc` exceeds the number of cpu cores,
                the number of cpu cores is used.

        Returns:
            dict[str, float]: COCO Panoptic style evaluation metric.
        """
        metrics = metric if isinstance(metric, list) else [metric]
        # Compatible with lowercase 'pq'
        metrics = ['PQ' if metric == 'pq' else metric for metric in metrics]
        allowed_metrics = ['PQ']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        if self.eval_types == []:
            self.eval_types = ['panoptic_all_results']
        for eval_type in self.eval_types:
            cur_results = [result[eval_type] for result in results]

            result_files, tmp_dir = self.format_results(cur_results, jsonfile_prefix)
            outfile_prefix = os.path.join(tmp_dir.name, 'results') \
                if tmp_dir is not None else jsonfile_prefix
            eval_pan_results = self.evaluate_pan_json(
                result_files, outfile_prefix, logger, classwise, nproc=nproc)

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return eval_pan_results


def parse_pq_results(pq_results):
    """Parse the Panoptic Quality results."""
    result = dict()
    result['PQ'] = 100 * pq_results['All']['pq']
    result['SQ'] = 100 * pq_results['All']['sq']
    result['RQ'] = 100 * pq_results['All']['rq']
    result['PQ_kth'] = 100 * pq_results['Known Things']['pq']
    result['SQ_kth'] = 100 * pq_results['Known Things']['sq']
    result['RQ_kth'] = 100 * pq_results['Known Things']['rq']
    result['PQ_ukth'] = 100 * pq_results['Unknown Things']['pq']
    result['SQ_ukth'] = 100 * pq_results['Unknown Things']['sq']
    result['RQ_ukth'] = 100 * pq_results['Unknown Things']['rq']
    result['PQ_st'] = 100 * pq_results['Stuff']['pq']
    result['SQ_st'] = 100 * pq_results['Stuff']['sq']
    result['RQ_st'] = 100 * pq_results['Stuff']['rq']
    return result

def print_panoptic_table(pq_results, classwise_results=None, logger=None):
    """Print the panoptic evaluation results table.

    Args:
        pq_results(dict): The Panoptic Quality results.
        classwise_results(dict | None): The classwise Panoptic Quality results.
            The keys are class names and the values are metrics.
        logger (logging.Logger | str | None): Logger used for printing
            related information during evaluation. Default: None.
    """

    headers = ['', 'PQ', 'SQ', 'RQ', 'Precision', 'Recall', 'categories']
    data = [headers]
    for name in ['All', 'Known Things', 'Unknown Things', 'Stuff']:
        numbers = [
            f'{(pq_results[name][k] * 100):0.3f}' for k in ['pq', 'sq', 'rq', 'precision', 'recall']
        ]
        row = [name] + numbers + [pq_results[name]['n']]
        data.append(row)
    table = AsciiTable(data)
    print_log('Panoptic Evaluation Results:\n' + table.table, logger=logger)

    if classwise_results is not None:
        class_metrics = [(name, ) + tuple(f'{(metrics[k] * 100):0.3f}'
                                          for k in ['pq', 'precision', 'recall'])
                         for name, metrics in classwise_results.items()]
        num_columns = min(8, len(class_metrics) * 4)
        results_flatten = list(itertools.chain(*class_metrics))
        headers = ['category', 'PQ', 'P', "R"] * (num_columns // 4)
        results_2d = itertools.zip_longest(
            *[results_flatten[i::num_columns] for i in range(num_columns)])
        data = [headers]
        data += [result for result in results_2d]
        table = AsciiTable(data)
        print_log(
            'Classwise Panoptic Evaluation Results:\n' + table.table,
            logger=logger)
