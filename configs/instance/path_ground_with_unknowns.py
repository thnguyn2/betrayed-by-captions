# For more information on how to configure MMDET, see https://mmdetection.readthedocs.io/en/v2.28.2/tutorials/config.html
_base_ = [
    '../_base_/default_runtime.py'
]

num_things_classes = 5 + 18
num_stuff_classes = 0
num_unknown_classes = 18
num_classes = num_things_classes + num_stuff_classes
num_known_classes = num_classes - num_unknown_classes

known_file = f'./datasets/unknown/path_ground_known_{num_classes}.txt'
unknown_file = f'./datasets/unknown/path_ground_unknown_{num_unknown_classes}.txt'
class_to_emb_file = f'./datasets/embeddings/quilt_class_with_pubmed_bert_emb.json'
embeding_type = 'pubmed-bert'
init_path = './pretrained/class_ag_pretrained_3x.pth'  # From class agnostic pretraining  # Class agnostic pretraining

model = dict(
    type='Mask2FormerOpen',  # Name of the model
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=3,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    panoptic_head=dict(
        type='Mask2FormerHeadOpen',
        in_channels=[256, 512, 1024, 2048],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_known_classes,
        num_stuff_classes=0,
        num_queries=100,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='MSDeformAttnPixelDecoder',
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=False,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        caption_generator=dict(
            type='CaptionTransformer',
            nb_layers=4,
            input_dim=768,
            hidden_dim=768,
            ff_dim=512,
            nb_heads=8,
            drop_val=0.1,
            pre_norm=False,
            seq_length=50,
            nb_tokens=30522),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.0,
            reduction='mean',
            class_weight=[1.0] * num_known_classes + [0.1]),
        loss_cls_emb=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * num_known_classes + [0.1]),
        loss_grounding=dict(
            type='GroundingLoss',
            loss_weight=2.0),
        loss_caption_generation=dict(
            type='CrossEntropyLoss',
            ignore_index=0,
            loss_weight=2.0),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        class_agnostic=False,
        use_caption=True,
        use_class_emb=True,
        use_caption_generation=True,
        class_to_emb_file=class_to_emb_file,
        known_file=known_file,
        unknown_file=unknown_file,
        softmax_temperature=10,
        pred_emb_norm=False,
        text_emb_norm=True,
        caption_emb_type=embeding_type,
        caption_gen_emb_type=embeding_type),
    panoptic_fusion_head=dict(
        type='MaskFormerFusionHeadOpen',
        num_things_classes=num_classes,
        num_stuff_classes=0,
        loss_panoptic=None,
        init_cfg=None,
        use_class_emb=True,
        class_to_emb_file=class_to_emb_file,
        known_file=known_file,
        unknown_file=unknown_file),
    
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(  # Assigner config
            type='MaskHungarianAssignerOpen',
            cls_cost=dict(type='ClassificationCost', weight=1.0),
            cls_emb_cost=dict(type='ClassificationCost', weight=2.0),
            mask_cost=dict(
                type='CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
            dice_cost=dict(
                type='DiceCost', weight=5.0, pred_act=True, eps=1.0)),
        sampler=dict(type='MaskPseudoSampler')
        ),
    
    test_cfg=dict(
        eval_types=['base_results'],
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.5,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True,
        use_class_emb=True),
    init_cfg=dict(type='Pretrained', checkpoint=init_path),  # Use the class agnostic pretraining since pretraining takes a long time.
)

# dataset settings
image_size = (1024, 1024)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadOpenAnnotations', with_bbox=True, with_mask=True, with_caption=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.8, 1.5),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='OpenFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks',
        'gt_caption_ids', 'gt_caption_mask', 'gt_caption_nouns_ids', 'gt_caption_nouns_mask']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'PathGroundOpenWithAllClasses'
data_root = '/jupyter-users-home/tan-2enguyen/datasets/pathology/anno_caption_merged/'

minibatch_size = 1
data = dict(
    _delete_=True,
    samples_per_gpu=minibatch_size,
    workers_per_gpu=minibatch_size,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_mixed/train_instances.json',
        caption_ann_file=data_root + 'annotations_mixed/train_captions.json',
        img_prefix=data_root + 'images/',
        transform_pipeline=train_pipeline,
        
        filter_empty_gt=False,
        known_file=known_file,
        unknown_file=unknown_file,
        class_agnostic=False,
        emb_type=embeding_type,
        use_reduced_size_dataset=True,
        ),
    
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_mixed/val_instances.json',
        caption_ann_file=data_root + 'annotations_mixed/val_captions.json',
        
        img_prefix=data_root + 'images/',
        transform_pipeline=test_pipeline,
        
        known_file=known_file,
        unknown_file=unknown_file,
        class_agnostic=False,
        eval_types=['base_results'],
        use_reduced_size_dataset=True,    
    ),
    
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations_mixed/val_instances.json',
        caption_ann_file=data_root + 'annotations_mixed/val_captions.json',
        img_prefix=data_root + 'images/',
        transform_pipeline=test_pipeline,
        
        known_file=known_file,
        unknown_file=unknown_file,
        class_agnostic=False,
        eval_types=['base_results'],  # We don't have the segmentation masks for novel classes!
        use_reduced_size_dataset=True,
        ),
    )

embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.00005,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat': embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    gamma=0.1,
    by_epoch=False,
    step=[70000, 80000],
    warmup='linear',
    warmup_by_epoch=False,
    warmup_ratio=1.0,  # no warmup
    warmup_iters=10)

runner = dict(
    type='EpochBasedRunner', 
    max_epochs=1000,
)

log_config = dict(
    interval=100,  # in the unit of iters, #iters = #images total / (mini batch size) * epoches
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook', by_epoch=False)
    ])
workflow = [('train', 1)]

checkpoint_config = dict(
    by_epoch=True, interval=1, save_last=True, max_keep_ckpts=2) 

evaluation = dict(
    interval=1,  # in the unit of epochs.
    metric=['bbox', 'segm'],
    classwise=True
)