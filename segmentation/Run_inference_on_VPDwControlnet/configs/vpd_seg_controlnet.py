_base_ = [
    './_base_/datasets/voc12_controlnet.py',  # customize to use bbox maps
    './_base_/default_runtime.py',
    './_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    type='EncoderDecoder',
    backbone=dict(  
        type='VPDSeg',
        controlnet=dict(
            type='MultiScaleControlNet',
            in_channels=1,  # assuming grayscale bbox map
            base_channels=64
        ),
        unet_wrapper_cfg=dict(
            use_attn=True,
            base_size=512,
            attn_selector='down_cross+up_cross'
        ),
        unet_ckpt='path/to/vpd_checkpoint.pth',
        freeze_unet=True
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=320,
        in_index=0,
        channels=256,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=21,  # VOC has 21 classes (including background)
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)

# dataset settings
dataset_type = 'PascalVOCDatasetWithBBoxes'  # Your custom dataset class
data_root = 'data/VOCdevkit/VOC2012/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='LoadBboxMapFromFile'),  # Custom transform to load bbox maps
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundleWithBboxMap'),  # Custom formatter
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'bbox_map']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=train_pipeline)
)

# optimizer
optimizer = dict(type='AdamW', lr=6e-5, weight_decay=0.01)

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-6,
    by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=20000)

checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU')
work_dir = "./workdirs/"