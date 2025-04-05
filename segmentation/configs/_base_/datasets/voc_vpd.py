# configs/vpd/vpd_voc12.py

_base_ = [
    '../_base_/models/fpn_r50.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'

# Use VQGAN-style normalization
img_norm_cfg = dict(
    mean=[x * 255 for x in [0.5, 0.5, 0.5]],
    std=[x * 255 for x in [0.5, 0.5, 0.5]],
    to_rgb=True
)

crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
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
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val.txt',
        pipeline=test_pipeline)
)

# VPD-specific model config
model = dict(
    type='VPDSeg',
    sd_path='checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='./stable-diffusion/configs/stable-diffusion/v1-inference.yaml',
    neck=dict(
        type='FPN',
        in_channels=[320, 790, 1430, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=21,  # VOC has 21 classes including background
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

log_level = 'INFO'
