# dataset settings


dataset_type = 'ADE20KDataset'
data_root = '/work3/s203520/advanced_computer_vision/filtered_dataset'

# use the normalization as the VQ-GAN in Stable-Diffusion
IMG_MEAN = [v * 255 for v in [0.5, 0.5, 0.5]]
IMG_VAR = [v * 255 for v in [0.5, 0.5, 0.5]]

img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)
crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(
#         type='LoadPerClassMasksFromFolder',
#         mask_root='/work3/s203520/advanced_computer_vision/filtered_dataset/prompt_masks',
#         types=['box', 'scribble', 'dot'],
#         suffix='.npy',
#         random_select = True
#     ),
#     dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
#     dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='Normalize', **img_norm_cfg),
#     dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_bbox_masks'],    meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'input_type')),
# ]
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='LoadPerClassMasksFromFolder',
        mask_root='/work3/s203520/advanced_computer_vision/filtered_dataset/prompt_masks',
        types=['box'],
        suffix='.npy',
        random_select=True,
    ),
    dict(type='ResizeWithBBox', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCropWithBBox', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlipWithBBox', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='PadToSizeWithBBox', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_bbox_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'input_type'),
    ),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
                dict(
                type='LoadPerClassMasksFromFolder',
                mask_root='/work3/s203520/advanced_computer_vision/filtered_dataset/prompt_masks',
                suffix='.npy',
                types=['box'],
                random_select=True,
            ),
            dict(type='ResizeWithBBox', keep_ratio=False),
            dict(type='RandomFlipWithBBox', prob=0.0),
            dict(type='Normalize', **img_norm_cfg),

            dict(type='ImageToTensor', keys=['img']),  # masks already tensors
            dict(
                type='Collect',
                keys=['img', 'gt_bbox_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'input_type'),
            ),
        ],
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
                dict(
                type='LoadPerClassMasksFromFolder',
                mask_root='/work3/s203520/advanced_computer_vision/filtered_dataset/prompt_masks',
                suffix='.npy',
                types=['box'],
                random_select=True,
            ),
            dict(type='ResizeWithBBox', keep_ratio=False),
            dict(type='RandomFlipWithBBox', prob=0.0),
            dict(type='Normalize', **img_norm_cfg),

            dict(type='ImageToTensor', keys=['img']),  # masks already tensors
            dict(
                type='Collect',
                keys=['img', 'gt_bbox_masks'],
                meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'input_type'),
            ),
        ],
    ),
]

# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(512, 512),
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=False),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(
#                 type='LoadPerClassMasksFromFolder',
#                 mask_root='/work3/s203520/advanced_computer_vision/filtered_dataset/prompt_masks',
#                 suffix='.npy',
#                 types=['box', 'scribble', 'dot'],
#                 random_select = False,
#             ),
#             dict(type='ImageToTensor', keys=['img']),  # masks already tensors
#             dict(type='Collect', keys=['img', 'gt_bbox_masks'],
#                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor'))
#             ])]



data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type="CustomDatasetWithClassFilter",
        class_filter = [42],
        data_root=data_root,
        img_dir='images/training',
        ann_dir='annotations/training',
        json_path = "/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/datasets/validation_class_info.json",

        pipeline=train_pipeline),
    val=dict(
        type="CustomDatasetWithClassFilter",
        class_filter = [42],
        data_root=data_root,
        img_dir='images/validation',
        ann_dir='annotations/validation',
        json_path = "/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/datasets/validation_class_info.json",
        pipeline=val_pipeline,
        ),
    test=dict(
        type="CustomDatasetWithClassFilter",
        class_filter = [92 ,95 ,146 ,69 ,138 ,116 ,148 ,96 ,122 ,61 ,149 ,142 ,102 ,124 ,94 ,53 ,107 ,105 ,109, 132, 85, 42, 54, 60],
        data_root=data_root,
        img_dir='images/validation/',
        ann_dir='annotations/validation/',
        json_path = "/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/datasets/validation_class_info.json",
        pipeline=test_pipeline))

