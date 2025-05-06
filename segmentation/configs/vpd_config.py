
_base_ = [  
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k_vpd.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_20k.py'
]

custom_imports = dict(
    imports=['segmentation.hooks.visualize_hook',"segmentation"],  # adjust path if needed
    allow_failed_imports=False
)
model = dict(
    type='VPDSeg',
    sd_path='/work3/s203557/checkpoints/v1-5-pruned-emaonly.ckpt',
    sd_config='/zhome/b6/d/154958/ADLCV_Project/VPD/segmentation/v1-inference.yaml',
    max_boxes = 6,
    neck=dict(
        type='FPN',
        in_channels=[320, 790, 1430, 1280],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

lr_config = dict(policy='poly', power=1, min_lr=0.0, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


# optimizer = dict(type='AdamW', lr=0.00008, weight_decay=0.001,
#         paramwise_cfg=dict(custom_keys={'unet.trainable_unet': dict(lr_mult=0.1),
#                                         'unet.encoder_vq': dict(lr_mult=0.0),
#                                         "unet.zero_convs": dict(lr_mult=1),
#                                         "unet.box_encoder": dict(lr_mult=1),
#                                         'text_encoder': dict(lr_mult=0.0),
#                                         'norm': dict(decay_mult=0.)
#                                         }))
                                        


optimizer = dict(type='AdamW', lr=0.00008, weight_decay=0.001,
        paramwise_cfg=dict(custom_keys={

                                        #Trainable Decoding
                                        'neck': dict(lr_mult=0.),
                                        "decode_head": dict(lr_mult=0.),
                                        
                                        #Frozen 
                                        'unet.unet': dict(lr_mult=0.),
                                        'unet': dict(lr_mult=0.),
                                        "backbone": dict(lr_mult=0.),
                                        'unet.encoder_vq': dict(lr_mult=0.0),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.),
                                        "gamma": dict(lr_mult=0.),

                                        ##Trainable Control Components
                                        'unet.trainable_unet': dict(lr_mult=1.),
                                        'trainable_unet': dict(lr_mult=1.),
                                        "unet.zero_convs": dict(lr_mult=1),
                                        "unet.box_encoder": dict(lr_mult=1),


                                        }))
                                        


work_dir = '/work3/s203557/experiments/control_net_vpd/'
fp16 = dict(loss_scale=512.0)

log_level = 'INFO'
custom_hooks = [
    dict(type='TrainVisualizeHook', interval=2000, num_samples=2, save_dir='vis'),
    # dict(
    #     type='RestoreLrMultHook',
    #     module_names=['decode_head', 'neck'],
    #     lr_mult=1.0,
    #     warmup_iters=2000
    # )
]
