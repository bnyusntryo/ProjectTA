# configs/my_project/retinanet_tinyvit_railway.py

_base_ = '../retinanet/retinanet_r50_fpn_1x_coco.py'

default_hooks=dict(
    checkpoint = dict(
        type = 'CheckpointHook',
        interval = 1,
        save_best = 'auto',
        max_keep_ckpts = 3
    )
)

# 1. Pengaturan Dataset (Sama seperti sebelumnya)
data_root = 'D:/TA/RAILWAY_TRACK_FAULT_DETECTION.v1i.coco/'
class_name = ('fishplate', 'fishplate_bolthead', 'fishplate_boltmissing',
              'fishplate_boltnut', 'track_bolt', 'track_boltmissing', 'track_crack')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

model = dict(
    # Tambahkan blok ini secara eksplisit
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    
    backbone=dict(
        _delete_=True,
        type='mmpretrain.TIMMBackbone',
        model_name='tiny_vit_21m_224',
        features_only=True,
        pretrained=True,
        out_indices=(1, 2, 3)
    ),

    neck=dict(
        in_channels=[192, 384, 576],
        start_level=0,
        num_outs=5
    ),
    
    bbox_head=dict(
        num_classes=num_classes
    )
)

# Ambil pre-trained weights untuk mmpretrain backbone
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth' # Contoh, bisa diganti

# Tambahkan blok kode ini ke dalam file configs/my_project/retinanet_tinyvit_railway.py

# Ganti train_pipeline Anda dengan versi agresif ini

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    # 1. Tambahkan augmentasi warna, kecerahan, kontras secara acak
    # dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', prob=0.5),
    # 2. Selalu resize ke skala yang lebih besar terlebih dahulu
    dict(type='Resize', scale=(1000, 1000), keep_ratio=True),
    # 3. Selalu lakukan crop acak ke ukuran input model
    dict(type='RandomCrop', crop_size=(800, 800)),
    dict(type='PackDetInputs')
]

# 3. Pengaturan Dataloader (Sama seperti sebelumnya)
train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='dataset_final_split/train_cv_folds/train.json',
        data_prefix=dict(img='all_images/'),
    )
)

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='dataset_final_split/train_cv_folds/valid.json',
        data_prefix=dict(img='all_images/'),
    )
)

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='dataset_final_split/train_cv_folds/test.json',
        data_prefix=dict(img='all_images/'),
    )
)

# 4. Pengaturan Evaluator (Sama seperti sebelumnya)
val_evaluator = dict(ann_file=data_root + 'dataset_final_split/train_cv_folds/valid.json')
test_evaluator = dict(ann_file=data_root + 'dataset_final_split/train_cv_folds/test.json')

# 5. Pengaturan Pelatihan
max_epochs = 50
train_cfg = dict(max_epochs=max_epochs, val_interval=5)

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.0001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 45],
        gamma=0.1)
]

optim_wrapper = dict (
    type = 'OptimWrapper',
    optimizer = dict (
        _delete_=True,
        type = 'AdamW',
        lr=0.0001,
        weight_decay=0.0001
    )
)