# configs/my_project/rtmdet_tiny_railway.py

_base_ = '../rtmdet/rtmdet-tiny_8xb32-300e_coco.py'

# 1. Pengaturan Dataset
data_root = 'D:/TA/RAILWAY_TRACK_FAULT_DETECTION.v1i.coco/'
class_name = ('fishplate', 'fishplate_bolthead', 'fishplate_boltmissing',
              'fishplate_boltnut', 'track_bolt', 'track_boltmissing', 'track_crack')
num_classes = len(class_name)
metainfo = dict(classes=class_name)

# 2. Pengaturan Dataloader
# Sesuaikan path anotasi ke dataset Anda yang sudah bersih
train_dataloader = dict(
    batch_size=16, # RTMDet-tiny sangat ringan, coba batch size lebih besar
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='train.json',
        data_prefix=dict(img='all_images/')))

val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='valid.json',
        data_prefix=dict(img='all_images/')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='test.json',
        data_prefix=dict(img='all_images/')))

# 3. Pengaturan Evaluator
val_evaluator = dict(ann_file=data_root + 'valid.json')
test_evaluator = dict(ann_file=data_root + 'test.json')

# 4. Modifikasi Kepala Model
model = dict(
    bbox_head=dict(
        num_classes=num_classes))

# 5. Pengaturan Pelatihan (Lebih Cepat!)
max_epochs = 25  # <-- CUKUP LATIH 25 EPOCH, JAUH LEBIH CEPAT!
train_cfg = dict(max_epochs=max_epochs, val_interval=5) # Validasi setiap 5 epoch

# Atur learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1.0e-5, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=1.0e-6,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True)
]

# Ambil bobot pre-trained RTMDet-tiny
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'