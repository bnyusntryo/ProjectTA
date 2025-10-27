# D:\mmdetection\export_manual.py

import torch
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from mmdet.registry import MODELS
from mmdet.apis import DetInferencer
import os

# --- PENGATURAN (SESUAIKAN JIKA PERLU) ---
config_file = 'configs/my_project/retinanet_tinyvit_railway.py'
checkpoint_file = 'work_dirs/retinanet_tinyvit_railway/best_coco_bbox_mAP_epoch_45.pth'
output_file = 'work_dirs/retinanet_final.onnx'
input_shape = (1, 3, 800, 800)
# ----------------------------------------

def main():
    print(f"Membaca konfigurasi dari: {config_file}")
    cfg = Config.fromfile(config_file)
    
    if cfg.model.get('test_cfg') is None:
        cfg.model.test_cfg = cfg.get('test_cfg')

    print("Membangun model RetinaNet+TinyViT...")
    model = MODELS.build(cfg.model)
    
    print(f"Memuat checkpoint dari: {checkpoint_file}")
    load_checkpoint(model, checkpoint_file, map_location='cpu')

    model.eval()
    dummy_input = torch.randn(*input_shape)

    print(f"\nMemulai proses ekspor ke ONNX. Ini mungkin memakan waktu...")
    try:
        # Baris yang menyebabkan error telah dihapus
        
        torch.onnx.export(
            model,
            dummy_input,
            output_file,
            opset_version=14,
            input_names=['input'],
            output_names=['dets', 'labels'],
            do_constant_folding=True,
            verbose=False,
            dynamic_axes={
                'input': {0: 'batch_size'},
                'dets': {0: 'batch_size'},
                'labels': {0: 'batch_size'}
            }
        )
        print("\n" + "="*30)
        print(f"✅ SUKSES! Model telah diekspor ke: {output_file}")
        print("="*30)

    except Exception as e:
        print("\n" + "!"*30)
        print(f" GAGAL! Terjadi error saat ekspor: {e}")
        print("!"*30)

if __name__ == '__main__':
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    main()