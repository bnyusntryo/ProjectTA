import cv2
import time
import firebase_admin
from firebase_admin import credentials, db
from mmdet.apis import DetInferencer
import argparse

# --- PENGATURAN ---
CONFIG_PATH = 'D:/mmdetection/configs/my_project/retinanet_tinyvit_railway.py'
CHECKPOINT_PATH = 'D:/mmdetection/work_dirs/retinanet_tinyvit_railway/best_coco_bbox_mAP_epoch_45.pth'
VIDEO_SOURCE_PATH = 'videoplayback.mp4'
FIREBASE_CREDENTIALS_PATH = 'firebase-credentials.json'
FIREBASE_DATABASE_URL = 'https://cctvrel-default-rtdb.asia-southeast1.firebasedatabase.app/'
CCTV_LOCATION_ID = 'CCTV-LAB-01'
CONFIDENCE_THRESHOLD = 0.65
FRAME_SKIP = 5
# --------------------

def initialize_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred, {'databaseURL': FIREBASE_DATABASE_URL})
    print("✅ Koneksi Firebase berhasil diinisialisasi.")
    return True

def main(device):
    if not initialize_firebase():
        return

    print("Memuat model PyTorch (RetinaNet + TinyViT)...")
    try:
        inferencer = DetInferencer(model=CONFIG_PATH, weights=CHECKPOINT_PATH, device=device)
        print(f"✅ Model berhasil dimuat di {device}.")
    except Exception as e:
        print(f"🔥 GAGAL memuat model: {e}")
        return

    cap = cv2.VideoCapture(VIDEO_SOURCE_PATH)
    if not cap.isOpened():
        print(f"🔥 GAGAL membuka video: {VIDEO_SOURCE_PATH}")
        return

    print("\n🚀 Memulai simulasi deteksi...")
    frame_count = 0
    visualized_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("\nVideo selesai.")
            break

        if frame_count % FRAME_SKIP == 0:
            results = inferencer(
                frame,
                return_vis=True,
                no_save_vis=True,
                pred_score_thr=CONFIDENCE_THRESHOLD
            )
            visualized_frame = results['visualization'][0]
            
            # --- BAGIAN YANG DIPERBAIKI ---
            predictions = results['predictions'][0]
            scores = predictions['scores'] # Ini adalah list Python
            labels = predictions['labels']
            
            # Cek apakah ada deteksi sama sekali di dalam list
            if scores: # Cek jika list 'scores' tidak kosong
                # Cari skor tertinggi di dalam list
                top_score = max(scores)

                # Jika skor tertinggi melampaui ambang batas, kirim notifikasi
                if top_score > CONFIDENCE_THRESHOLD:
                    top_index = scores.index(top_score)
                    top_label = labels[top_index]
                    class_name = inferencer.model.dataset_meta['classes'][top_label]

                    print(f"TERDETEKSI: {class_name} dengan keyakinan {top_score*100:.2f}%")
                    
                    # Kirim notifikasi ke Firebase
                    ref = db.reference('alerts')
                    ref.push().set({
                        'type': class_name,
                        'confidence': float(top_score),
                        'location': CCTV_LOCATION_ID,
                        'timestamp': int(time.time())
                    })
            # --- AKHIR BAGIAN YANG DIPERBAIKI ---

        frame_count += 1

        if visualized_frame is not None:
            cv2.imshow('CCTV Simulator Monitor', visualized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Simulasi selesai.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CCTV Simulator for Railway Fault Detection.')
    parser.add_argument('--device', default='cpu', help='Device for inference, e.g., "cpu" or "cuda:0".')
    args = parser.parse_args()
    main(args.device)