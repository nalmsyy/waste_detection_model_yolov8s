import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


# Threshold confidence khusus per kelas
# Tujuan: organik dibuat lebih tinggi karena sering false positive
CLASS_CONF = {
    "plastik": 0.10,
    "kertas": 0.25,
    "logam": 0.25,
    "lainnya": 0.35,
}


# Warna bounding box per kelas
CLASS_COLORS = {
    "plastik": (0, 255, 255),
    "kertas": (255, 255, 0),
    "logam": (255, 0, 255),
    "lainnya": (0, 165, 255),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Deteksi Sampah YOLOv8 5 Kelas menggunakan kamera OpenCV"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path model YOLOv8 best.pt",
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Index kamera. Biasanya 0 atau 1",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence awal YOLO. Filter akhir tetap memakai CLASS_CONF",
    )

    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IOU threshold untuk NMS",
    )

    parser.add_argument(
        "--show-fps",
        action="store_true",
        help="Tampilkan FPS",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Lebar kamera",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Tinggi kamera",
    )

    parser.add_argument(
        "--max-area",
        type=float,
        default=0.45,
        help="Maksimal rasio luas box terhadap frame. Box terlalu besar akan dibuang",
    )

    parser.add_argument(
        "--min-area",
        type=float,
        default=0.0005,
        help="Minimal rasio luas box terhadap frame. Box terlalu kecil akan dibuang",
    )

    return parser.parse_args()


def draw_text_with_bg(
    frame,
    text,
    position,
    font_scale=0.7,
    thickness=2,
    text_color=(255, 255, 255),
    bg_color=(0, 0, 0),
):
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX

    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    cv2.rectangle(
        frame,
        (x, y - text_h - baseline - 6),
        (x + text_w + 8, y + baseline),
        bg_color,
        -1,
    )

    cv2.putText(
        frame,
        text,
        (x + 4, y - 4),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def main():
    args = parse_args()

    print("Memuat model...")
    model = YOLO(args.model)

    print("Daftar kelas model:")
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")

    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"Kamera index {args.camera} tidak bisa dibuka.")
        print("Coba gunakan --camera 1")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window_name = "Deteksi Sampah YOLOv8 - 5 Kelas"

    prev_time = time.time()
    fps = 0.0

    print("Kamera aktif. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Frame kamera tidak terbaca.")
            break

        h, w = frame.shape[:2]
        frame_area = w * h

        # Inference YOLO
        # conf dibuat rendah dulu supaya semua kandidat masuk,
        # lalu kita filter manual per kelas di bawah.
        results = model(
            frame,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
        )

        boxes = results[0].boxes
        names = model.names

        class_counter = Counter()

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = names[cls_id]
            
            if class_name == "organik":
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w <= 0 or box_h <= 0:
                continue

            box_area = box_w * box_h
            area_ratio = box_area / frame_area

            # 1. Threshold khusus per kelas
            min_conf = CLASS_CONF.get(class_name, 0.50)

            if conf < min_conf:
                continue

            # 2. Buang box terlalu besar
            # Biasanya false positive pada tembok, tubuh, pintu, background.
            if area_ratio > args.max_area:
                continue

            # 3. Buang box terlalu kecil
            # Biasanya noise.
            if area_ratio < args.min_area:
                continue

            class_counter[class_name] += 1

            color = CLASS_COLORS.get(class_name, (0, 255, 120))
            label = f"{class_name} {conf:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            draw_text_with_bg(
                frame,
                label,
                (x1, max(y1 - 8, 25)),
                font_scale=0.75,
                thickness=2,
                text_color=(0, 0, 0),
                bg_color=color,
            )

        # Hitung FPS
        current_time = time.time()
        elapsed = current_time - prev_time
        prev_time = current_time

        if elapsed > 0:
            fps = 1.0 / elapsed

        # Panel jumlah deteksi
        panel_x = 20
        panel_y = 40

        draw_text_with_bg(
            frame,
            "Jumlah Deteksi:",
            (panel_x, panel_y),
            font_scale=0.8,
            thickness=2,
            text_color=(0, 255, 0),
            bg_color=(0, 0, 0),
        )

        y_offset = panel_y + 35

        if class_counter:
            for class_name, count in class_counter.items():
                text = f"{class_name}: {count}"
                draw_text_with_bg(
                    frame,
                    text,
                    (panel_x, y_offset),
                    font_scale=0.65,
                    thickness=2,
                    text_color=(0, 255, 0),
                    bg_color=(0, 0, 0),
                )
                y_offset += 30
        else:
            draw_text_with_bg(
                frame,
                "Tidak ada deteksi",
                (panel_x, y_offset),
                font_scale=0.65,
                thickness=2,
                text_color=(0, 255, 255),
                bg_color=(0, 0, 0),
            )

        # FPS
        if args.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            draw_text_with_bg(
                frame,
                fps_text,
                (20, h - 25),
                font_scale=0.75,
                thickness=2,
                text_color=(0, 255, 255),
                bg_color=(0, 0, 0),
            )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()