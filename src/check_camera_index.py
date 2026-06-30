import cv2


def check_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print(f"Kamera index {index}: TIDAK TERBUKA")
        return

    ret, frame = cap.read()

    if ret:
        h, w = frame.shape[:2]
        print(f"Kamera index {index}: TERBUKA | Resolusi: {w}x{h}")
    else:
        print(f"Kamera index {index}: TERBUKA tapi frame tidak terbaca")

    cap.release()


def main():
    print("Mengecek kamera index 0 sampai 10...\n")

    for i in range(0, 11):
        check_camera(i)

    print("\nSelesai.")
    print("Kalau kamera HP terdeteksi, gunakan index tersebut di --camera")


if __name__ == "__main__":
    main()