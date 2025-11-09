# capture_samples.py
# Capture photos from webcam to populate dataset/<person>/

import cv2
from pathlib import Path

def capture_person(person_name, out_dir="dataset", cam_index=0, count=20):
    out_path = Path(out_dir) / person_name
    out_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press SPACE to capture an image, 'q' to quit.")
    saved = 0
    while saved < count:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capture - press SPACE to save", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            fname = out_path / f"{person_name}_{saved+1}.jpg"
            cv2.imwrite(str(fname), frame)
            print(f"Saved {fname}")
            saved += 1
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="person name (folder will be created under dataset/)")
    parser.add_argument("--count", type=int, default=10, help="how many images to capture")
    parser.add_argument("--cam", type=int, default=0, help="camera index")
    args = parser.parse_args()
    capture_person(args.name, out_dir="dataset", cam_index=args.cam, count=args.count)
