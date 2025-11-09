# recognize_image.py
# Test recognition on a static image. Draws rectangles and names below.

import cv2
import json
import os
import argparse

MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
FACE_SIZE = (200, 200)

def draw_label_below_box(frame, x, y, w, h, label):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    label_left = x
    label_top = y + h + 6
    label_right = label_left + text_w + 10
    label_bottom = label_top + text_h + baseline + 6
    frame_h, frame_w = frame.shape[:2]
    if label_right > frame_w - 1:
        label_right = frame_w - 1
        label_left = max(0, label_right - (text_w + 10))
    cv2.rectangle(frame, (label_left, label_top), (label_right, label_bottom), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (label_left + 5, label_top + text_h + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def recognize_image(image_path, confidence_threshold=90):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise SystemExit("[ERROR] Model or labels not found. Run train_lbph.py first.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf8") as f:
        label_names = json.load(f)

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not open {image_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, FACE_SIZE)
        label_id, confidence = recognizer.predict(roi_resized)
        name = label_names.get(str(label_id), "Unknown")
        if confidence > confidence_threshold:
            name = "Unknown"
        draw_label_below_box(img, x, y, w, h, f"{name}")
    cv2.imshow("Image Recognition", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to image file")
    parser.add_argument("--threshold", type=float, default=90.0)
    args = parser.parse_args()
    recognize_image(args.image, confidence_threshold=args.threshold)
