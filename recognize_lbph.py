# recognize_lbph.py
# Real-time webcam LBPH recognition. Draws rectangle and name below the box.

import cv2
import json
import os
import argparse

MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
FACE_SIZE = (200, 200)

def draw_label_below_box(frame, x, y, w, h, label):
    # draw rectangle
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # label background and text below rectangle
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

def recognize_from_webcam(cam_index=0, confidence_threshold=90):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        raise SystemExit("[ERROR] Model or labels not found. Run train_lbph.py first.")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, "r", encoding="utf8") as f:
        label_names = json.load(f)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {cam_index}")
        return
    print("[INFO] Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi, FACE_SIZE)
            label_id, confidence = recognizer.predict(roi_resized)  # lower confidence => better match
            name = label_names.get(str(label_id), "Unknown")
            if confidence > confidence_threshold:
                name = "Unknown"
            draw_label_below_box(frame, x, y, w, h, f"{name}")
        cv2.imshow("LBPH Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0, help="camera index (default 0)")
    parser.add_argument("--threshold", type=float, default=90.0, help="confidence threshold (default 90)")
    args = parser.parse_args()
    recognize_from_webcam(cam_index=args.cam, confidence_threshold=args.threshold)
