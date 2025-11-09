# train_lbph.py
# Train an LBPH face recognizer from images in dataset/<person>/*

import os
import cv2
import numpy as np
import json

DATASET_DIR = "dataset"
MODEL_PATH = "lbph_model.yml"
LABELS_PATH = "labels.json"
FACE_SIZE = (200, 200)  # size to which face ROIs are resized

# Haar cascade for face detection (bundled with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def gather_images_and_labels(dataset_dir):
    images = []
    labels = []
    label_names = {}
    current_label = 0

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder '{dataset_dir}' not found. Create dataset/<person>/images.")

    for person_name in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
        label_names[str(current_label)] = person_name
        file_count = 0
        for fname in sorted(os.listdir(person_dir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"[WARN] Could not read {path}, skipping.")
                continue
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                # try a more permissive detect (sometimes needed)
                faces = face_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3)
            for (x, y, w, h) in faces:
                roi = img[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, FACE_SIZE)
                images.append(roi_resized)
                labels.append(current_label)
                file_count += 1
        print(f"[INFO] {person_name}: {file_count} face ROIs added (label {current_label})")
        current_label += 1

    return images, labels, label_names

if __name__ == "__main__":
    imgs, labs, label_names = gather_images_and_labels(DATASET_DIR)
    if len(imgs) == 0:
        print("[ERROR] No faces found. Make sure dataset contains subfolders with face images.")
        raise SystemExit(1)

    # create LBPH recognizer and train
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(np.array(imgs), np.array(labs))
    recognizer.write(MODEL_PATH)

    # save label mapping
    with open(LABELS_PATH, "w", encoding="utf8") as f:
        json.dump(label_names, f, ensure_ascii=False, indent=2)

    print(f"[OK] Trained on {len(imgs)} samples for {len(label_names)} people. Model saved to {MODEL_PATH}")
