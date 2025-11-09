# 🧠 Face Recognition System (Python + OpenCV + LBPH)

This project is a **real-time facial recognition system** built using **Python**, **OpenCV**, and the **LBPH (Local Binary Patterns Histogram)** algorithm.  
It detects and recognizes faces from **live webcam video** or **static images**, based on a dataset of known individuals.  

The system is lightweight, works **offline**, and can be easily retrained with new faces.

---

## 🚀 Features

- ✅ Real-time face detection & recognition via webcam  
- ✅ Recognition from static images  
- ✅ Easy training with your own dataset  
- ✅ Modular architecture (Capture → Train → Recognize)  
- ✅ Works completely offline (no internet required)  
- ✅ Supports multiple image formats (`.jpg`, `.png`, `.webp`)  

---

## 🧠 How It Works

1. **Capture Phase** – Capture multiple face images for each person (using webcam).  
2. **Training Phase** – The system extracts facial features using LBPH and trains a model.  
3. **Recognition Phase** – The trained model identifies known faces in real-time through the webcam or static images.

---

## 🛠️ Requirements

### Hardware
- A system with a **webcam** (built-in or external)

### Software
- **Windows 10/11 (64-bit)**  
- **Python 3.12+**  
- **pip** (Python package manager)

### Python Libraries Used
- `opencv-contrib-python`  
- `numpy`  
- `Pillow`

---

## ⚙️ 1️⃣ Installation & Setup

Follow these steps to set up and run the project on your local machine 👇

###  Clone or Download the Project
If you have Git installed:
```powershell
git clone https://github.com/AswinSreehari/Face-Recognition.git
cd "Face-Recognition"
```

### 1️⃣ Navigate to project folder
cd "C:\Users\<YourName>\Desktop\Face Recognition"

### 2️⃣ Activate virtual environment
.\venv312\Scripts\Activate.ps1

### 3️⃣ (Optional) Capture your own images
python capture_samples.py "Your_Name" --count 20 --cam 0

### 4️⃣ Train the model
python train_lbph.py

### 5️⃣ Run live recognition
python recognize_lbph.py --cam 0 --threshold 90
