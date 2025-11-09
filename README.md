# 🧠 Face Recognition System (Python + OpenCV + LBPH)

This is a **real-time facial recognition system** built using **Python**, **OpenCV**, and the **LBPH (Local Binary Patterns Histogram)** algorithm.  
It can identify people both from webcam video feed and static images, based on pre-trained datasets of faces.

---

## 🚀 Features

- ✅ Real-time face detection & recognition via webcam  
- ✅ Recognition from static images  
- ✅ Dataset-based training (easy to add new people)  
- ✅ Modular architecture: capture / train / recognize  
- ✅ Works offline (no internet needed)  
- ✅ Supports `.jpg`, `.png`, `.webp` formats  

---

## 🛠️ Requirements

- **Python 3.12+**  
- **pip** (Python package manager)  
- **A working webcam**  

### Python Libraries
- `opencv-contrib-python`
- `numpy`
- `Pillow`

---

 # 1️⃣ Activate virtual environment
.\venv312\Scripts\Activate.ps1

# 2️⃣ Capture your own images (optional)
python capture_samples.py "Your_Name" --count 20 --cam 0

# 3️⃣ Train the recognizer model
python train_lbph.py

# 4️⃣ Run live recognition via webcam
python recognize_lbph.py --cam 0 --threshold 90
