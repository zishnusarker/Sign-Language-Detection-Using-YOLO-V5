<h1 align="center">🤟 Sign Language Detection using YOLOv5</h1>

<p align="center">
  <strong>Real-time sign language detection using the YOLOv5 object detection framework - a final-year B.Tech project comparing YOLOv5 with ANN and CNN approaches for sign language recognition.</strong>
</p>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"></a>
  <a href="#"><img src="https://img.shields.io/badge/YOLOv5-00FFFF?style=for-the-badge&logo=yolo&logoColor=black" alt="YOLOv5"></a>
  <a href="#"><img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="#"><img src="https://img.shields.io/badge/Final%20Year-Project-purple?style=for-the-badge" alt="Final Year"></a>
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-motivation">Motivation</a> •
  <a href="#-why-yolov5">Why YOLOv5</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-installation">Installation</a> •
  <a href="#-training">Training</a> •
  <a href="#-detection">Detection</a> •
  <a href="#-comparison">Comparison</a>
</p>

---

## 📖 Overview

A **real-time sign language detection system** built with the **YOLOv5 (You Only Look Once, version 5)** object detection framework. The model detects and classifies sign language gestures from images and live webcam feeds with bounding box localization and confidence scores.

This is the **YOLOv5 implementation** of a comparative study conducted as a 7th semester B.Tech final-year project. The goal: evaluate object detection (YOLOv5) against traditional image classification approaches (ANN and CNN) for sign language recognition.

> **Companion Repository:** The ANN and CNN baseline implementations can be found in the [Sign-Language-Detection-Using-ANN-CNN](https://github.com/zishnusarker/Sign-language-Detection-Using-ANN-CNN) repository.

---

## 💡 Motivation

**Sign language** is the primary mode of communication for millions of hearing-impaired individuals worldwide. However, the communication gap between signers and non-signers remains a significant barrier to inclusion.

This project aims to:
- 🌍 **Bridge communication gaps** - Enable real-time sign language interpretation
- 🎯 **Compare approaches** - Benchmark object detection (YOLOv5) vs classification (ANN/CNN)
- 📱 **Real-world usability** - Build a system that works via webcam in real-time
- 🎓 **Academic contribution** - Provide empirical data on deep learning approaches for sign language

---

## 🎯 Why YOLOv5?

YOLOv5 is a state-of-the-art object detection framework chosen for this project because:

| Advantage | Benefit |
|-----------|---------|
| **⚡ Real-time speed** | Processes frames at 30+ FPS on modern GPUs |
| **📍 Localization** | Provides bounding boxes, not just classification |
| **🎯 High accuracy** | State-of-the-art mAP on COCO benchmark |
| **🔧 Transfer learning** | Pre-trained weights enable fast training with small datasets |
| **🖥 Cross-platform** | Export to ONNX, TorchScript, CoreML, TFLite |
| **📦 Easy to use** | Well-documented training and inference pipeline |

### YOLOv5 vs Traditional Classification

| Aspect | ANN / CNN | YOLOv5 |
|--------|-----------|--------|
| **Output** | Single class label | Class + bounding box + confidence |
| **Input** | Pre-cropped sign image | Full scene with sign |
| **Real-time** | Requires pre-processing | End-to-end detection |
| **Multi-sign** | One sign at a time | Multiple signs simultaneously |
| **Use case** | Static image classification | Live video / real-world scenes |

---

## 🛠 Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Language** | Python 3.7+ | Core implementation |
| **Deep Learning** | PyTorch | YOLOv5 framework backbone |
| **Detection Model** | YOLOv5 (Ultralytics) | Object detection architecture |
| **Computer Vision** | OpenCV | Webcam capture and image processing |
| **Notebook** | Jupyter | Interactive training and detection |
| **Model Format** | PyTorch `.pt` | Serialized trained weights |
| **Visualization** | Matplotlib, PIL | Display detection results |

---

## 📁 Project Structure

```
Sign-Language-Detection-Using-YOLO-V5/
├── README.md
├── LICENSE
│
└── Sign Language Recognition YOLO v5/
    ├── (YOLOV5)SignLanguageRecognition.ipynb   # Main Jupyter notebook
    ├── best.pt                                  # Trained YOLOv5 weights
    │
    ├── Result SS/                               # Detection result screenshots
    │   ├── Screenshot 2022-04-29 142723.png
    │   ├── Screenshot 2022-04-29 142751.png
    │   ├── Screenshot 2022-04-29 142822.png
    │   ├── Screenshot 2022-04-29 143438.png
    │   └── webcan visualization.png             # Real-time webcam demo
    │
    └── code SS/                                 # Code walkthrough screenshots
        ├── s1.png
        ├── s2.png
        ├── s3.png
        ├── s4.png
        ├── s6.png
        └── s7.png
```

---

## 🚀 Installation

### Prerequisites
- Python 3.7+
- CUDA-capable GPU (recommended for training; CPU works for inference)
- Webcam (for real-time detection)

### Setup

```bash
# Clone this repository
git clone https://github.com/zishnusarker/Sign-Language-Detection-Using-YOLO-V5.git
cd Sign-Language-Detection-Using-YOLO-V5

# Clone YOLOv5 framework
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install YOLOv5 dependencies
pip install -r requirements.txt

# Install Jupyter (if not already installed)
pip install jupyter notebook
```

### Launch the Notebook

```bash
cd "Sign Language Recognition YOLO v5"
jupyter notebook "(YOLOV5)SignLanguageRecognition.ipynb"
```

---

## 🏋️ Training

The notebook walks through the complete training pipeline:

1. **Dataset Preparation** - Organize images and labels in YOLO format (`train/images`, `train/labels`)
2. **Custom `data.yaml`** - Define classes and dataset paths
3. **Transfer Learning** - Start from pre-trained YOLOv5s/m/l weights
4. **Training Command** - Run `train.py` with custom hyperparameters
5. **Evaluation** - Monitor loss, mAP, precision, and recall
6. **Best Weights** - Trained model saved as `best.pt`

### Example Training Command

```bash
python train.py --img 640 --batch 16 --epochs 100 \
                --data sign_language.yaml \
                --weights yolov5s.pt \
                --name sign_language_yolov5
```

---

## 🎥 Detection

### On Images

```bash
python detect.py --weights best.pt --img 640 --conf 0.25 --source path/to/image.jpg
```

### Real-time Webcam Detection

```bash
python detect.py --weights best.pt --img 640 --conf 0.25 --source 0
```

The system will display bounding boxes around detected signs with class labels and confidence scores in real-time.

### Sample Detection Output

See the `Result SS/` folder for screenshots including:
- Static image detections
- Live webcam visualization demonstrating real-time inference

---

## 📊 Comparison: YOLOv5 vs ANN vs CNN

This project is part of a comparative study. The full comparison is discussed across two repositories:

| Model | Repository | Approach |
|-------|-----------|----------|
| **ANN** | [Sign-language-Detection-Using-ANN-CNN](https://github.com/zishnusarker/Sign-language-Detection-Using-ANN-CNN) | Fully-connected neural network on flattened pixels |
| **CNN** | [Sign-language-Detection-Using-ANN-CNN](https://github.com/zishnusarker/Sign-language-Detection-Using-ANN-CNN) | Convolutional network with feature extraction |
| **YOLOv5** | This repository | Object detection with localization |

### Key Takeaways

- **ANN**: Simple baseline, struggles with spatial features
- **CNN**: Better at learning hierarchical features, good for static classification
- **YOLOv5**: Superior for real-time detection with localization - the clear winner for real-world deployment

---

## 📸 Results

The model successfully detects sign language gestures with:
- ✅ Real-time webcam inference
- ✅ Bounding box localization
- ✅ Class labels with confidence scores
- ✅ Multi-sign detection in a single frame

Check the `Result SS/` folder for visual examples of the model in action.

---

## 🎓 Key Concepts Demonstrated

<details>
<summary><strong>What is YOLOv5 and how does it work?</strong></summary>

YOLOv5 is a single-stage object detector that divides an input image into a grid and predicts bounding boxes, class probabilities, and confidence scores for each grid cell in a single forward pass. This makes it much faster than two-stage detectors (like Faster R-CNN) while maintaining competitive accuracy.

</details>

<details>
<summary><strong>Why use transfer learning?</strong></summary>

YOLOv5 models are pre-trained on the COCO dataset (80 classes, 330K images). By starting from these weights, the model already knows how to detect generic visual features (edges, textures, shapes). Fine-tuning on a smaller sign language dataset is much faster and more effective than training from scratch.

</details>

<details>
<summary><strong>What's inside `best.pt`?</strong></summary>

The `best.pt` file contains the PyTorch state dict with the trained model weights from the epoch that achieved the best validation mAP during training. It can be loaded directly with `torch.load()` or used with YOLOv5's `detect.py` script.

</details>

<details>
<summary><strong>What is mAP (mean Average Precision)?</strong></summary>

mAP is the standard evaluation metric for object detection. It measures both classification accuracy and localization quality by averaging precision across all classes at various IoU (Intersection over Union) thresholds. Higher mAP = better detection.

</details>

---

## 🔮 Future Improvements

- Expand dataset to cover more sign language alphabets (ASL, BSL, ISL)
- Deploy as a web app using Flask/Streamlit with webcam streaming
- Convert model to ONNX/TFLite for mobile deployment
- Add word-level and sentence-level sign detection (temporal models like LSTM + CNN)
- Integrate text-to-speech for detected signs
- Build a full accessibility application for hearing-impaired users
- Collect diverse dataset (different skin tones, lighting, backgrounds)
- Compare with YOLOv7, YOLOv8, and other modern detectors

---

## 📚 References

- **YOLOv5**: [Ultralytics YOLOv5 Repository](https://github.com/ultralytics/yolov5)
- **Original YOLO Paper**: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- **PyTorch**: [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ as a B.Tech 7th Semester Final Year Project
</p>

<p align="center">
  <strong>Breaking communication barriers with computer vision 🤟</strong>
</p>
