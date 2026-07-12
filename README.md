# 🍏📱 Smart Device Brand Classifier (Deep Learning Inference Pipeline)

An advanced computer vision and deep learning project deployed inside Google Colab to automate mobile hardware ecosystem classification. By leveraging a highly optimized convolutional neural network (CNN) model topology saved in Keras (`.h5`) format, the pipeline executes real-time inference on unseen validation target images—such as Apple iPhones, iPads, Apple Watches, and Samsung Galaxy flagship devices. The system dynamically parses multi-dimensional pixel arrays to return discrete category predictions paired with structural classification confidence parameters.

---

## 📊 Core Objectives & System Architecture

This specialized project was developed during the **Smart Methods Summer Training** to validate fundamental machine learning engineering concepts, including edge dataset structuring, localized model serialization, dynamic path binding, and tensor array parsing.

### 🎯 Key Project Milestones:
* **Deep Learning Inference Loop:** Programming a complete python-based prediction network utilizing serialized Keras structures without compilation overhead (`compile=False`).
* **Multi-Dimensional Array Operations:** Utilizing NumPy networks to dynamically instantiate, reshape, and normalize strict raw image input matrices to match predefined array boundaries:
  $$\text{Shape} = (1, 224, 224, 3), \quad \text{DataType} = \text{float32}$$
* **Brand Ecosystem Disambiguation:** Training the system to dynamically distinguish visual features across premium consumer hardware tech giants (Apple vs. Samsung) with high precision output logs.

---

## ⚙️ Software Dependencies & Environment Setup

The execution environment runs completely inside a hosted cloud workspace utilizing Python's robust scientific computing ecosystem:

| Software Layer | Framework / Library | Primary Operational Duty |
| :--- | :--- | :--- |
| **Execution Platform** | Google Colab Workspace | Serves as the interactive terminal for cloud compute and standard execution logs. |
| **Model Architecture** | `tf_keras` (TensorFlow) | Handles internal convolutional network parameter loadings and forward-pass inference. |
| **Array Preprocessing**| `numpy` (np) | Controls array initializations, matrix normalizations, and high-performance index argmax tracking. |
| **Image IO Manipulation**| `PIL` (Pillow) | Responsible for raw asset loading, color channel conversions (`.convert("RGB")`), and pixel resampling. |

---

## 🧠 Dataset & Targeted Brand Classes

The computer vision pipeline is evaluated across custom-curated hardware asset directories targeting real-world premium consumer electronics:

### 1. Apple Device Validation Classes (iPhones, iPads, & Apple Watches)
Multi-angle testing samples used to evaluate the model's spatial feature matching capability on iOS and watchOS physical builds.

<img width="5858" height="5858" alt="photo-output" src="https://github.com/user-attachments/assets/9ac4d3d1-01ed-4ce2-8ba4-f0d130c9726a" />

### 2. Samsung Galaxy Device Validation Classes
Representative product samples reflecting distinct hardware industrial design features unique to Samsung mobile configurations.

<img width="4992" height="4992" alt="photo-output" src="https://github.com/user-attachments/assets/854401be-e6e0-4bd9-a2a2-8bb55aef9ceb" />


---


## 🧠 Code Interface & Implementation Pipeline

The execution architecture relies on modular pipeline boundaries designed to streamline storage fetching and tensor transformations:

* **Storage Drive Gating:** Establishes an independent cloud path connection to map global asset variables inside the core notebook terminal layer.
* **Asset Allocation Validation:** Runs automated workspace scans using standard terminal utilities to evaluate the initialization status of core weights and dictionary arrays (`keras_model.h5` and `labels.txt`).
* **Mathematical Inference Activation:** Feeds raw incoming pixel vectors into deep matrix layers to output deterministic category predictions by executing code lookup blocks via index matching.

---

## 🚀 Virtual Workspace Verification & Inference Output

### 1. Model Execution Output (Real-Time Target Classification)
Live validation snapshot verifying an ultra-high prediction outcome with zero system latency during test sweeps:

<img width="2367" height="596" alt="IMG_5689" src="https://github.com/user-attachments/assets/8a37e4a9-ff54-49cb-8b57-c4183faa753d" />


### 2. Google Colab Workspace Pipeline Layout
Comprehensive overview of the underlying python codebase, library imports, file loading logic, and standard dictionary mapping configurations.


<img width="1194" height="834" alt="IMG_5692" src="https://github.com/user-attachments/assets/ec26fa65-f5ad-47e7-917d-3981d0cd91d1" />

---

## 🌐 Real-World & Industrial Applications

The deep learning principles mastered across this deployment scale directly into large-scale commercial pipelines:
* **E-Commerce Inventory Automation:** Automating multi-vendor product intakes at massive fulfillment hubs through immediate camera tagging.
* **Counterfeit Product Detection:** Instantly cross-checking external manufacturing footprints to isolate fake devices from certified retail lines.
* **Automated Device Trade-In Kiosks:** Powering retail consumer kiosks to automatically assess device models and catalog phone models during upgrade cycles.

---

## 📈 Engineering Outlook & Future Enhancements

To upgrade this classification script into an enterprise-ready, platform-agnostic application, the following modifications can be introduced:
1. **Edge Deployment Conversion:** Translating the bulky Keras model structure into a highly compressed TensorFlow Lite (`.tflite`) file optimized for mobile hardware architectures.
2. **Real-Time Webcam Pipeline:** Interfacing the python model layer with a native OpenCV (`cv2`) video stream capture module to handle living multi-frame object recognition.
3. **Data Augmentation Expansion:** Injecting robust geometric transformations (random shearing, mirroring, and extreme contrast adjustments) during the training phase to eliminate edge orientation prediction bias.

---

## 👥 Project Team & Affiliation

* **Lead AI Engineer:** Eng. Fajr Aldajani
* **Training Program:** Smart Methods Summer Training
* **Domain Focus:** Embedded Systems & AI Lab Integration
* **Project Status:** Completed Successfully! 🚀
