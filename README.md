# 🏥 Automatic Segmentation of the Uterus, Endometrium, and Myometrium Using Ultrasound Images  

## 📖 Overview  
This project focuses on the **automatic segmentation of the endometrium using ultrasound images** using a **U-Net deep learning model**. The dataset consists of **DICOM ultrasound scans**, which are used for training and testing the model.  

⚠️ **Important Note on Patient Data Privacy**  
Due to **privacy and ethical restrictions**, the original dataset **cannot be shared** in this repository. However, we provide instructions on how you can **prepare your own dataset** to replicate this study.  

---

## 🚀 Project Workflow  
### **1️⃣ Data Collection (NOT INCLUDED)**
- The dataset consists of **DICOM ultrasound images**.
- The dataset is **not publicly available** due to patient privacy regulations.
- Users should **use their own medical ultrasound datasets** to train the model.

### **2️⃣ Data Preprocessing**
- Convert `.mat` segmentations into **JPG format**.
- Extract **grayscale B-mode and CEUS images**.
- Normalize pixel intensities for model consistency.
- Implemented in:
  - **`convert-mat-to-jpg.m`** - Converts `.mat` segmentation masks into JPG images.
  - **`delineationScript_Eva.m`** - Loads DICOM frames and applies manual delineation.

### **3️⃣ Data Augmentation**
- Augments images to **increase dataset diversity** and reduce overfitting.
- Includes:
  - Rotation
  - Flipping
  - Contrast adjustment
- Implemented in:
  - **`augmented.ipynb`** - Applies data augmentation techniques.

### **4️⃣ Training the Model (U-Net)**
- A **U-Net model** is trained to segment the **endometrium region**.
- Training includes:
  - **Preprocessing grayscale images**
  - **Splitting data (80% train, 20% test)**
  - **Training the U-Net model**
  - **Saving the trained weights**
- Implemented in:
  - **`create_h5_with_aug.ipynb`** - Prepares dataset for training.
  - **`prediction.ipynb`** - Trains and evaluates the U-Net model.

### **5️⃣ Evaluation and Prediction**
- The trained model predicts segmentation masks for test images.
- The model's performance is evaluated using:
  - **Dice Coefficient**
  - **Intersection over Union (IoU)**
- Implemented in:
  - **`first_frame_of_six_of_data_test_with_each_dice_and_iou.py`** - Evaluates segmentation quality.
  - **`prediction.ipynb`** - Runs inference on test images.

---

# Automatic Segmentation of the Uterus, Endometrium, and Myometrium Using Ultrasound Images

## Overview  
(description of project)

## Project Workflow  
(step-by-step explanation)

## Project Structure  
(folder structure of repository)

## Dataset Privacy Policy  
(why dataset is not included)

## Installation & Dependencies  
(how to set up the project)

## Run the Scripts  
(how to execute the code)

## Results  
(what the model outputs)

## 📌 Future Work  ✅ 
- Improve segmentation accuracy by **fine-tuning U-Net hyperparameters**.
- Train on a **larger dataset** with more variations in ultrasound scans.
- Implement **real-time segmentation** for clinical use.
- Explore **alternative deep learning models** for better segmentation results.
- Optimize computational efficiency for **faster inference in real-world settings**.

