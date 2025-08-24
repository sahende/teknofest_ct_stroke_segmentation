# 🧠 Stroke Segmentation – Brain CT using Swin UPerNet + CBAM

This repository contains a **binary segmentation model** trained to detect **stroke regions** in brain CT images using a **SwinUPerNet architecture with CBAM attention**. The model predicts pixel-wise stroke masks from 2D axial CT slices.

---

## 🧪 Task Description

**Objective:**  
Given a 2D brain CT slice, predict the stroke region as a binary mask:

- 0 → Background / Healthy tissue  
- 1 → Stroke region  

The model performs slice-level segmentation.

---

## 🧠 Dataset Sources

### 🏥 Primary Training Set
- **Source:** TEKNOFEST 2021 AI in Healthcare Dataset  
- **Link:** [Public Data Portal](https://acikveri.saglik.gov.tr/Home/DataSets?categoryId=10)  
- **Modality:** Non-contrast Brain CT  
- **Labels:** Expert segmentation masks (stroke vs. no stroke)

## 📚 Citation / Disclaimer

Dataset used is public and anonymized, for research and educational purposes only.

TEKNOFEST 2021 AI in Healthcare dataset is credited to the **Turkish Ministry of Health.**

---

## 🔧 Preprocessing

DICOM slices were processed as follows:

1. Converted to **Hounsfield Units (HU)** if original PNGs did not exist.  
2. Normalized using training set parameters: `mean ≈ 0.189`, `std ≈ 0.318`.  
3. Enhanced using **CLAHE (Contrast Limited Adaptive Histogram Equalization)** if PNGs were not provided.  
4. If PNG images already existed, they were used without further modification.

Data augmentation applied during training:

- Slight **rotations**  
- Slight **shifts**
Medically unsafe augmentations (cropping, strong geometric transforms...) were avoided.
---

## 🏗️ Model

| Component          | Details |
|-------------------|---------|
| **Base Model**     | SwinUPerNet with CBAM (Swin-Tiny) |
| **Input Size**     | 224 × 224 RGB |
| **Output**         | Binary segmentation mask |
| **Loss Function**  | `BCEWithLogitsLoss` |
| **Optimizer**      | AdamW (lr = 5e-5) |
| **Regularization** | ReduceLROnPlateau scheduler |
| **Gradient Accumulation** | Yes (steps = 2) |
| **Mixed Precision** | Enabled with AMP |

---

## 📊 Results

**Testing with TTA (Test-Time Augmentation):**  

**Dice** = 0.8310
**F1** = 0.8062
**IoU** = 0.6754
**Threshold** = 0.50

---

## 🔽 Download Model

You can download the trained segmentation model from **Hugging Face**:

- **Model URL:** [Sahende/teknofest_ct_stroke_seg](https://huggingface.co/Sahende/teknofest_ct_stroke_seg)

**(Optional) For faster model downloads from Hugging Face**
```bash
pip install "huggingface_hub[hf_xet]"
```
## 📦 Installation

### Create virtual environment
```bash
conda create -n stroke_seg python=3.9
conda activate stroke_seg

# Install dependencies
pip install -r requirements.txt
```
---
## 🔽 Usage

The pre-trained SwinUPerNet-CBAM model is automatically downloaded from Hugging Face when you run `notebook/run_eval.py`.  
Simply make sure your test images and masks are in the correct folders (`cfg.TEST_IMG_DIR` and `cfg.TEST_MASK_DIR`) and run:

```bash
python notebook/run_eval.py
```
---

## 🐳 Docker Usage

You can also run the demo inside Docker using the provided Dockerfile.
```bash
# Build the Docker image
docker build -t teknofest-stroke-seg .

#Run the container
docker run -it --rm teknofest-stroke-seg
```
By default, it will execute notebook/run_demo.py.

---

## ⚡ Notes

Mixed precision training is enabled for speed and memory efficiency.

Gradient accumulation is used to handle small batch sizes.

The repository contains scripts for training, validation, and inference.

---


