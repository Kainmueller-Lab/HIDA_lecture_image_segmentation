# Machine Learning for Image Segmentation — HIDA Lecture

This lecture introduces **Machine Learning for Image Segmentation** using a hands-on, guided Jupyter Notebook designed for **Google Colab**.  
It is aimed at research scientists who are curious about deep learning techniques for biomedical image analysis and have **basic Python knowledge** but may not be machine learning experts.

The notebook provides a step-by-step introduction to modern segmentation methods and walks through both **foreground–background segmentation** using a U-Net and **instance segmentation** using CellPose.  
You will explore how models learn to separate cells in microscopy images and how these techniques can be applied to **quantitative biological analysis** such as cell counting and morphology statistics.

---

## Lecture Content Overview

The lecture notebook is organized into the following sections:

1. **Setup and Environment**
   - Configuration for Google Colab  
   - Installation of required packages and imports  
   - Loading toy data from the **Data Science Bowl 2018 (DSB 2018)**

2. **Exploring the Dataset**
   - Visual inspection and understanding of input images and segmentation masks  

3. **Understanding the U-Net Architecture**
   - Step-by-step illustration of encoder, decoder, and skip connections  
   - Discussion of how U-Nets are used for **foreground–background segmentation**

4. **Training a U-Net Model**
   - Model training and performance visualization  
   - Evaluation using segmentation metrics  

5. **Using a Pretrained Model**
   - Loading and comparing performance against a pretrained U-Net  

6. **Instance Segmentation with CellPose**
   - Applying CellPose for cell detection and segmentation  
   - Performing downstream analysis such as **cell counting** and **statistical measurements**

---

## How to Use This Notebook

### Access via Google Colab

You can open the notebook directly in Google Colab by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kainmueller-Lab/HIDA_lecture_image_segmentation/blob/main/segmentation.ipynb)

---

### Setting up Google Colab

**Requirements**  
- A Google account with access to Google Drive.

**Important Notes**  
- Colab resources (CPU/GPU/RAM) are not guaranteed and may become temporarily unavailable.  
- Idle sessions longer than **90 minutes** or total runtimes exceeding **12 hours** will disconnect.  
- Unsaved work (e.g., model weights) will be lost when the session ends.

**Enable GPU Acceleration**  
Before running the notebook:
1. In Colab, go to the **Runtime** menu.  
2. Select **Change runtime type**.  
3. Under *Hardware accelerator*, choose **GPU**, then click **Save**.

---

### Manual Repository Setup (Alternative)

If you prefer to run the notebook locally or in your own Jupyter environment:

```bash
git clone https://github.com/Kainmueller-Lab/HIDA_lecture_image_segmentation.git
cd HIDA_lecture_image_segmentation
pip install -r requirements.txt
jupyter notebook segmentation.ipynb
