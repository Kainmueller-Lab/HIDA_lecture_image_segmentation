# Machine Learning for Image Segmentation — HIDA Lecture

This lecture introduces **Machine Learning for Image Segmentation** using a hands-on, guided Jupyter Notebook designed for **Google Colab**.  

The notebook provides a step-by-step introduction to modern segmentation methods and walks through both **foreground–background segmentation** using a U-Net and **instance segmentation** using CellPose.  
You will explore how models learn to separate cells in microscopy images and how these techniques can be applied to **quantitative biological analysis** such as cell counting and morphology statistics.

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
