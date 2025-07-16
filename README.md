# 🚘 Car Brand Classification using Contrastive Learning & ReXNet150

This project implements a car brand classification system using a **contrastive learning** setup with triplet inputs (anchor, positive, negative) and a **fine-tuned ReXNet150** model. The workflow includes data visualization, model training, performance monitoring, Grad-CAM++ for interpretability, and a confusion matrix to evaluate predictions.

---
```
kaggle datasets download mohamedaziz15/cars-brands-in-egypt
unzip cars-brands-in-egypt.zip -d car_brands/
```
---

## 🧠 Key Features

- 🔀 **Triplet-based contrastive learning** to enhance feature discrimination between car brands.
- 🧱 **Custom PyTorch Dataset** class to dynamically form triplet batches.
- 🏋️ **Fine-tuned ReXNet150** with transfer learning for final classification.
- 📉 Tracked **training & validation metrics** (loss, accuracy, F1-score) with plotted learning curves.
- 🔥 Applied **Grad-CAM++** to visualize where the model focuses while making predictions.
- 📊 Built and visualized a **confusion matrix** to analyze class-wise performance.

---

## 📁 Dataset Structure

Organized as:
```
/car_brands/    
├── Audi/     
│   ├── img1.jpg     
│   └── …     
├── BMW/     
│   └── …     
└── Tesla/     
└── …           
```
---

## 🏗️ Pipeline Overview

### 1. Triplet Data Loader
- Anchor (Query) image
- Positive (same class) image
- Negative (different class) image  
📌 Enables contrastive learning by maximizing intra-class similarity and minimizing inter-class similarity.

### 2. Model Training
- Fine-tuned ReXNet150 on embeddings.
- Used F1-score-based checkpointing, LR scheduling, and early stopping.

---
### 3. Visualization
- **Train data visualization**
  
<img width="1607" height="1603" alt="image" src="https://github.com/user-attachments/assets/445caed9-306a-47d7-a623-847b03714cf8" />

- **Validation data visualization**
<img width="1607" height="1603" alt="image" src="https://github.com/user-attachments/assets/ec41e352-0417-4725-b3d8-14ef847fd2ff" />

- **Test data visualization**
<img width="1607" height="1603" alt="image" src="https://github.com/user-attachments/assets/8aaba55b-35ca-41a8-a341-eaad2ff9a4f4" />



  
- **Dataset imbalance** visualized via bar plots.
  <img width="1634" height="855" alt="image" src="https://github.com/user-attachments/assets/dca2b7d5-7159-4b80-8ae9-7b49460894f3" />


### 4. Evaluation & Explainability
- **Training Curves**: Visualized training vs. validation loss, accuracy, and F1-score across epochs.
<img width="833" height="470" alt="image" src="https://github.com/user-attachments/assets/3356ed41-f5e9-473d-80fb-eb0cd9710d9b" />
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/51f087a8-18fe-4b81-9ac6-046d63f85fa3" />
<img width="846" height="470" alt="image" src="https://github.com/user-attachments/assets/9d22041d-d863-4593-a0e0-de2f99538936" />


- **Grad-CAM++**: Heatmaps overlaid on input images to understand model decisions.
- <img width="1766" height="927" alt="image" src="https://github.com/user-attachments/assets/4a6549aa-8088-48ff-84c1-b2ece2544fd2" />

- **Confusion Matrix**: Matrix showing true vs. predicted labels with color-coded accuracy.

- <img width="1537" height="946" alt="image" src="https://github.com/user-attachments/assets/9d18d859-9d3a-43b8-aa4c-d7d6894dd446" />




## 🛠️ Tech Stack

- **PyTorch**, **torchvision**, **ReXNet150** via `timm`
- **Contrastive Learning** setup with triplet inputs
- **matplotlib**, **seaborn** for visualization
- **Grad-CAM++** from `pytorch-grad-cam`


