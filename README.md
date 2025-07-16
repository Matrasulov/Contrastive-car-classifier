# 🚘 Car Brand Classification using Contrastive Learning & ReXNet150

This project implements a car brand classification system using a **contrastive learning** setup with triplet inputs (anchor, positive, negative) and a **fine-tuned ReXNet150** model. The workflow includes data visualization, model training, performance monitoring, Grad-CAM++ for interpretability, and a confusion matrix to evaluate predictions.

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

/car_brands/    
├── Audi/     
│   ├── img1.jpg     
│   └── …     
├── BMW/     
│   └── …     
└── Tesla/     
└── …           



       
