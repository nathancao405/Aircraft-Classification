# Aircraft Classification using Deep Learning

## Overview

This project builds an end-to-end deep learning image classifier capable of identifying **100 different aircraft variants** using TensorFlow and MobileNetV2. The model uses transfer learning and fine-tuning to classify aircraft images from the FGVC Aircraft Dataset.

This project demonstrates the complete machine learning workflow, including:

- Data preprocessing
- Exploratory data analysis (EDA)
- TensorFlow data pipelines
- Data augmentation
- Transfer learning with MobileNetV2
- Model fine-tuning
- Performance evaluation
- Prediction visualization

---

## Dataset

**Dataset:** FGVC Aircraft Dataset

- 100 aircraft variants
- Approximately 10,000 aircraft images
- Images divided into training, validation, and testing sets

The dataset can be downloaded from the official FGVC Aircraft website.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Pillow
- Scikit-learn
- Jupyter Notebook

---

## Model Architecture

The classifier uses **MobileNetV2** pretrained on ImageNet.

Model pipeline:

- Data Augmentation
- MobileNetV2 Feature Extractor
- Global Average Pooling
- Dropout
- Dense Softmax Output Layer (100 classes)

Transfer learning was first performed by freezing the pretrained network. The final 30 layers were then unfrozen and fine-tuned using a smaller learning rate to improve aircraft-specific classification performance.

---

## Results

Final Test Accuracy:

**29.7%**

Considering the dataset contains **100 visually similar aircraft variants**, this demonstrates the difficulty of fine-grained image classification.

---

## Repository Structure

```
Aircraft-Classification/
│
├── Aircraft_Classification.ipynb
├── README.md
├── requirements.txt
├── aircraft_classifier.keras
├── class_names.txt
└── figures/
```

---

## Future Improvements

Potential improvements include:

- Training on higher-resolution images
- Testing EfficientNet or ResNet architectures
- Hyperparameter tuning
- Larger batch sizes
- More advanced data augmentation
- Class imbalance analysis

---

## Author

Nathan Cao
