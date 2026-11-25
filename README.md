# Skin Lesion Classification and Explainable AI

## Overview

This project investigates deep learning models for multi-class skin lesion classification using the HAM10000 dataset. It compares three convolutional neural network architectures and evaluates explainability techniques to understand how the selected model makes its predictions.

The main research focus is to compare model performance and investigate whether different Explainable AI (XAI) methods provide useful insights into model decision-making.

## Research Objectives

- Develop and compare multiple deep learning models for seven-class skin lesion classification.
- Evaluate models using accuracy, balanced accuracy, macro F1-score, and ROC-AUC.
- Investigate model interpretability using Grad-CAM and custom grid-based LIME.
- Compare explanation methods using a region-aware deletion experiment.
- Examine the limitations of post-hoc explanations in medical image classification.

## Dataset

This project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

- Total images: 10,015
- Number of diagnostic categories: 7
- Number of unique patients: 7,470

The seven diagnostic categories are:

| Code | Diagnosis |
|---|---|
| akiec | Actinic keratoses and intraepithelial carcinoma |
| bcc | Basal cell carcinoma |
| bkl | Benign keratosis-like lesions |
| df | Dermatofibroma |
| mel | Melanoma |
| nv | Melanocytic nevi |
| vasc | Vascular lesions |

The dataset is not class-balanced, with melanocytic nevi representing the majority class.

**Note:** The dataset images are not included in this repository. Refer to the original dataset source and its terms of use before downloading or using it.

## Models Compared

Three architectures were implemented and evaluated:

1. **Simple CNN** – A baseline convolutional neural network.
2. **ResNet50** – A deep residual network using skip connections.
3. **EfficientNet-B0** – A convolutional architecture using compound scaling.

### Test Set Results

| Metric | Simple CNN | ResNet50 | EfficientNet-B0 |
|---|---:|---:|---:|
| Accuracy | 0.5087 | 0.6723 | 0.6038 |
| Balanced Accuracy | 0.4217 | 0.5518 | 0.5824 |
| Macro F1-score | 0.3329 | 0.4530 | 0.4388 |
| Melanoma Recall | 0.5975 | 0.4654 | 0.4465 |
| Macro ROC-AUC | 0.8303 | 0.9024 | 0.8830 |

ResNet50 achieved the strongest overall results on accuracy, macro F1-score, and macro ROC-AUC. However, EfficientNet-B0 achieved higher balanced accuracy, while the Simple CNN achieved higher melanoma recall. Therefore, no single model performed best on every metric.

ResNet50 was selected for the explainability experiments based on its overall performance.

## Explainable AI Experiments

Two explanation approaches were investigated:

### Grad-CAM

Gradient-weighted Class Activation Mapping generates visual heatmaps highlighting image regions that contribute to a model's target-class prediction.

### Custom Grid-Based LIME

A custom grid-based perturbation approach was implemented using LIME's `LimeBase`. Each image was divided into an 8 × 8 grid, and random combinations of regions were hidden to estimate their influence on the model prediction.

This implementation should not be confused with the standard SLIC-based image segmentation commonly used in LIME image explanations.

### Region-Aware Deletion Experiment

Grad-CAM and custom grid-based LIME were compared by progressively removing important image regions and measuring the change in the model's target-class confidence.

A random-region baseline was also included.

The experiment used a fixed sample of 35 images, with five images from each diagnostic category.

| Method | Mean Deletion AUC |
|---|---:|
| Grad-CAM | 27.73 |
| Custom Grid-Based LIME | 38.96 |
| Random baseline | 23.91 |

Lower deletion AUC indicates a faster decrease in target-class confidence under this particular perturbation protocol.

Grad-CAM had significantly lower deletion AUC than LIME. However, it was not significantly different from the random baseline. This means the experiment does not establish that Grad-CAM identifies medically meaningful regions better than random selection.

## Important Limitations

- The XAI comparison was conducted on a small sample of 35 images.
- The deletion experiment used black-region replacement, which can create unrealistic images.
- A decrease in model confidence does not necessarily mean that an explanation identifies medically meaningful features.
- The model's classification performance is not sufficient for clinical diagnosis.
- Class imbalance affects the interpretation of aggregate performance metrics.

This project is an academic research experiment and is not intended for clinical use.

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Grad-CAM
- LIME

## Repository Structure

The repository will contain notebooks, reusable source code, evaluation results, visualizations, and project documentation.

## Author

Developed as an academic project exploring deep learning and Explainable AI for medical image classification.
