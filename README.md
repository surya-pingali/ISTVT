# ISTVT: Interpretable Spatial-Temporal Video Transformer for Deepfake Detection

## Overview
The **Interpretable Spatial-Temporal Video Transformer (ISTVT)** is a deep learning architecture designed to achieve robust and accurate Deepfake detection by effectively exploiting joint spatial-temporal information in videos. Moving beyond the limitations of traditional frame-based methods (which struggle with low-quality videos and new manipulations like Neural Textures) and early video-based methods, ISTVT captures inter-frame inconsistencies and within-frame artifacts while providing clear, interpretable insights into its decision-making process. 

This repository contains the implementation of the ISTVT model developed as an EE656 Course Project.

---

## Key Features
* **Feature Extractor:** Utilizes a lightweight Convolutional Neural Network (CNN) consisting of Xception blocks to extract fine-grained, low-level textural feature maps from individual video frames.
* **Decomposed Spatial-Temporal Self-Attention:** Instead of joint attention, the model separates self-attention into two streams:
    * *Temporal Self-Attention:* Analyzes sequences across the temporal dimension to catch inter-frame inconsistencies.
    * *Spatial Self-Attention:* Analyzes within each frame to catch independent spatial artifacts.
    * This decomposition significantly reduces computational complexity from $O(T^2H^2W^2)$ to $O(T^2+H^2W^2)$.
* **Self-Subtract Mechanism:** Applied within the temporal attention module, this mechanism subtracts adjacent frame tensors ($I(2:,:,:) - I(1:-1,:,:)$) to remove redundant, changeless features and emphasize discriminative temporal differences.
* **Model Interpretability:** Employs Layer-wise Relevance Propagation (LRP) and Deep Taylor Decomposition to generate separate spatial and temporal relevance heatmaps. This visualizes the exact regions and frames the model uses to detect deepfakes, building trust through explainability.

---

## Repository Structure
The project repository is structured to support training, evaluation, and visualization:

* `train_CNN.py`: The primary entry point script for training the model.
* `test_time.py`: Script dedicated to testing the model and measuring inference performance.
* `loss_fn.py`: Contains the loss functions, including the Binary Cross-Entropy (BCE) loss used for Real/Fake classification.
* `visualize_feat_map.py` & `visualize_rel.py`: Scripts used to compute relevance scores and generate the spatial/temporal LRP heatmaps.
* `network/`: Directory containing the architectural building blocks:
    * `vivit/`, `fast_transformer_torch/`, `resnet3d/`: Modules for various spatial-temporal transformer and 3D CNN baseline components.
    * `xception.py` & `efficientnet_cdc.py`: CNN backbones used for the entry flow feature extraction.
    * `fastformer.py`, `unet.py`, `mesonet.py`: Additional architectures for comparison or feature processing.

---

## Dataset & Preprocessing
* **Dataset:** Evaluated on the high-quality FaceForensics++ (FF++) benchmark dataset, which includes DeepFakes, FaceSwap, Face2Face, and NeuralTextures.
* **Preprocessing:** The training pipeline processes videos by uniformly sampling 10 frames per video. 

---

## Training Details
The model was trained using the following hyperparameters and hardware:
* **Optimizer:** SGD with momentum.
* **Learning Rate:** 0.0005 with a linear warm-up phase.
* **Batch Size:** 64.
* **Epochs:** 100 (Max).
* **Hardware:** Nvidia RTX 3060 GPU and an AMD 5800H CPU.

---

## Results
* **Performance:** On a subset of the FF++ dataset, the ISTVT model achieved an impressive test accuracy of $\approx98.57\%$ with an epoch training loss dropping to 0.0086.
* **Analysis:** The self-subtract mechanism and decomposed attention blocks successfully highlight subtle artifacts in manipulated regions while retaining robust performance under various perturbations.
