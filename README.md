# Beyond Accuracy: Comparative Analysis of Neural Network Architectures, Explainability, and Robustness in Yoga Pose Classification

This repository contains the code and report for a deep learning project that goes beyond traditional accuracy metrics to provide a holistic evaluation of various neural network architectures for yoga pose classification. This project systematically compares multiple well-known neural network architectures, focusing not only on **accuracy** but also on **explainability** and **robustness**.

## 🧘‍♀️ Project Goal

The primary goal of this project is to move beyond the standard accuracy metric and explore:

* **Comparative Analysis**: A systematic evaluation of accuracy, hyperparameters, and other performance metrics across different neural network architectures.
* **Explainability**: Utilization of Grad-CAM and Grad-CAM++ to visualize and understand the focus of the models, ensuring they are making decisions based on meaningful features.
* **Robustness to Gaussian Noise**: Testing the models' performance under varying levels of Gaussian noise to assess their resilience and reliability in non-ideal conditions.

## Dataset

The project uses a curated dataset of five yoga poses:
* Downdog
* Goddess
* Plank
* Tree
* Warrior2

The dataset is split into training, validation, and test sets. To improve generalization and reduce overfitting, the training data is augmented with techniques like random resized cropping, horizontal flipping, rotation, and color jittering.

## 🧠 Models Explored

This project explores three distinct families of models:

1.  **Convolutional Neural Networks (CNNs)**: We used transfer learning for established CNN architectures, replacing the final classifier to fit the five yoga poses. The analyzed architectures include:
    * **AlexNet**: An early deep learning model with sequential convolutional and fully connected layers.
    * **VGG16**: A deep, simple design featuring stacked 3x3 convolutions.
    * **GoogLeNet**: Utilizes Inception modules for efficient multi-scale feature capture.
    * **ResNet18**: Employs residual connections to facilitate the training of deeper networks.
    * **CustomCNN**: A lightweight CNN built from scratch to serve as a baseline.

2.  **Vision Transformers (ViTs)**: ViTs treat an image as a sequence of patches, which are then processed by a Transformer encoder, allowing for global context modeling. The project explores:
    * **ViT-Base**: This model splits images into patches, flattens them, and uses self-attention to capture global relationships.
    * **DINOv2**: A self-supervised ViT pre-trained on large unlabeled datasets, providing strong and transferable visual features.

3.  **Graph Neural Networks (GNNs)**: GNNs are used to analyze the body's structural representation by first extracting 2D keypoints (joints) and then connecting them to form a skeletal graph. The model explored is:
    * **PoseGNN**: This network classifies poses by learning the relational patterns between body parts, with a focus on alignment and posture.

## 🛠️ Getting Started

### Prerequisites

You will need Python 3 and the following libraries to run the project. You can install them using pip:

```bash
pip install torch torchvision tqdm matplotlib scikit-learn seaborn numpy timm mediapipe opencv-python torch-geometric
