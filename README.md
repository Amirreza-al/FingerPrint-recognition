# Fingerprint Recognition System

## Overview

This project implements a biometric authentication system based on fingerprint recognition using deep learning techniques. It utilizes Siamese Neural Networks to create embeddings for fingerprint images, enabling accurate identity verification with low processing time. The system is designed to be integrated into fingerprint-based attendance devices and security systems.

## Files

- **fingerprint_recognition_train.ipynb**: Jupyter Notebook for training the fingerprint recognition model. It handles data loading from the FVC2002 dataset, implements a Siamese Network architecture with ResNet18 backbone, and trains the model using contrastive loss.

- **fingerprint_recognition_eval.ipynb**: Jupyter Notebook for evaluating the trained model's performance. It implements functions for registering fingerprints, recognizing identities from fingerprint images, and validating system accuracy.

- **siamese_model.py**: Python module containing the Siamese Network architecture implementation and the contrastive loss function used for training.

## Installation Requirements

Before running the notebooks, ensure you have the following dependencies installed:

torch
torchvision
numpy
matplotlib
seaborn
PIL (Pillow)
torchsummary
pickle
statistics
graphviz

You can install the required packages using pip:
pip install torch torchvision numpy matplotlib seaborn pillow torchsummary graphviz

## Dataset

The model was trained on the FVC2002 fingerprint dataset. The evaluation notebook expects fingerprint images with filenames formatted as "user_id.png".
('id' in image name should be 01, 02, 03, ...)

## Running the Code

### Training the Model

1. Ensure you have the FVC2002 dataset downloaded and stored locally.
2. Open and run the Jupyter Notebook:

  jupyter notebook fingerprint_recognition_train.ipynb

4. Execute each cell in sequence to:
   - Load and preprocess the dataset
   - Define and initialize the Siamese Network model
   - Train the model for the specified number of epochs
   - Evaluate model performance on the test set

### visualization of siamese network
![image](https://github.com/user-attachments/assets/40943a4e-95a4-49d2-ad38-2f46f2ac6378)

### training loss
there is no significant improvement after 10 epochs
![image](https://github.com/user-attachments/assets/6b97369e-68ba-484b-9e7f-6ca32d2752dc)

### Evaluating the Model

1. Prepare fingerprint images in the "user_finger_prints" directory
2. Open and run the evaluation notebook:

jupyter notebook fingerprint_recognition_eval.ipynb

3. Execute each cell to:
   - Load the trained model
   - Register fingerprints for known users
   - Test the system's recognition capabilities
   - View accuracy and processing time metrics

## Performance

The fingerprint recognition system achieves the following performance metrics:

- **Test Accuracy**: 93.75% on the FVC2002 dataset
- **Average Processing Time**: 0.0055 seconds (used Nvidia Tesla T4)
![image](https://github.com/user-attachments/assets/c433d577-6c46-4abf-867f-1317ac4cf207)

The system first extracts a 128-dimensional embedding from each fingerprint using the trained Siamese Network, then compares embeddings using Euclidean distance to determine identity.

## Model Architecture

The fingerprint recognition model uses a modified ResNet18 backbone with the following adaptations:

- Input layer modified to accept grayscale images
- Final fully connected layers replaced with a custom embedding network
- Trained using contrastive loss to minimize distance between same-identity fingerprints and maximize distance between different identities

## Contact

For questions or collaborations, please contact: Amirrexaalipour76@gmail.com
