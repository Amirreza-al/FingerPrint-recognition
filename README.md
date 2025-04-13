# Fingerprint Recognition System

## Overview

This project implements a biometric authentication system based on fingerprint recognition using deep learning techniques. It utilizes Siamese Neural Networks to create embeddings for fingerprint images, enabling accurate identity verification with low processing time. The system is designed to be integrated into fingerprint-based attendance devices and security systems.

## Files

- **fingerprint_recognition_train.ipynb**: Jupyter Notebook for training the fingerprint recognition model. It handles data loading from the FVC2002 dataset, implements a Siamese Network architecture with ResNet18 backbone, and trains the model using contrastive loss.

- **fingerprint_recognition_eval.ipynb**: Jupyter Notebook for evaluating the trained model's performance. It implements functions for registering fingerprints, recognizing identities from fingerprint images, and validating system accuracy.

- **siamese_model.py**: Python module containing the Siamese Network architecture implementation and the contrastive loss function used for training.

## Installation Requirements

Before running the notebooks, ensure you have the following dependencies installed:
