# TrashNet Image Classifier

This project implements a deep learning model to classify images of trash into categories such as cardboard, glass, metal, paper, plastic, and others, using the TrashNet dataset. The model architecture is based on a simple convolutional neural network (CNN) built in PyTorch, and the project includes everything needed for training, evaluation, and deployment.

## Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup](#setup)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Acknowledgements](#acknowledgements)

## Project Overview

This repository is part of a pipeline that automates the classification of trash images, aiming to support environmental efforts for sorting recyclables. The main components include data loading, data preprocessing, model training, and evaluation.

### Model Architecture

The model uses two convolutional layers, max-pooling, and fully connected layers with dropout regularization. The final layer provides the classification output for six trash categories.

## Requirements

To set up and run the project, you'll need the following libraries and packages:

- Python 3.12
- [PyTorch](https://pytorch.org/)
- [torchvision](https://pytorch.org/vision/stable/index.html)
- scikit-learn
- matplotlib
- seaborn
- and any other packages listed in `requirements.txt`

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Setup
Clone this repository and navigate to the project directory:
```bash
git clone https://github.com/your-username/your-repo-name.git
```
```bash
cd your-repo-name
```
Set up your environment:

Install dependencies: Run ```bash pip install -r requirements.txt```
Download the TrashNet dataset from Hugging Face: [TrashNet on Hugging Face](https://huggingface.co/datasets/garythung/trashnet)

## Training the Model
To train the model, execute:
```bash
python modelling-DL.ipynb
```
This script will handle data loading, augmentations, and training. Adjust hyperparameters in train.py to experiment with different model configurations.

## Evaluation
Run the evaluation script to generate accuracy metrics, a classification report, and a confusion matrix:
```bash
python modelling-DL.ipynb
```
The model’s accuracy, precision, recall, and F1-score will be printed for each trash category. Additionally, the confusion matrix provides insights into which categories are most often misclassified.

## Inference
To make predictions with the trained model, load it using:
```bash
import torch
from model import SimpleCNN

# Load model and set to evaluation mode
model = SimpleCNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Code to process an input image and get predictions
```

## Acknowledgements
Special thanks to the creators of the TrashNet dataset and Hugging Face for dataset hosting, as well as the PyTorch community for supporting deep learning research.

