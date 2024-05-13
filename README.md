# Facial Expression Recognition with Delaunay Triangulation

This project aims to classify facial expressions using the CK+ dataset and compare the performance of different machine learning models trained on raw image data versus landmark features extracted using Delaunay triangulation.

## Table of Contents
1. Introduction
2. Installation
3. Usage
4. Results
5. Neural Network Model

## Introduction

Facial expression recognition is an important task in computer vision with various applications, such as emotion analysis, human-computer interaction, and affective computing. In this project, we explore the effectiveness of using Delaunay triangulation-based features alongside traditional raw image data for facial expression classification.

## Installation

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine:

   ```
   git clone https://github.com/Arshia-CE/FER-with-Triangulation.git
   ```

2. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook `Facial_Expression_Recognition.ipynb` in your Jupyter environment.
2. Run each cell in the notebook sequentially to execute the code and see the results.

## Results

The project evaluates several machine learning models on both raw image data and landmark features extracted using Delaunay triangulation. The following models are evaluated:

- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayesian Classification (GNB)
- Decision Tree
- Quadratic Discriminant Analysis (QDA)
- Random Forest

The performance of each model is compared in terms of accuracy and classification report.

## Neural Network Model

Additionally, a neural network model is introduced to outperform traditional machine learning models. The neural network architecture consists of several densely connected layers with dropout regularization and softmax activation for multiclass classification.

## Additional Notes

- Landmark features are extracted using Delaunay triangulation to capture facial geometry information.
- Standard scalar normalization is applied to both raw image data and landmark features for better model performance.

