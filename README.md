# Ultrasound_Defect_Detection
# Ultrasound Signal-Based Defect Detection and Classification Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Notebooks](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter)](https://jupyter.org/)

##  Executive Summary

A professional portfolio project demonstrating an **end-to-end machine learning pipeline** for automated defect detection and classification in ultrasonic non-destructive testing (NDT). The project simulates realistic ultrasonic A-scan signals, applies advanced signal processing techniques, extracts physics-informed features, and compares classical machine learning with state-of-the-art deep learning models.

**Key Achievement:** Achieved **98.2% accuracy** using a 1D CNN on raw signals and **97.8% F1-score** with XGBoost on engineered features for multi-class defect classification across 5 material conditions.

---

##  Project Goals

- Simulate realistic ultrasonic A-scan signals with physical accuracy
- Apply professional signal preprocessing pipelines (filtering, denoising, scaling)
- Extract domain-specific time-domain, frequency-domain, and time-frequency features
- Perform comprehensive exploratory data analysis with PCA and t-SNE
- Benchmark classical ML algorithms (Random Forest, XGBoost, SVM, Logistic Regression, KNN)
- Implement deep learning architectures (1D CNN, LSTM, CNN+LSTM Hybrid, 2D CNN on spectrograms)
- Demonstrate model interpretability and professional visualization skills
- Deliver a portfolio-quality technical report suitable for hiring managers

---

##  Project Structure

- Simulate realistic ultrasonic A-scan signals with physical accuracy
- Apply professional signal preprocessing pipelines (filtering, denoising, scaling)
- Extract domain-specific time-domain, frequency-domain, and time-frequency features
- Perform comprehensive exploratory data analysis with PCA and t-SNE
- Benchmark classical ML algorithms (Random Forest, XGBoost, SVM, Logistic Regression, KNN)
- Implement deep learning architectures (1D CNN, LSTM, CNN+LSTM Hybrid, 2D CNN on spectrograms)
- Demonstrate model interpretability and professional visualization skills
- Deliver a portfolio-quality technical report suitable for hiring managers


##  Project Structure
Ultrasound_Defect_Detection/
│
├── data/
│ ├── raw/ # Raw simulated signals
│ │ ├── signals.csv # 5000 signals × 500 time points
│ │ └── signals.pkl # Pickled data for faster loading
│ └── processed/
│ └── features.csv # 22 engineered features
│
├── notebooks/
│ ├── 01_Signal_Simulation.ipynb # Physics-based signal generation
│ ├── 02_Signal_Preprocessing.ipynb # Filtering, denoising, scaling
│ ├── 03_Feature_Engineering.ipynb # Time/frequency/wavelet features
│ ├── 04_EDA.ipynb # Statistical analysis & visualization
│ ├── 05_Machine_Learning.ipynb # Classical ML models
│ └── 06_Deep_Learning.ipynb # CNN, LSTM, Hybrid architectures
│
├── models/
│ ├── best_model.pkl # Best performing ML model
│ ├── scaler.pkl # Feature scaler for inference
│ └── best_dl_model.h5 # Best deep learning model
│
├── images/ # Generated visualizations
│ ├── sample_signals.png
│ ├── preprocessing_demo.png
│ ├── fft_crack.png
│ ├── spectrogram_crack.png
│ ├── pca_scatter.png
│ ├── tsne_scatter.png
│ ├── ml_comparison.png
│ ├── confusion_matrix_best.png
│ ├── roc_curve.png
│ └── feature_importance.png
│
├── README.md # Project documentation (this file)
├── requirements.txt # Python dependencies
└── .gitignore
