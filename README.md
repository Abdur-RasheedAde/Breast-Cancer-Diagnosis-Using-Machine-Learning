# 🧬 Breast Cancer Diagnosis Using Machine Learning

## 📘 Overview

This project applies machine learning techniques to diagnose breast cancer using clinical features extracted from digitized images of fine needle aspirate (FNA) of breast masses. It demonstrates my ability to work with medical datasets, perform feature selection, and evaluate multiple classification models.

## 🎯 Objectives

- Preprocess and explore breast cancer diagnostic data
- Visualize feature correlations and distributions
- Train and evaluate multiple ML models for diagnosis
- Optimize model performance using GridSearchCV

## 📂 Dataset

- **Source**: Breast Cancer Wisconsin (Diagnostic) Dataset
- **Features**: Radius, Texture, Perimeter, Area, Smoothness, etc.
- **Target**: `diagnosis` (M = Malignant, B = Benign)
- **Preprocessing**:
  - Dropped irrelevant columns (`id`, `Unnamed: 32`)
  - Mapped diagnosis to binary (M=1, B=0)
  - Selected key features based on correlation

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`
- **Models**: Random Forest, KNN, SVM, MLPClassifier

## 📊 Exploratory Data Analysis

- Correlation heatmaps for mean, SE, and worst features
- Pairplots for feature relationships
- Feature selection based on correlation with diagnosis

## 🤖 Machine Learning Models

### 🔹 Models Trained

- **Random Forest**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Multi-layer Perceptron (MLPClassifier)**

### 🔸 Evaluation Metrics

- Accuracy
- Precision
- Recall
- Confusion Matrix

### 🏆 Model Optimization

- **GridSearchCV** used to tune hyperparameters of Random Forest
- RMSE calculated for best model

## 📈 Results

- **Best Accuracy**: Achieved with [insert best model name]
- **Optimized RMSE**: [insert value]
- **Key Features**: Radius, Perimeter, Area, Compactness, Concavity

## 📚 Key Learnings

- Handling medical diagnostic data
- Comparing multiple ML models for classification
- Using GridSearchCV for hyperparameter tuning
- Interpreting model performance in a clinical context

## 🚀 Future Work

- Apply deep learning models (e.g., CNNs) to image-based diagnosis
- Integrate genomic data for multi-modal prediction
- Deploy as a diagnostic support tool for clinicians

## 👨‍🔬 About Me

I'm **Abdur-Rasheed Abiodun Adeoye**, a data scientist with a strong interest in applying AI to biological and environmental challenges. This is part of my portfolio for in **Bioinformatics**.


## 📎 How to Run

```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-diagnosis-ml.git
cd breast-cancer-diagnosis-ml

# Run the Python script
python Diagnose_Breast_Cancer.py
