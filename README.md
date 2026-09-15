# 🧠 Learn Machine Learning & Deep Learning (Learn_ML_DL)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV" />
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  <img src="https://img.shields.io/badge/Status-Active%20Learning-success?style=for-the-badge" alt="Status" />
</p>

---

## 📌 About The Repository

Welcome to **Learn_ML_DL** — a structured, hands-on repository documenting my journey through **Machine Learning**, **Deep Learning**, **Computer Vision**, and **Applied AI**. 

This repository serves as both a comprehensive learning path and a reference portfolio containing:
- Mathematical intuitions and code from scratch.
- Industry-standard pipelines using **Scikit-Learn**, **TensorFlow**, and **OpenCV / YOLOv8**.
- End-to-end projects, feature engineering techniques, lab preparations, and model evaluations.

---

## 📂 Repository Structure

`	ext
Learn_ML_DL/
│
├── 01_Machine_Learning/
│   ├── 100_days_of_ml/             # Daily curriculum: Preprocessing, Regression, Classification, Ensembles
│   ├── algorithms/                 # Core algorithms implemented from scratch & with Scikit-Learn (KNN, etc.)
│   ├── feature_selection/          # Advanced feature selection techniques & wine quality case study
│   └── roadmap/                    # ML learning roadmap & visualizations
│
├── 02_Deep_Learning/
│   ├── tensors/                    # Tensor operations, shapes, broadcasting, and manipulation
│   ├── activation_functions/       # Sigmoid, ReLU, LeakyReLU, Tanh, Softmax implementations & notes
│   └── CNN/                        # Convolutional Neural Networks for vision & classification
│
├── 03_Computer_Vision/
│   ├── main.py                     # Real-time object detection using YOLOv8 & OpenCV
│   ├── yolov8n.pt                  # Pre-trained YOLOv8 nano model weights
│   └── samples/                    # Sample test images and media
│
├── 04_AI_Lab_Preparation/
│   ├── datasets/                   # Lab course datasets
│   ├── files/                      # Academic reference files and solution scripts
│   └── photos/                     # Diagrams, visual proofs, and workflow snapshots
│
├── 05_Datasets/                    # Shared datasets (Titanic, Boston Housing, KNN, Data.csv, etc.)
│
├── .gitignore                      # Optimized Git ignore rules for ML, Jupyter, and media
├── requirements.txt                # Python environment dependencies
└── README.md                       # Project documentation & guide
`

---

## 🚀 Curriculum & Topics Covered

### 1️⃣ 01_Machine_Learning
- **Data Preprocessing & Feature Engineering**:
  - Missing Data Imputation: Mean/Median, Arbitrary, Frequent Category, KNN Imputer, Iterative Imputer.
  - Encoding: One-Hot Encoding, Ordinal Encoding, Mixed Variable Handling.
  - Transformations: Function Transformer, Power Transformer (Box-Cox, Yeo-Johnson), ColumnTransformer, Pipelines.
  - Outlier Detection & Removal: Z-score, IQR Method, Percentiles / Trimming / Capping.
  - Dimensionality Reduction: Principal Component Analysis (PCA) step-by-step.
- **Supervised Learning**:
  - Linear Regression: Simple, Multiple, Polynomial, Gradient Descent (Batch, Stochastic, Mini-batch).
  - Regularization: Ridge (L2), Lasso (L1), ElasticNet Regression.
  - Classification: Logistic Regression, Softmax, K-Nearest Neighbors (KNN).
  - Metrics: Confusion Matrix, Precision, Recall, F1-Score, ROC-AUC, Binary & Multi-class evaluation.
- **Ensemble Techniques**:
  - Bagging & Random Forests (OOB Score, Feature Importance).
  - Boosting: AdaBoost, Gradient Boosting.
  - Stacking & Blending.
- **Unsupervised Learning**:
  - K-Means Clustering (Elbow Method, Step-by-Step implementation).
- **Feature Selection**:
  - Filter, Wrapper, and Embedded methods tested on multi-dimensional datasets.

### 2️⃣ 02_Deep_Learning
- **Tensor Fundamentals**: Rank, dimensions, tensor indexing, mathematical operations in TensorFlow.
- **Activation Functions**: Detailed intuition and code for Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax.
- **Neural Network Architectures**: Multi-layer Perceptrons (MLP), Forward & Backward Propagation, Loss functions, Optimizers.
- **Convolutional Neural Networks (CNN)**: Feature extractors, filters, pooling layers, and classification heads.

### 3️⃣ 03_Computer_Vision
- **OpenCV**: Image transformations, frame capturing, color spaces, drawing utilities.
- **Real-Time Object Detection**: Integrated **YOLOv8** (You Only Look Once) with OpenCV and CVZone for real-time person/object detection and tracking.

### 4️⃣ 04_AI_Lab_Preparation
- University / academic laboratory preparation problems, search algorithms, heuristic searches, and practical lab tests with full step-by-step visual documentation.

---

## 🌿 Git Branching Strategy

To keep the repository organized as more projects are added in the future, the following branching strategy is recommended:

| Branch Name | Purpose |
|---|---|
| main | **Production / Portfolio Hub**: Polished, well-documented, clean codebase containing all aggregated modules. |
| eature/<topic-name> | **New Features / Experiments**: For new experimental modules (e.g., eature/nlp-transformers, eature/reinforcement-learning). |
| project/<project-name> | **Dedicated Projects**: For standalone end-to-end applications before merging into main. |
| i_lab_prep | **Academic Lab Work**: Specific branch for lab assignments and exams. |

> 💡 **Tip:** Always develop new topics in a feature branch, and merge into main using Pull Requests once tested and documented.

---

## 💻 Getting Started & Local Setup

### 1. Clone the Repository
`ash
git clone https://github.com/SOURAVcse9/Learn_ML_DL.git
cd Learn_ML_DL
`

### 2. Create and Activate a Virtual Environment
`ash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
`

### 3. Install Dependencies
`ash
pip install -r requirements.txt
`

### 4. Launch Jupyter Notebook / Lab
`ash
jupyter notebook
# or
jupyter lab
`

### 5. Run Object Detection (OpenCV + YOLOv8)
`ash
cd 03_Computer_Vision
python main.py
`

---

## 👤 Author

**SOURAV DEBNATH**
- GitHub: [@SOURAVcse9](https://github.com/SOURAVcse9)
- Repository: [Learn_ML_DL](https://github.com/SOURAVcse9/Learn_ML_DL)

---

<p align="center">⭐ If you find this repository helpful for your ML/DL journey, feel free to give it a star!</p>