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

Welcome to **Learn_ML_DL** — a comprehensive, hands-on repository containing all learning modules, projects, lab preparations, and practical codes for **Machine Learning**, **Deep Learning**, **Computer Vision**, and **Applied AI**. 

All modules from across different branches (ML, DL, i_lab_prep, eatures_selection_models) have been cleanly unified here in the **main** branch.

---

## 📂 Complete Repository Structure

`	ext
Learn_ML_DL/
│
├── 01_Machine_Learning/
│   ├── 100_days_of_ml/             # Daily curriculum: Preprocessing, Regression, Classification, Ensembles
│   ├── algorithms/                 # KNN, Linear/Logistic/Multiple/Polynomial Regression implementations
│   │   └── regression/             # Linear, Multiple, Polynomial, Logistic regression scripts & visualizations
│   ├── feature_selection/          # Feature selection methods on wine quality dataset & reference guides
│   ├── python_basics/              # Python fundamentals & common libraries notes
│   └── roadmap/                    # Machine Learning visual roadmap
│
├── 02_Deep_Learning/
│   ├── tensors/                    # Tensor operations, shapes, and broadcasting in TensorFlow
│   ├── activation_functions/       # Sigmoid, ReLU, LeakyReLU, Tanh, Softmax implementations
│   └── CNN/                        # Convolutional Neural Networks for image classification
│
├── 03_Computer_Vision/
│   ├── main.py                     # Real-time object detection using YOLOv8 & OpenCV
│   ├── yolov8n.pt                  # Pre-trained YOLOv8 nano weights
│   └── samples/                    # Sample test images
│
├── 04_AI_Lab_Preparation/
│   ├── project_heart_disease/      # HealthPlus Heart Disease Prediction (Notebook, Script, Dataset, Report)
│   ├── datasets/                   # Cancer dataset, Heart disease datasets & archives
│   ├── reports/                    # Final exam lab reports, prompt histories & documentation
│   └── photos/                     # Lab diagram photos and workflow screenshots (1.png to 33.png)
│
├── 05_Datasets/                    # Shared datasets (Titanic, Boston Housing, KNN, Cancer, data.csv)
│
├── .gitignore                      # Clean Git ignore rules for ML, Jupyter, and heavy media
├── requirements.txt                # Python environment dependencies
└── README.md                       # Main repository documentation & guide
`

---

## 🚀 Key Modules & Curriculum

### 1️⃣ 01_Machine_Learning
- **Python Basics & Libraries**: Core syntax, NumPy, Pandas, Matplotlib, and Scikit-Learn basics.
- **Algorithms from Scratch**:
  - Linear Regression, Multiple Linear Regression, Polynomial Regression, Logistic Regression.
  - K-Nearest Neighbors (KNN).
- **100 Days of ML Curriculum**:
  - Missing Value Imputation (Mean/Median, Arbitrary, Frequent Category, KNN Imputer, Iterative Imputer).
  - Categorical Encoding (One-Hot, Ordinal, Mixed Variables).
  - Feature Transformations (Function Transformer, Power Transformer, ColumnTransformer, Pipelines).
  - Outlier Detection & Handling (Z-Score, IQR, Percentiles, Trimming, Capping).
  - Dimensionality Reduction (PCA step-by-step).
  - Ensemble Methods (Random Forest, AdaBoost, Gradient Boosting, Stacking & Blending).
  - Unsupervised Learning (K-Means clustering).
- **Feature Selection**: Filter, Wrapper, and Embedded methods tested on multi-attribute datasets.

### 2️⃣ 02_Deep_Learning
- **Tensor Operations**: Tensors, mathematical transformations, dimensions, and indexing.
- **Activation Functions**: Visualizations and implementations of Sigmoid, ReLU, Leaky ReLU, Tanh, Softmax.
- **CNN (Convolutional Neural Networks)**: Convolutions, pooling, feature maps, and image classification architectures.

### 3️⃣ 03_Computer_Vision
- **Real-Time Object Detection**: Real-time object and person detection using **YOLOv8** and **OpenCV / CVZone**.

### 4️⃣ 04_AI_Lab_Preparation
- **Heart Disease Prediction Project**: Complete ML pipeline with exploratory data analysis, clean dataset, model evaluation, and final report.
- **Lab Assignments & Reports**: Full course lab solutions, exam project reports, prompt history, and visual proofs.

---

## 🌿 Git Branching & Workflow

| Branch | Description |
|---|---|
| main | **Master Repository**: Central hub containing all consolidated learning modules, projects, and datasets. |
| eature/<name> | **Future Experiments**: Use separate feature branches for new exploratory projects, then merge to main. |

---

## 💻 Local Setup & Execution

### 1. Clone & Navigate
`ash
git clone https://github.com/SOURAVcse9/Learn_ML_DL.git
cd Learn_ML_DL
`

### 2. Activate Virtual Environment & Install Dependencies
`ash
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
`

### 3. Launch Notebooks or Scripts
`ash
jupyter notebook
# or run Computer Vision module
python 03_Computer_Vision/main.py
`

---

## 👤 Author

**SOURAV DEBNATH**  
- GitHub: [@SOURAVcse9](https://github.com/SOURAVcse9)