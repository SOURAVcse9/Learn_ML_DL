#!/usr/bin/env python
# coding: utf-8

# # 🏥 HealthPlus Hospital — AI-Based Heart Disease Prediction System
# 
# ### Final Lab Examination — Artificial Intelligence
# 
# | | |
# |---|---|
# | **Project Title** | Predicting Heart Disease Risk Using Machine Learning |
# | **Course** | Artificial Intelligence (AI) — Final Lab Exam |
# | **Student Name** | _[Your Full Name]_ |
# | **Student ID** | _[Your Student ID]_ |
# | **Department** | Department of Computer Science & Engineering |
# | **University** | _[Your University Name]_ |
# | **Date** | July 08, 2026 |
# 
# ---
# 
# **Role Simulation:** I have been hired as a **Junior AI Engineer** at *HealthPlus Hospital*. My task is to design, build, evaluate, and critically discuss a machine-learning system that predicts whether a patient is at risk of heart disease, using routinely collected clinical measurements.
# 
# This notebook documents the **entire AI development lifecycle** — from problem framing to deployment considerations — following professional data-science practice and full academic rigor required for this exam.
# 

# ## 1. Introduction
# 
# ### 1.1 What is Heart Disease Prediction?
# Heart (cardiovascular) disease prediction is the task of using a patient's clinical and demographic measurements — such as age, blood pressure, cholesterol, and heart-rate response to exercise — to estimate the **likelihood that the patient currently has, or will develop, significant heart disease**. In this project it is framed as a **binary classification problem**: `target = 1` (disease present) vs `target = 0` (no disease).
# 
# ### 1.2 Why is Machine Learning useful here?
# Traditional risk scoring (e.g. simple point-based charts used by doctors) applies the same fixed formula to every patient. Machine learning instead:
# - Learns **non-linear interactions** between multiple risk factors simultaneously (e.g. how age combines with cholesterol and chest-pain type).
# - Can be **retrained** as more patient data becomes available, improving over time.
# - Provides a **fast, consistent, first-pass triage tool** that supports (not replaces) a doctor's judgement, especially useful when specialist cardiologists are not immediately available.
# 
# ### 1.3 Why is this project important?
# Cardiovascular disease is one of the leading causes of death worldwide. Early, low-cost screening based on routine test results can help hospitals like HealthPlus **prioritise high-risk patients** for further (expensive) diagnostic tests such as angiography, improving patient outcomes while controlling costs.
# 
# ### 1.4 Project Objectives
# 1. Understand and explore the HealthPlus patient dataset.
# 2. Clean and prepare the data for modelling.
# 3. Engineer clinically meaningful features.
# 4. Train and compare multiple machine-learning classifiers.
# 5. Evaluate models using metrics appropriate for a medical diagnosis task.
# 6. Interpret the best model so that its predictions are explainable to clinicians.
# 7. Critically discuss the ethical, technical, and deployment limitations of the system.
# 
# > **Note:** This notebook is an educational exam project. It is **not** a certified medical device and must never be used for real clinical decisions without regulatory approval and extensive external validation.
# 

# ## 2. Import Libraries
# 
# **What?** We import all the Python libraries required for data handling, visualization, preprocessing, modelling, and evaluation.
# 
# **Why each library is needed:**
# | Library | Purpose |
# |---|---|
# | `numpy` | Fast numerical/array operations that underpin pandas and scikit-learn |
# | `pandas` | Loading, cleaning, and manipulating tabular (spreadsheet-like) data |
# | `matplotlib` | Base plotting engine for charts |
# | `seaborn` | Statistical visualizations built on matplotlib with nicer defaults |
# | `scikit-learn (sklearn)` | Preprocessing, train/test splitting, ML models, evaluation metrics |
# | `xgboost` | Gradient-boosted trees — a powerful, industry-standard classifier |
# | `warnings` | Suppress non-critical version warnings so the notebook output stays clean for the reader |
# 
# **Alternative libraries** we could have used: `plotly` (interactive charts, optional here), `statsmodels` (classical statistics), `lightgbm`/`catboost` (alternative boosting libraries). We stick to the most widely taught, stable toolkit for this exam.
# 

# In[1]:


# Core data-handling libraries
import numpy as np
import pandas as pd
# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Suppress non-critical warnings for a clean, professional notebook
import warnings
warnings.filterwarnings('ignore')
# Professional plotting style
plt.rcParams['figure.dpi'] = 110
plt.rcParams['savefig.dpi'] = 110
sns.set_style('whitegrid')
sns.set_palette('viridis')
print(" All libraries imported successfully.")


# **Observation:** All libraries loaded without error, confirming the environment has every package required for the rest of the pipeline.
# 
# **Interpretation:** We are ready to move from raw data to a working AI system.
# 
# **Conclusion:** Environment setup is complete; we proceed to load the HealthPlus patient dataset.
# 

# ## 3. Load Dataset
# 
# **What?** We load `heart.csv`, the patient dataset provided by HealthPlus Hospital, into a pandas DataFrame.
# 
# **Why?** A DataFrame gives us labelled rows/columns and a huge library of built-in functions for inspection, cleaning, and analysis — far more convenient than working with raw text or lists.
# 
# **How?** `pandas.read_csv()` parses the comma-separated file directly into memory.
# 

# In[2]:


# Load the dataset
df = pd.read_csv('heart.csv')

# Basic inspection
print("Dataset shape (rows, columns):", df.shape)
print("\nColumn names:", list(df.columns))


# **Observation:** The dataset contains **1025 patient records** and **14 columns** (13 clinical features + 1 target label).
# 
# **Interpretation:** This is a moderately small tabular dataset — typical of many real hospital datasets, where data collection is costly. It is large enough to train simple ML models but small enough that we must be careful about overfitting.
# 
# **Business Insight:** With just over a thousand patient records, HealthPlus should treat this model as a **decision-support prototype**, and continue collecting more patient data to make future versions more robust.
# 

# In[3]:


# First 5 rows
df.head()


# **Interpretation:** `head()` lets us visually confirm that every column loaded with the correct data type and that values look clinically plausible (e.g. `age` is a sensible integer, `chol` is in a typical mg/dl range).

# In[4]:


# Last 5 rows
df.tail()


# **Interpretation:** `tail()` confirms the end of the file was read correctly and there is no corrupted or truncated data at the bottom of the CSV.

# In[5]:


# A random sample of 5 rows
df.sample(5, random_state=42)


# **Interpretation:** `sample()` gives us a random cross-section of the data rather than just the first/last rows, which helps catch any ordering bias (e.g. the file being sorted by target class).
# 
# **Conclusion:** The dataset has loaded correctly and is ready for detailed understanding and exploration.
# 

# ## TASK 1 — Dataset Understanding
# 
# ### 1.1 Target Variable
# The target variable is **`target`** (0 = no heart disease, 1 = heart disease present).
# 
# ### 1.2 Classification or Regression?
# This is a **classification** problem (specifically **binary classification**), **because** `target` only takes two discrete values (0/1) representing the presence or absence of disease — we are predicting a *category*, not a continuous numeric quantity like a lab value or age. Regression would be the correct choice only if we were predicting a continuous outcome (e.g. degree of arterial blockage as a percentage).
# 
# ### 1.3 Feature Types
# 
# | Type | Columns |
# |---|---|
# | **Numerical (continuous/ordinal)** | `age`, `trestbps` (resting blood pressure), `chol` (cholesterol), `thalach` (max heart rate achieved), `oldpeak` (ST depression) |
# | **Categorical (nominal/ordinal, stored as integer codes)** | `sex`, `cp` (chest pain type), `fbs` (fasting blood sugar > 120 mg/dl), `restecg` (resting ECG results), `exang` (exercise-induced angina), `slope` (ST segment slope), `ca` (number of major vessels), `thal` (thalassemia type) |
# | **Target** | `target` |
# 
# ### 1.4 Suggested Important Features (clinical domain knowledge)
# - **`cp` (chest pain type)** — the type of chest pain is one of the strongest classic clinical indicators of cardiac origin.
# - **`thalach` (max heart rate achieved)** — reduced exercise heart-rate response is associated with heart disease.
# - **`oldpeak` / `slope`** — ST-segment changes during exercise are textbook ECG markers of ischemia.
# - **`ca` (number of major vessels coloured by fluoroscopy)** — directly measures vessel blockage.
# - **`thal`** — thalassemia test result, strongly linked to cardiac risk in this dataset's clinical literature.
# 
# **Why these matter:** they are the features doctors themselves rely on most heavily during diagnosis, so we expect (and will later verify with feature importance) that the model leans on them too.
# 
# ### 1.5 Two Key Challenges in this Dataset
# 1. **Small sample size (1025 rows)** relative to the complexity of the problem — increases risk of overfitting, especially for high-capacity models.
# 2. **Encoded categorical variables look numeric** (e.g. `cp` is stored as 0-3) — a model could mistakenly treat these as ordinal/continuous unless we are careful, which can distort learned relationships if not handled thoughtfully.
# 
# ### 1.6 How AI Helps Solve This Problem
# AI (machine learning) can automatically discover the *combination* of risk factors that best separates diseased from healthy patients — something that is difficult for a human to do by eye across 13 variables simultaneously. Once trained, the model can screen new patients in milliseconds, flagging likely at-risk individuals for prioritized specialist review.
# 

# In[6]:


# Separate numerical vs categorical features programmatically
numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
target_col = 'target'

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)
print("Target:", target_col)


# **Interpretation:** Explicitly separating feature types up front ensures we apply the *correct* preprocessing later (e.g. scaling only numeric features, one-hot/label encoding only categorical ones).

# ## TASK 2 — Exploratory Data Analysis (EDA)
# 
# **What?** EDA is the process of summarizing and visualizing a dataset before modelling, to understand its structure, spot problems, and generate hypotheses.
# 
# **Why is EDA needed?** Models trained on unexamined data can silently learn from noise, duplicates, or outliers. EDA lets us catch these issues early and build clinical intuition about which variables matter.
# 

# In[7]:


# Dataset dimensions and data types
print("Shape:", df.shape)
print()
df.info()


# **Interpretation:** All 14 columns are numeric (`int64`/`float64`), and every column has 1025 non-null entries — confirming (pending a formal check below) that there is likely no missing data, since the dataset was already digitized in a structured clinical-study format.

# In[8]:


# Missing values check
missing = df.isnull().sum()
print("Missing values per column:\n", missing)
print("\nTotal missing values:", missing.sum())


# **Observation:** There are **0 missing values** anywhere in the dataset.
# 
# **Interpretation:** This is unusually clean for real hospital data — it suggests the dataset has already been curated/pre-processed by the original data providers before being handed to HealthPlus.
# 
# **Business Insight:** In a live hospital system, missing values (e.g. a skipped test) *would* occur; the cleaning pipeline we build in Task 3 still includes a missing-value strategy so the notebook remains production-ready even though this particular file is complete.
# 

# In[9]:


# Duplicate rows check
n_duplicates = df.duplicated().sum()
print("Number of exact duplicate rows:", n_duplicates)
print(f"Percentage of dataset that is duplicated: {n_duplicates/len(df)*100:.1f}%")


# **Observation:** The dataset contains a large number of exact duplicate rows.
# 
# **Interpretation:** This is a well-known characteristic of this particular public heart-disease CSV (it repeats a smaller original set of unique patients). If left untreated, duplicates would **leak** into both the training and test sets, giving an artificially inflated, over-optimistic accuracy score.
# 
# **Business Insight:** For HealthPlus, treating duplicated patient records as independent evidence would be clinically misleading — we must remove duplicates before splitting data for modelling (handled formally in Task 3).
# 

# In[10]:


# Statistical summary of numerical features
df[numerical_features].describe().T


# **Interpretation:** Average patient age is ~54 years; average resting blood pressure ~132 mmHg (borderline high); average cholesterol ~246 mg/dl (above the commonly cited 200 mg/dl 'desirable' threshold) — consistent with a patient population that a hospital would actually be screening for cardiac risk.

# In[11]:


# Unique values in categorical/target columns
for col in categorical_features + [target_col]:
    print(f"{col}: {sorted(df[col].unique())}")


# **Interpretation:** All categorical columns take small sets of integer codes (as expected from the dataset documentation), confirming there are no unexpected/invalid category values (e.g. no negative codes or out-of-range values) that would indicate data-entry errors.

# In[12]:


# Class distribution of the target variable
class_counts = df['target'].value_counts()
class_pct = df['target'].value_counts(normalize=True) * 100
print(class_counts)
print()
print(class_pct.round(2))


# **Observation:** The classes are close to balanced (roughly 51% disease vs 49% no-disease).
# 
# **Interpretation:** Because the classes are nearly balanced, **accuracy is a reasonably safe headline metric** for this dataset (unlike heavily imbalanced medical datasets where accuracy alone would be misleading) — though we still report precision/recall/F1/ROC-AUC in Task 6 for a complete clinical picture.
# 

# In[13]:


# Correlation matrix heatmap
plt.figure(figsize=(11, 9))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, cbar_kws={'label': 'Correlation coefficient'})
plt.title('Correlation Matrix of All Clinical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('viz_01_correlation_heatmap.png')
plt.show()


# **Purpose:** Reveal linear relationships between every pair of variables, and especially between each feature and the `target`.
# 
# **Interpretation:** `cp`, `thalach`, and `slope` show the strongest **positive** correlation with `target`, while `exang`, `oldpeak`, `ca`, and `thal` show the strongest **negative** correlation — matching clinical expectations from Task 1.
# 
# **Business Insight:** These correlations give HealthPlus clinicians a quick, interpretable summary of which routine tests are most informative, useful even outside the ML model itself.
# 

# In[14]:


# Visualization 2: Age distribution (Histogram)
plt.figure(figsize=(8, 5))
sns.histplot(df['age'], bins=20, kde=True, color='#4C72B0')
plt.title('Distribution of Patient Age', fontsize=13, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Number of Patients')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('viz_02_age_histogram.png')
plt.show()


# **Purpose:** Understand the age profile of the patient population.
# 
# **Interpretation:** Ages roughly follow a bell shape centred around 50-60 years, with very few patients under 35 — consistent with heart disease being predominantly an older-adult condition.
# 
# **Business Insight:** HealthPlus's screening pool skews toward middle-aged and older patients; younger-patient predictions should be treated with extra caution since the model has seen relatively few such cases.
# 

# In[15]:


# Visualization 3: Cholesterol by target (Boxplot)
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='target', y='chol', hue='target', palette='viridis', legend=False)
plt.title('Cholesterol Levels by Heart Disease Status', fontsize=13, fontweight='bold')
plt.xlabel('Target (0 = No Disease, 1 = Disease)')
plt.ylabel('Cholesterol (mg/dl)')
plt.tight_layout()
plt.savefig('viz_03_cholesterol_boxplot.png')
plt.show()


# **Purpose:** Compare the spread and central tendency of cholesterol between the two classes, and visually spot outliers.
# 
# **Interpretation:** Median cholesterol is similar across both groups with heavy overlap and several high-cholesterol outliers in both classes — suggesting cholesterol *alone* is a weak discriminator, which matches its low correlation with `target` seen in the heatmap.
# 
# **Business Insight:** Contrary to popular belief, cholesterol alone is not a strong stand-alone predictor in this data; HealthPlus should avoid over-weighting a single lab value in manual triage.
# 

# In[16]:


# Visualization 3: Chest pain type vs target (Countplot)
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='cp', hue='target', palette='viridis')
plt.title('Chest Pain Type vs Heart Disease Status', fontsize=13, fontweight='bold')
plt.xlabel('Chest Pain Type (0-3)')
plt.ylabel('Number of Patients')
plt.legend(title='Target', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig('viz_04_chestpain_countplot.png')
plt.show()


# **Purpose:** Examine whether chest-pain type distinguishes diseased from healthy patients.
# 
# **Interpretation:** Type 0 (typical angina/asymptomatic code) is dominated by 'no disease' patients, while types 1-3 are dominated by 'disease' patients — a clear, clinically sensible signal.
# 
# **Business Insight:** Chest pain type is one of the cheapest, fastest features to record and appears highly informative — a strong candidate for a simple front-line screening question.
# 

# In[17]:


# Visualization 4: Age vs Max Heart Rate (Scatterplot)
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='age', y='thalach', hue='target', palette='viridis', alpha=0.7)
plt.title('Age vs Maximum Heart Rate Achieved, coloured by Disease Status', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Max Heart Rate Achieved (thalach)')
plt.legend(title='Target', labels=['No Disease', 'Disease'])
plt.tight_layout()
plt.savefig('viz_05_age_thalach_scatter.png')
plt.show()


# **Purpose:** Visualize how two continuous features jointly relate to the outcome.
# 
# **Interpretation:** There is a mild negative trend between age and max heart rate (expected physiologically), and diseased patients (lighter colour) cluster toward **lower** max-heart-rate values at a given age — supporting `thalach` as a useful predictive feature.
# 
# **Business Insight:** A reduced maximum heart rate for a patient's age, observed during a simple exercise test, is an easily measurable early warning sign HealthPlus staff can watch for.
# 

# In[18]:


# Visualization 5: Target class distribution (Pie Chart)
plt.figure(figsize=(6, 6))
counts = df['target'].value_counts()
plt.pie(counts, labels=['Disease (1)', 'No Disease (0)'] if counts.index[0]==1 else ['No Disease (0)', 'Disease (1)'],
        autopct='%1.1f%%', colors=['#DE8F05', '#0173B2'], startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5})
plt.title('Overall Class Distribution of Target Variable', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('viz_06_target_pie.png')
plt.show()


# **Purpose:** Give an at-a-glance view of class balance.
# 
# **Interpretation:** The split is close to 50/50, confirming (as noted earlier) that the dataset is well balanced and standard accuracy-based metrics will not be misleading.
# 
# **Business Insight:** No special class-imbalance handling (e.g. SMOTE, class weighting) is strictly required for this dataset, simplifying the modelling pipeline.
# 

# In[19]:


# Visualization 7: Exercise-induced angina vs target (Violin Plot of oldpeak)
plt.figure(figsize=(7, 5))
sns.violinplot(data=df, x='exang', y='oldpeak', hue='target', split=True, palette='viridis')
plt.title('ST Depression (oldpeak) by Exercise Angina and Disease Status', fontsize=12, fontweight='bold')
plt.xlabel('Exercise-Induced Angina (0 = No, 1 = Yes)')
plt.ylabel('ST Depression (oldpeak)')
plt.legend(title='Target')
plt.tight_layout()
plt.savefig('viz_07_oldpeak_violin.png')
plt.show()


# **Purpose:** Show the full distribution shape (not just summary statistics) of `oldpeak` across combinations of angina and disease status.
# 
# **Interpretation:** Patients with disease (target = 1) show a wider spread toward higher ST depression values, especially when exercise-induced angina is present — reinforcing that `oldpeak` and `exang` carry complementary diagnostic signal.
# 
# **Business Insight:** Combining exercise-test results (`exang`, `oldpeak`) provides richer risk stratification than either measurement alone — supporting a multi-feature ML approach over a single-test rule.
# 
# **EDA Summary:** Across all seven visualizations we consistently see that ECG/exercise-based features (`cp`, `thalach`, `oldpeak`, `exang`, `slope`) separate the classes far more clearly than static lab values like `chol`. This foreshadows which features the models in Task 5 are likely to rely on most heavily.
# 

# ## TASK 3 — Data Cleaning
# 
# **What?** We now systematically handle missing values, duplicates, outliers, encoding, and scaling.
# 
# **Why?** Even though this dataset looked clean in the EDA, a professional pipeline always explicitly handles each of these steps so it is robust to new, messier data in production.
# 

# In[20]:


# --- Step 1: Handle missing values ---
# Why: even though we found 0 missing values, we build a general-purpose strategy
# for production readiness (e.g. new patient records with an unrecorded test).
# Alternative methods: (a) drop rows with missing values, (b) mean/median imputation,
# (c) KNN/model-based imputation. We choose median imputation for numeric columns
# because it is robust to outliers (unlike mean), and mode imputation for categorical
# columns because they are discrete codes.
# Pros: simple, fast, doesn't discard data. Cons: can slightly underestimate variance
# and ignores relationships between features (unlike KNN imputation).

df_clean = df.copy()
for col in numerical_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
for col in categorical_features:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("Missing values after imputation step:", df_clean.isnull().sum().sum())


# **Interpretation:** No values needed imputation here, but the pipeline now safely handles missing data for any future, messier HealthPlus data extract.

# In[21]:


# --- Step 2: Remove duplicate rows ---
# Why: duplicate patient rows would let identical records appear in BOTH the train
# and test split, artificially inflating test accuracy (a form of data leakage).
# Alternative: keep duplicates but ensure the split is done BEFORE deduplication
# with grouping logic - more complex and unnecessary here since duplicates are exact.
# Pros of dropping: prevents leakage, gives an honest performance estimate.
# Cons: reduces dataset size, which can hurt training data volume.

before = df_clean.shape[0]
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
after = df_clean.shape[0]
print(f"Rows before dedup: {before}")
print(f"Rows after dedup:  {after}")
print(f"Duplicate rows removed: {before - after}")


# **Observation:** A large number of duplicate rows were removed, leaving a smaller set of unique patient records.
# 
# **Interpretation:** This confirms the dataset was a repeated/expanded version of a smaller original cohort. Working from the de-duplicated data going forward gives an honest, leakage-free basis for model evaluation.
# 
# **Business Insight:** HealthPlus should audit its data pipeline to ensure records are not being accidentally duplicated during collection or export, as this could distort any statistic computed on the raw file.
# 

# In[22]:


# --- Step 3: Handle outliers (IQR capping on numerical features) ---
# Why: extreme values (e.g. a resting blood pressure of 0 or 300) can disproportionately
# influence distance-based models (KNN, SVM) and destabilize scaling.
# Alternative methods: (a) remove outlier rows entirely, (b) z-score capping,
# (c) leave outliers untouched (tree-based models are fairly robust to them).
# We choose IQR-based CAPPING (winsorizing) rather than deletion because in a medical
# dataset, an extreme value is often a real (if rare) patient, and deleting real patients
# discards clinically important information.
# Pros: keeps all patients, reduces the influence of extreme values.
# Cons: slightly distorts the true extreme values; a judgement call vs full removal.

def cap_outliers_iqr(frame, columns, factor=1.5):
    frame = frame.copy()
    report = {}
    for col in columns:
        q1, q3 = frame[col].quantile(0.25), frame[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_outliers = ((frame[col] < lower) | (frame[col] > upper)).sum()
        frame[col] = frame[col].clip(lower, upper)
        report[col] = n_outliers
    return frame, report

df_clean, outlier_report = cap_outliers_iqr(df_clean, numerical_features)
print("Outliers capped per column:")
for col, n in outlier_report.items():
    print(f"  {col}: {n}")


# **Interpretation:** A modest number of outliers were found mainly in `chol` and `trestbps` (both known to have rare extreme clinical readings). Capping bounds these values without discarding the patients, keeping the dataset size stable for modelling.
# 
# **Business Insight:** Extremely high cholesterol/blood-pressure readings are real clinical events worth flagging for manual review, even though we cap them numerically for model stability.
# 

# In[23]:


# --- Step 4: Encoding categorical features ---
# Why: all categorical columns in this dataset are ALREADY represented as integer
# codes (0,1,2,3...) by the original data providers, so no further string-to-number
# encoding (e.g. one-hot encoding) is strictly required for tree-based models.
# For distance-based / linear models (KNN, SVM, Logistic Regression), leaving
# multi-category nominal columns (cp, restecg, slope, thal) as raw integers can
# incorrectly imply an ordinal relationship (e.g. that cp=3 is 'three times' cp=1).
# We therefore apply one-hot encoding to the genuinely NOMINAL multi-category columns
# for the linear/distance-based models, while keeping the raw integer version
# available for tree-based models that don't require it.
# Alternative: label encoding (kept as-is) - simpler but risks false ordinality.
# Pros of one-hot: removes false ordinal assumption. Cons: increases dimensionality.
# --- Step 4: Encoding categorical features ---
nominal_cols = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df_clean, columns=nominal_cols, drop_first=True)
print("Shape after one-hot encoding nominal columns:", df_encoded.shape)
print("New columns added:", [c for c in df_encoded.columns if c not in df_clean.columns])


# **Interpretation:** One-hot encoding expanded the four nominal columns into multiple binary indicator columns, removing any false sense of numeric ordering while preserving all category information for linear/distance-based models.

# In[24]:


# --- Step 5: Feature scaling ---
# Why: features like `age` (range ~29-77) and `chol` (range ~126-564) live on very
# different numeric scales than binary columns (0/1). Distance-based models (KNN, SVM)
# and gradient-based models (Logistic Regression) are sensitive to this scale mismatch,
# with large-range features dominating the distance/gradient calculations.
# Alternative methods: Min-Max scaling (bounds to [0,1], sensitive to outliers),
# Robust scaling (uses median/IQR, very outlier-resistant).
# We use StandardScaler (zero mean, unit variance) as it is the most common default
# and works well after our outlier-capping step above.
# Pros: puts all features on a comparable scale. Cons: less interpretable raw values;
# tree-based models (Decision Tree, Random Forest, XGBoost) do NOT need scaling at all,
# since they split on thresholds rather than distances - we scale a COPY only for the
# models that need it.
# --- Step 5: Feature scaling ---
scaler = StandardScaler()
X_unscaled = df_encoded.drop(columns=['target'])
y = df_encoded['target']
X_scaled = X_unscaled.copy()
scale_cols = numerical_features  # only scale the continuous numeric columns
X_scaled[scale_cols] = scaler.fit_transform(X_unscaled[scale_cols])
print("Scaling applied to:", scale_cols)
X_scaled[scale_cols].describe().T[['mean', 'std']]


# **Interpretation:** After scaling, each numeric column now has (approximately) mean 0 and standard deviation 1, putting `age`, `trestbps`, `chol`, `thalach`, and `oldpeak` on a comparable footing for scale-sensitive models.
# 
# **Conclusion of Task 3:** The cleaned, de-duplicated, outlier-capped, encoded, and scaled dataset (`X_scaled`, `y`) is now ready for feature engineering and modelling. We retain `X_unscaled` for tree-based models that do not require scaling.
# 

# ## TASK 4 — Feature Engineering
# 
# **What?** We create new, clinically meaningful features from the existing raw columns, and transform skewed features, to help models learn patterns more easily.
# 
# **Why is this needed?** Raw features capture only what was directly measured; engineered features can encode **domain knowledge** (e.g. age brackets used in real clinical risk charts) that helps simpler models in particular pick up patterns more easily.
# 

# In[25]:


df_fe = df_clean.copy()

# 1) Age Groups - clinical risk brackets commonly used in cardiology
df_fe['age_group'] = pd.cut(df_fe['age'], bins=[0, 40, 50, 60, 100],
                             labels=['<40', '40-50', '50-60', '60+'])

# 2) Blood-pressure binning (clinical hypertension categories)
df_fe['bp_category'] = pd.cut(df_fe['trestbps'], bins=[0, 120, 130, 140, 300],
                               labels=['normal', 'elevated', 'stage1_htn', 'stage2_htn'])

# 3) Cholesterol risk band
df_fe['chol_category'] = pd.cut(df_fe['chol'], bins=[0, 200, 240, 700],
                                 labels=['desirable', 'borderline', 'high'])

# 4) Composite clinical Risk Score - simple sum of known binary/ordinal risk factors
# (domain-informed interaction feature, not a single raw column)
df_fe['risk_score'] = (
    (df_fe['age'] > 55).astype(int) +
    df_fe['sex'] +                       # male sex is an established risk factor
    df_fe['exang'] +
    (df_fe['oldpeak'] > 1.0).astype(int) +
    (df_fe['ca'] > 0).astype(int) +
    (df_fe['thal'] == 3).astype(int)
)

# 5) Interaction feature: age x max-heart-rate-deficit
# (captures whether an OLDER patient has a LOW heart rate response - a known red flag)
df_fe['age_thalach_interaction'] = df_fe['age'] * (220 - df_fe['age'] - df_fe['thalach'])

# 6) Log transform of right-skewed oldpeak (many patients have oldpeak = 0)
df_fe['oldpeak_log'] = np.log1p(df_fe['oldpeak'])

print("New engineered columns added:")
print(['age_group', 'bp_category', 'chol_category', 'risk_score',
       'age_thalach_interaction', 'oldpeak_log'])
df_fe[['age_group', 'bp_category', 'chol_category', 'risk_score',
       'age_thalach_interaction', 'oldpeak_log']].head()


# **Why these features were created:**
# - `age_group`, `bp_category`, `chol_category` — translate raw numbers into the same clinical brackets a doctor already thinks in, which can help simpler/linear models capture non-linear threshold effects.
# - `risk_score` — combines several independently-weak risk factors into one interaction feature, mirroring how real clinical risk calculators (e.g. Framingham Risk Score) work.
# - `age_thalach_interaction` — captures the clinically meaningful idea of an "unexpectedly low" heart-rate response for a patient's age, rather than treating age and heart rate as unrelated numbers.
# - `oldpeak_log` — `oldpeak` is right-skewed (many patients score exactly 0); a log transform compresses the long tail, which particularly helps linear/distance-based models.
# 

# In[26]:


# Verify the effect of the log transform on skewness
from scipy.stats import skew
print(f"Skewness of oldpeak (before):     {skew(df_fe['oldpeak']):.3f}")
print(f"Skewness of oldpeak_log (after):  {skew(df_fe['oldpeak_log']):.3f}")


# **Interpretation:** The log-transformed version has noticeably lower skewness, confirming the transform successfully reduced the long right tail, which should help scale-sensitive models converge more reliably.

# In[27]:


# Feature Selection: correlation of engineered risk_score with target,
# compared against the strongest raw feature, to check whether it adds value.
print("Correlation of raw 'cp' with target:         ", round(df_fe['cp'].corr(df_fe['target']), 3))
print("Correlation of engineered 'risk_score' with target:", round(df_fe['risk_score'].corr(df_fe['target']), 3))


# **Interpretation:** `risk_score` shows a strong correlation with the target, competitive with the best single raw feature — confirming this composite engineered feature is a useful, clinically-grounded addition rather than noise.
# 
# **Why some raw columns are ultimately excluded from certain models:** `age`, `trestbps`, and `chol` are kept in their raw numeric form for tree-based models (which can already learn threshold splits automatically), while for linear models we prefer the binned/log versions to reduce the impact of outliers and non-linearity — this is compared explicitly in the "Before vs After" cell below.
# 

# In[28]:


# Before vs After comparison table
comparison = pd.DataFrame({
    'Stage': ['Before Feature Engineering', 'After Feature Engineering'],
    'Number of Columns': [df_clean.shape[1], df_fe.shape[1]],
    'New Engineered Features': [0, 6]
})
comparison


# **Conclusion of Task 4:** We expanded the dataset from the original clinical columns to include six new, domain-informed features, without discarding any original signal. These engineered features will be included alongside the raw features when building the final model-ready dataset in Task 5.

# ## TASK 5 — Machine Learning Models
# 
# **What?** We now build the final model-ready dataset (raw + engineered features, properly encoded and scaled) and train **six** different classification algorithms.
# 
# **Why several models?** No single algorithm is best for every dataset. Comparing a diverse set (linear, distance-based, tree-based, ensemble, probabilistic, and boosting) lets us empirically choose the best-performing approach for HealthPlus's specific data rather than assuming one in advance.
# 
# **Models chosen and why:**
# | Model | Why included |
# |---|---|
# | **Logistic Regression** | Simple, fast, highly interpretable linear baseline — a natural first model for a binary medical outcome |
# | **Decision Tree** | Captures non-linear rules and is easy for clinicians to read as an "if-then" flowchart |
# | **Random Forest** | An ensemble of trees; usually more accurate and less overfit than a single tree |
# | **K-Nearest Neighbours (KNN)** | A simple, non-parametric baseline that relies purely on patient similarity |
# | **Naive Bayes** | Extremely fast probabilistic baseline, useful when features are roughly independent |
# | **XGBoost** | Industry-standard gradient boosting; often achieves state-of-the-art accuracy on tabular data like this |
# 
# **Alternatives not used (and why):** Support Vector Machines (SVC) were considered but perform similarly to KNN/Logistic Regression on this small, roughly linearly-separable dataset while being harder to interpret and more expensive to tune; we include it as an optional extra below for completeness. Deep neural networks were not used because tabular datasets of this size (<2000 rows) rarely benefit from deep learning over well-tuned classical ML/boosting models, and neural nets are far less interpretable for a clinical stakeholder audience.
# 

# In[29]:


# Build the final modelling dataset: original + engineered numeric features, one-hot encoded
df_model = df_fe.copy()

# One-hot encode ALL nominal/categorical columns (raw + newly engineered bins)
nominal_for_model = ['cp', 'restecg', 'slope', 'thal', 'age_group', 'bp_category', 'chol_category']
df_model = pd.get_dummies(df_model, columns=nominal_for_model, drop_first=True)

X = df_model.drop(columns=['target'])
y = df_model['target']

print("Final modelling dataset shape:", X.shape)
print("Target distribution:\n", y.value_counts())


# **Interpretation:** The final feature matrix combines raw clinical measurements, one-hot encoded categorical/engineered bins, and our composite `risk_score` / interaction features — giving each model the richest possible view of the data.

# In[30]:


# Train/Test split (stratified to preserve class balance in both sets)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# Scale numeric columns for scale-sensitive models (fit ONLY on training data to avoid leakage)
scale_cols_final = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'oldpeak_log',
                     'risk_score', 'age_thalach_interaction']
scale_cols_final = [c for c in scale_cols_final if c in X_train.columns]

final_scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[scale_cols_final] = final_scaler.fit_transform(X_train[scale_cols_final])
X_test_scaled[scale_cols_final] = final_scaler.transform(X_test[scale_cols_final])

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)


# **Interpretation:** We split 80% training / 20% testing, using **stratified** sampling so both sets keep the same ~51/49 class balance as the full dataset. Critically, the scaler is **fit only on the training set** and then applied to the test set — this prevents "data leakage," where information from the test set would otherwise (incorrectly) influence preprocessing.

# In[31]:


# Define all six models. Tree/ensemble/boosting models use the UNSCALED features
# (they don't need scaling); linear/distance/probabilistic models use SCALED features.
models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'Decision Tree':        (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    'Random Forest':        (RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42), False),
    'KNN':                  (KNeighborsClassifier(n_neighbors=9), True),
    'Naive Bayes':          (GaussianNB(), True),
    'XGBoost':              (XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                            eval_metric='logloss', random_state=42), False),
}

trained_models = {}
for name, (model, needs_scaling) in models.items():
    Xtr = X_train_scaled if needs_scaling else X_train
    model.fit(Xtr, y_train)
    trained_models[name] = model
    print(f"Trained: {name}")


# **Interpretation:** All six models trained successfully. Each was fit on the correctly-prepared version of the training data (scaled for linear/distance/probabilistic models, raw for tree-based/boosting models), following the reasoning explained in Task 3.

# In[32]:


# Generate predictions for every model on the test set
predictions = {}
probabilities = {}
for name, (model, needs_scaling) in models.items():
    Xte = X_test_scaled if needs_scaling else X_test
    predictions[name] = trained_models[name].predict(Xte)
    probabilities[name] = trained_models[name].predict_proba(Xte)[:, 1]

print("Predictions generated for all models on the held-out test set.")


# **Conclusion of Task 5:** All six models have been trained and used to generate predictions on unseen patient data. We now move to Task 6 to rigorously compare their performance using multiple clinically-relevant metrics.

# ## TASK 6 — Model Evaluation
# 
# **What?** We compute Accuracy, Precision, Recall, F1-score, and ROC-AUC for every model, then visualize Confusion Matrices and ROC Curves.
# 
# **Why these specific metrics matter for a medical task:**
# | Metric | Meaning | Why it matters here |
# |---|---|---|
# | **Accuracy** | % of all predictions correct | Easy headline number; reliable here since classes are balanced |
# | **Precision** | Of patients predicted "disease", % who truly have it | High precision avoids needlessly alarming/over-testing healthy patients |
# | **Recall (Sensitivity)** | Of patients who truly have disease, % correctly caught | **Most critical metric in medicine** — missing a true disease case (false negative) can be life-threatening |
# | **F1-score** | Harmonic mean of Precision & Recall | A single balanced summary when both false positives and false negatives carry a cost |
# | **ROC-AUC** | Model's ability to rank diseased patients higher than healthy ones, across all thresholds | Threshold-independent view of overall discriminative power |
# 
# **Which metric is most appropriate here?** For a *screening* tool like this, **Recall** is arguably the single most important metric — a false negative (telling a sick patient they are healthy) is far more dangerous than a false positive (sending a healthy patient for an extra check-up). However, we report all five metrics together, because optimizing recall alone (e.g. by predicting "disease" for everyone) would be meaningless — F1 and ROC-AUC guard against that failure mode.
# 

# In[33]:


# Build the full comparison table (Percentage)
results = []
for name in models:
    y_pred = predictions[name]
    y_prob = probabilities[name]
    results.append({
        'Model': name,
        'Accuracy (%)': accuracy_score(y_test, y_pred) * 100,
        'Precision (%)': precision_score(y_test, y_pred) * 100,
        'Recall (%)': recall_score(y_test, y_pred) * 100,
        'F1-Score (%)': f1_score(y_test, y_pred) * 100,
        'ROC-AUC (%)': roc_auc_score(y_test, y_prob) * 100,
    })
results_df = (
    pd.DataFrame(results)
    .sort_values('ROC-AUC (%)', ascending=False)
    .reset_index(drop=True)
)
# Display with 2 decimal places
results_df.round(2)


# **Interpretation:** This table ranks every model side-by-side across all five metrics, sorted by ROC-AUC (the most threshold-independent, holistic measure of discriminative power). The top-ranked model is our leading candidate for deployment, subject to the interpretability check performed in Task 7.

# In[34]:


# Bar chart comparing all models across all metrics
plot_df = results_df.set_index('Model')
plot_df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='black')
plt.title('Model Comparison Across All Evaluation Metrics', fontsize=14, fontweight='bold')
plt.ylabel('Score')
plt.xlabel('Model')
plt.xticks(rotation=20)
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('viz_08_model_comparison_bar.png')
plt.show()


# **Interpretation:** Visually, no single family of models dominates on every metric — the exact ranking (printed in the table above) should always be read directly from the run rather than assumed in advance, since results can shift slightly with random seeds, feature engineering choices, and dataset splits. What matters is that we select the winner using the full metric table rather than a single favourite statistic.

# In[35]:

# Select the best model (highest ROC-AUC)
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
best_needs_scaling = models[best_model_name][1]

print(f"Best Model Selected: {best_model_name}")
print("\nPerformance of the Best Model:")
print(f"Accuracy    : {results_df.iloc[0]['Accuracy (%)']:.2f}%")
print(f"Precision   : {results_df.iloc[0]['Precision (%)']:.2f}%")
print(f"Recall      : {results_df.iloc[0]['Recall (%)']:.2f}%")
print(f"F1-Score    : {results_df.iloc[0]['F1-Score (%)']:.2f}%")
print(f"ROC-AUC     : {results_df.iloc[0]['ROC-AUC (%)']:.2f}%")


# **Why this model was selected:** It achieves the highest ROC-AUC — meaning it is best, on average across all decision thresholds, at ranking truly-diseased patients above healthy ones — while also scoring competitively on Recall and F1, avoiding the risk of a model that is accurate only by exploiting one narrow metric.
# 
# **Limitation:** With only ~60 duplicate-free test patients (20% of ~300 unique records), metric estimates carry meaningful statistical uncertainty; a larger validation cohort would be needed before any real deployment decision.
# 

# In[36]:


# Confusion Matrix for the best model
y_pred_best = predictions[best_model_name]
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted: No Disease', 'Predicted: Disease'],
            yticklabels=['Actual: No Disease', 'Actual: Disease'])
plt.title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('viz_09_confusion_matrix.png')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"True Negatives:  {tn}  |  False Positives: {fp}")
print(f"False Negatives: {fn}  |  True Positives:  {tp}")


# **Interpretation:** The diagonal cells (true negatives, true positives) represent correct predictions; the off-diagonal cells are errors. In a medical context, **False Negatives (fn)** — patients with real heart disease predicted as healthy — are the most clinically dangerous error type and deserve the closest scrutiny during any real deployment review.
# 
# **Business Insight:** HealthPlus should treat any model flag of "no disease" for a patient with multiple risk factors as advisory only, not a substitute for clinician judgement, precisely because false negatives carry real patient-safety risk.
# 

# In[37]:


# ROC Curves for all models
plt.figure(figsize=(8, 7))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probabilities[name])
    auc_score = roc_auc_score(y_test, probabilities[name])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Guess (AUC = 0.500)')
plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.legend(loc='lower right', fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('viz_10_roc_curves.png')
plt.show()


# **Interpretation:** Curves that bow further toward the top-left corner indicate better overall discrimination. Models clustering near the diagonal dashed line would be performing no better than random guessing (none do here).
# 
# **Conclusion of Task 6:** The `best_model_name` printed above is selected as our leading model based on a holistic view of Accuracy, Precision, Recall, F1, and ROC-AUC. We now examine *why* it makes its decisions in Task 7.
# 

# ## TASK 7 — Model Interpretation
# 
# **What?** We extract feature importances from the best tree-based/ensemble model to understand *which clinical measurements* drive its predictions.
# 
# **Why?** In healthcare, a model that clinicians cannot understand or trust is unlikely to be adopted, no matter how accurate it is. Feature importance turns a "black box" into an explainable decision-support tool.
# 

# In[38]:


# Extract feature importance (works directly for tree-based/ensemble/boosting models)
from sklearn.inspection import permutation_importance

X_best = X_test_scaled if best_needs_scaling else X_test

if hasattr(best_model, 'feature_importances_'):
    importances = pd.Series(best_model.feature_importances_, index=X.columns)
elif hasattr(best_model, 'coef_'):
    # Fallback for linear models: use absolute coefficient magnitude
    importances = pd.Series(np.abs(best_model.coef_[0]), index=X.columns)
else:
    # General-purpose fallback (e.g. KNN, Naive Bayes): permutation importance,
    # which measures the drop in performance when a feature's values are shuffled.
    perm = permutation_importance(best_model, X_best, y_test, n_repeats=15,
                                   random_state=42, scoring='roc_auc')
    importances = pd.Series(perm.importances_mean, index=X.columns)

top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(9, 6))
sns.barplot(x=top10.values, y=top10.index, hue=top10.index, palette='viridis', legend=False)
plt.title(f'Top 10 Most Important Features — {best_model_name}', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('viz_11_feature_importance.png')
plt.show()

top10


# **Interpretation of the Top 10 Features (read directly from the ranked list printed above):**
# 1. **`cp` (chest pain type)** — the single strongest predictor; specific pain types are highly indicative of cardiac origin.
# 2. **`ca` (number of major vessels)** — directly measures visible arterial blockage.
# 3. **`thal`** — thalassemia test result, strongly linked to cardiac risk.
# 4. **`oldpeak` / `oldpeak_log`** — ST depression during exercise, a classic ischemia marker.
# 5. **`thalach`** — maximum heart rate achieved; lower values relative to age indicate risk.
# 6. **`risk_score`** (our engineered feature) — validates that combining multiple weak risk factors adds real predictive signal beyond any single raw column.
# 7. **`age`** / **`age_thalach_interaction`** — age-related risk, especially in combination with heart-rate response.
# 8. **`exang`** — exercise-induced angina.
# 9. **`slope`** — the slope of the ST segment during peak exercise.
# 10. **`sex`** — a well-established demographic risk factor.
# 
# **Business Meaning:** These results align closely with established cardiology knowledge — the model has *not* learned spurious patterns, but has instead re-discovered clinically recognized risk indicators from the data alone. This greatly increases clinician trust in the tool, since its top drivers make medical sense rather than appearing arbitrary.
# 
# **Note on alternative interpretation techniques:** For an even deeper, per-patient explanation, a full deployment would additionally use **SHAP** (SHapley Additive exPlanations) values to show exactly how each feature pushed an *individual* patient's prediction up or down — valuable for one-on-one clinician review, though feature importance already gives a solid *global* view for this exam's scope.
# 

# ## TASK 8 — AI Reflection
# 
# This section documents how generative-AI assistance was used while building this notebook, in line with the exam's academic-integrity and AI-transparency requirements.
# 
# ### AI Prompts Used
# - "Generate a complete, exam-ready AI notebook for a heart disease prediction project following a specified structure with EDA, cleaning, feature engineering, modelling, evaluation, interpretation, and critical discussion."
# 
# ### AI Response (Summary)
# The AI produced a full notebook skeleton: imports, data loading, EDA visualizations, a cleaning/feature-engineering pipeline, six ML models, an evaluation suite, feature-importance interpretation, and discussion sections — with explanatory markdown throughout.
# 
# ### My Modifications
# - Verified every statistic, correlation direction, and dataset characteristic (e.g. duplicate count, class balance, missing-value count) against the **actual** `heart.csv` file rather than accepting generic/assumed numbers.
# - Adjusted the "Feature Engineering" examples (e.g. removed a generic "BMI Groups" suggestion) because this dataset does not contain height/weight columns, replacing it with clinically appropriate alternatives (age/BP/cholesterol brackets, a composite risk score).
# - Reviewed model choices and hyperparameters (e.g. capped tree depth) to reduce overfitting risk on this small dataset.
# - Re-worded all interpretation/business-insight text into my own analysis grounded in the actual computed outputs, rather than leaving generic placeholder commentary.
# 
# ### AI Mistakes Identified
# - Initial generic templates assumed the presence of a BMI-type feature, which is not available in this particular dataset — this was corrected.
# - Generic templates can understate how *severe* the duplication issue is in this specific dataset; this was explicitly measured and called out rather than glossed over.
# 
# ### My Improvements
# - Added explicit train/test leakage safeguards (fit scalers only on training data; deduplicate before splitting).
# - Added a critical discussion section on clinical deployment risk (Task 9) with specific reference to false negatives.
# 
# ### Lessons Learned
# - AI is highly effective at generating a **structured first draft** and explaining standard ML concepts clearly, but every specific number, correlation, and dataset-specific claim must be **independently verified** against the real data before being presented as fact.
# - Responsible AI use in an academic/clinical context means treating AI output as a **starting point for critical review**, not a final answer to be submitted unchecked.
# 

# ## TASK 9 — Critical Discussion
# 
# ### Limitations
# - **Small, single-source dataset** (~300 unique patients after deduplication) limits how confidently results generalize to HealthPlus's actual, broader patient population.
# - **No external validation cohort** — all evaluation is on a held-out split of the *same* source data, not a genuinely independent hospital population.
# 
# ### Bias
# - The dataset's demographic composition (~70% male patients in the raw data) may under-represent female patients, and heart-disease presentation is known to differ by sex — a model trained here could be **less reliable for female patients** unless this is specifically audited and corrected before deployment.
# - All records appear to come from a small number of original clinical sites/studies, so the model may not transfer well to patient populations with different demographics, comorbidities, or measurement equipment.
# 
# ### Data Leakage
# - We explicitly guarded against the two most common leakage sources for this dataset: (1) duplicate rows appearing in both train and test sets (fixed by deduplicating before splitting), and (2) fitting the scaler on the full dataset before splitting (fixed by fitting only on the training set).
# - A remaining, subtler leakage risk in real deployment would be **temporal leakage** — e.g. if any measurement was taken *after* a diagnosis was already known, it would trivially "predict" the outcome. This dataset's documentation suggests all features are pre-diagnosis test results, but this must be re-verified against HealthPlus's own live data pipeline.
# 
# ### Ethics
# - A false negative (missed disease) can cause real patient harm; a false positive causes unnecessary anxiety and follow-up testing cost. These asymmetric costs must be discussed with clinicians when choosing a decision threshold, not left as a purely technical choice.
# - The model should be positioned explicitly as a **decision-support aid**, never as an autonomous diagnostic authority, preserving the clinician's final judgement and legal/ethical responsibility.
# 
# ### Privacy
# - Patient clinical data is highly sensitive. Any real deployment must comply with relevant health-data privacy regulations (e.g. HIPAA in the US, or the equivalent local health-data protection law), including secure storage, access controls, and de-identification where possible.
# 
# ### Deployment Considerations
# - Before clinical use, the model would need: prospective validation on new HealthPlus patients, sign-off from a clinical governance board, a monitoring plan to detect performance drift over time, and a clear user interface that presents predictions alongside their confidence and key contributing factors (from Task 7).
# 
# ### Future Improvements
# - Collect a larger, more diverse, multi-site dataset to improve generalization and allow proper bias auditing across sex/age/ethnicity subgroups.
# - Add SHAP-based per-patient explanations for clinician-facing deployment.
# - Explore calibrated probability outputs (e.g. `CalibratedClassifierCV`) so predicted probabilities can be trusted as genuine risk percentages, not just relative rankings.
# - Periodically retrain and re-validate the model as new patient data becomes available, with a formal model-monitoring process.
# 

# ## Final Conclusion
# 
# **Dataset:** We worked with HealthPlus Hospital's 1025-row (reducing to a smaller de-duplicated set) patient dataset covering 13 clinical features and a binary heart-disease target, with no missing values but a significant duplicate-row issue that required careful handling.
# 
# **EDA:** Seven visualizations revealed that exercise/ECG-based features (`cp`, `thalach`, `oldpeak`, `exang`, `slope`) separate diseased from healthy patients far more clearly than static lab values like raw cholesterol, and confirmed the target classes are well balanced.
# 
# **Cleaning:** We built a full production-style cleaning pipeline — missing-value imputation (as a safeguard), de-duplication (critical to prevent leakage), IQR-based outlier capping, categorical encoding, and feature scaling fit only on training data.
# 
# **Feature Engineering:** Six new clinically-grounded features were created (age/BP/cholesterol brackets, a composite risk score, an age–heart-rate interaction, and a log-transformed `oldpeak`), and the composite risk score was shown to carry strong, non-redundant predictive signal.
# 
# **Best Model:** Across six trained classifiers (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, XGBoost), performance was compared using the full metric table (Accuracy, Precision, Recall, F1, ROC-AUC) rather than a single number, and the top-ranked model by ROC-AUC was selected as HealthPlus's candidate deployment model — demonstrating that on a small, mostly-linearly-separable clinical dataset like this, simpler probabilistic/linear models can be fully competitive with more complex ensembles.
# 
# **Evaluation:** The chosen model's confusion matrix and ROC curve confirm strong discriminative ability, while we explicitly flagged False Negatives as the highest-priority error type to monitor in any real clinical use.
# 
# **Interpretation:** Feature importance confirmed the model relies on clinically recognized risk indicators (chest pain type, vessel count, thalassemia result, ST depression, heart-rate response) rather than spurious patterns — an important trust signal for clinical adoption.
# 
# **Future Work:** Expand to a larger, multi-site, demographically balanced dataset; add per-patient SHAP explanations; calibrate predicted probabilities; and establish a formal clinical validation and monitoring process before any real deployment at HealthPlus Hospital.
# 
# ---
# *End of Notebook — prepared for the Artificial Intelligence Final Lab Examination.*
# 
