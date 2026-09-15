# Feature Selection Methods - Quick Reference Guide

---

## 🎯 At a Glance

| Method | Type | Speed | Accuracy | Best For |
|--------|------|-------|----------|----------|
| **Variance Threshold** | Filter | ⚡⚡⚡ | ⭐⭐ | Removing constant features |
| **Mutual Information** | Filter | ⚡⚡⚡ | ⭐⭐⭐ | Dependency detection |
| **ANOVA F-Test** | Filter | ⚡⚡⚡ | ⭐⭐⭐ | Statistical ranking |
| **Correlation** | Filter | ⚡⚡⚡ | ⭐⭐ | Quick exploration |
| **Forward Selection** | Wrapper | ⚡⚡ | ⭐⭐⭐⭐ | Incremental building |
| **Backward Elimination** | Wrapper | ⚡ | ⭐⭐⭐⭐ | Iterative removal |
| **RFE** | Wrapper | ⚡⚡ | ⭐⭐⭐⭐⭐ | All-purpose wrapper |
| **LASSO** | Embedded | ⚡⚡ | ⭐⭐⭐⭐ | Sparse solutions |

---

## 📊 FILTER METHODS

### 1. **Variance Threshold**

**What it does**: Removes features with variance below a threshold (constant/near-constant features)

**When to use**: 
- Preprocessing step
- Remove truly constant features
- High-dimensional data cleaning

**Code**:
```python
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.01)
X_selected = selector.fit_transform(X)
selected_features = X.columns[selector.get_support()]
```

**Pros** ✅
- Very fast
- Simple and interpretable
- Good preprocessing step

**Cons** ❌
- Ignores target variable
- Removes informative features with low variance
- Not suitable for main feature selection

**Example Output**:
```
Features removed: ['constant_feature_1', 'constant_feature_2']
Remaining features: 9/11
```

---

### 2. **Mutual Information**

**What it does**: Measures how much information one feature provides about the target

**When to use**:
- Detect non-linear relationships
- Interaction detection
- Quick feature importance ranking

**Code**:
```python
from sklearn.feature_selection import mutual_info_regression

mi_scores = mutual_info_regression(X, y, random_state=42)
mi_ranking = pd.DataFrame({
    'Feature': X.columns,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)
```

**Pros** ✅
- Detects non-linear relationships
- Fast computation
- No assumptions about data distribution
- Good for initial ranking

**Cons** ❌
- Ignores feature interactions
- Not robust with small samples
- Difficult to interpret for non-statisticians

**Example Output**:
```
Feature              MI_Score
alcohol              0.1234
sulphates            0.0856
volatile_acidity     0.0743
density              0.0621
```

---

### 3. **ANOVA F-Test**

**What it does**: Tests if feature means differ significantly across target groups

**When to use**:
- Classification problems (mainly)
- Statistical significance ranking
- Interpretable selection

**Code**:
```python
from sklearn.feature_selection import f_regression

f_scores, p_values = f_regression(X, y)
f_ranking = pd.DataFrame({
    'Feature': X.columns,
    'F_Score': f_scores,
    'P_Value': p_values
}).sort_values('F_Score', ascending=False)

# Select features with p-value < 0.05
significant_features = f_ranking[f_ranking['P_Value'] < 0.05]['Feature']
```

**Pros** ✅
- Statistically grounded
- Provides p-values for significance
- Good for linear relationships
- Interpretable results

**Cons** ❌
- Assumes linear relationships
- Sensitive to outliers
- Multiple testing issues
- Less suitable for regression

**Example Output**:
```
Feature              F_Score    P_Value
alcohol              876.543    0.0001
sulphates            654.321    0.0001
volatile_acidity     432.109    0.0001
```

---

### 4. **Correlation-Based Selection**

**What it does**: Ranks features by their correlation with the target

**When to use**:
- Quick exploration
- Baseline comparison
- Linear relationship detection

**Code**:
```python
# Calculate correlations with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)

# Select top N features
top_features = correlations.head(5).index.tolist()

# Alternative: Use SelectKBest with f_regression
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)
```

**Pros** ✅
- Very fast and simple
- Highly interpretable
- Good baseline method
- Visualizes easily

**Cons** ❌
- Only detects linear relationships
- Ignores feature interactions
- Affected by multicollinearity
- May select redundant features

**Example Output**:
```
Feature              Correlation
alcohol              0.4356
sulphates            0.3642
volatile_acidity     0.2648
citric_acid          0.1885
```

---

## 🎯 WRAPPER METHODS

### 5. **Forward Selection**

**What it does**: Starts with 0 features, iteratively adds the one that most improves performance

**When to use**:
- Building features incrementally
- When you know final feature count
- Smaller feature spaces

**Code**:
```python
from sklearn.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(
    LinearRegression(),
    n_features_to_select=5,
    direction='forward',
    n_jobs=-1
)

selector.fit(X_train_scaled, y_train)
selected_features = X.columns[selector.get_support()]
```

**Pros** ✅
- Uses actual model performance
- Considers feature interactions
- Greedy approach finds good solutions
- Interpretable process

**Cons** ❌
- Computationally expensive
- Greedy = may miss optimal subset
- Risk of overfitting
- No backtracking (once added, can't remove)

**Example Process**:
```
Step 1: Add 'alcohol' (RMSE: 0.75)
Step 2: Add 'sulphates' (RMSE: 0.74)
Step 3: Add 'volatile_acidity' (RMSE: 0.73)
Step 4: Add 'density' (RMSE: 0.732)
Step 5: Add 'free_sulfur_dioxide' (RMSE: 0.731)
```

---

### 6. **Backward Elimination**

**What it does**: Starts with all features, iteratively removes the one that least impacts performance

**When to use**:
- Removing unimportant features
- When removing is easier than adding
- Larger feature spaces

**Code**:
```python
from sklearn.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(
    LinearRegression(),
    n_features_to_select=5,
    direction='backward',
    n_jobs=-1
)

selector.fit(X_train_scaled, y_train)
selected_features = X.columns[selector.get_support()]
```

**Pros** ✅
- Considers full feature set
- May find different subsets than forward
- Actual model performance based
- Good for removing redundancy

**Cons** ❌
- More computationally expensive than forward
- Still greedy approach
- Can't recover removed features
- Risk of overfitting

**Example Process**:
```
Start: 11 features (RMSE: 0.754)
Remove 'free_sulfur_dioxide' (RMSE: 0.743)
Remove 'pH' (RMSE: 0.742)
Remove 'citric_acid' (RMSE: 0.741)
Remove 'chlorides' (RMSE: 0.740)
Remove 'residual_sugar' (RMSE: 0.740)
Remove 'total_sulfur_dioxide' (RMSE: 0.740)
→ Final: 5 features
```

---

### 7. **RFE (Recursive Feature Elimination)**

**What it does**: Recursively removes least important features based on model weights/importance

**When to use**:
- Best overall wrapper method
- When you have feature importance scores
- Good balance of speed and accuracy

**Code**:
```python
from sklearn.feature_selection import RFE

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=5,
    step=1
)

rfe.fit(X_train_scaled, y_train)
selected_features = X.columns[rfe.get_support()]

# Get rankings
rankings = pd.DataFrame({
    'Feature': X.columns,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')
```

**Pros** ✅
- Uses feature importance scores
- More efficient than sequential methods
- Considers feature interactions
- Stable and reproducible

**Cons** ❌
- Still computationally expensive
- Model-dependent results
- Assumes feature importance is meaningful
- Not suitable for high-dimensional data

**Example Output**:
```
Feature              Ranking
alcohol              1  ✓ Selected
sulphates            1  ✓ Selected
volatile_acidity     1  ✓ Selected
citric_acid          1  ✓ Selected
density              1  ✓ Selected
free_sulfur_dioxide  6
total_sulfur_dioxide 7
residual_sugar       8
chlorides            9
pH                   10
fixed_acidity        11
```

---

## 🔧 EMBEDDED METHODS

### 8. **LASSO Regression**

**What it does**: Uses L1 regularization to shrink unimportant coefficients to zero

**When to use**:
- Regularized regression problems
- Interpretable feature selection
- Production systems
- When you have many correlated features

**Code**:
```python
from sklearn.linear_model import Lasso, LassoCV

# Find optimal alpha
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)

# Train with optimal alpha
lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000)
lasso.fit(X_train_scaled, y_train)

# Get selected features (non-zero coefficients)
selected_features = X.columns[lasso.coef_ != 0].tolist()

# Get rankings by absolute coefficient value
coef_ranking = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': np.abs(lasso.coef_)
}).sort_values('Coefficient', ascending=False)
```

**Pros** ✅
- Automatic feature selection
- Prevents overfitting (regularization)
- Interpretable coefficients
- Efficient and scalable
- Handles multicollinearity
- Great for high-dimensional data

**Cons** ❌
- Assumes linear relationships
- Biased toward correlated features
- Alpha tuning required
- Less interpretable than filters
- May select arbitrary feature from groups

**Example Output**:
```
Optimal Alpha: 0.001234

Feature                Coefficient
alcohol                0.1842
sulphates              0.1156
volatile_acidity      -0.0943
citric_acid            0.0732
density               -0.0521
free_sulfur_dioxide    0.0000  ← Zeroed out
total_sulfur_dioxide   0.0000  ← Zeroed out
residual_sugar         0.0000  ← Zeroed out
chlorides              0.0000  ← Zeroed out
pH                     0.0000  ← Zeroed out
fixed_acidity          0.0000  ← Zeroed out
```

---

## 🎓 Decision Tree: Which Method to Use?

```
START: Feature Selection Needed?
│
├─ Need SPEED? (< 1 second)
│  └─ YES → Use FILTER METHODS
│     ├─ Need interpretability? → Correlation
│     ├─ Need statistical rigor? → ANOVA F-Test
│     └─ Detect non-linearity? → Mutual Information
│
├─ Need BEST ACCURACY?
│  └─ YES → Use WRAPPER METHODS (RFE preferred)
│
├─ Have MANY FEATURES? (> 1000)
│  └─ YES → Use EMBEDDED METHODS (LASSO)
│
├─ Need AUTOMATED SELECTION?
│  └─ YES → Use EMBEDDED METHODS (LASSO)
│
└─ Production System?
   └─ YES → Use EMBEDDED METHODS (LASSO)
      or Combination of multiple methods
```

---

## 🔄 Multi-Method Approach (RECOMMENDED)

**Best Practice**: Use **ensemble approach**
1. Apply all three method types
2. Identify features appearing in multiple methods
3. These are likely the most important
4. Use consensus features for final model

**Code Example**:
```python
# Get features from each method
filter_features = set(['alcohol', 'sulphates', 'volatile_acidity', 'citric_acid', 'density'])
wrapper_features = set(['alcohol', 'sulphates', 'volatile_acidity', 'free_sulfur_dioxide'])
embedded_features = set(['alcohol', 'sulphates', 'volatile_acidity', 'citric_acid'])

# Find consensus (features in all three methods)
consensus_features = filter_features & wrapper_features & embedded_features
# Result: {'alcohol', 'sulphates', 'volatile_acidity'}

# Alternative: Features in at least 2 methods
consensus_2 = set(f for f in filter_features | wrapper_features | embedded_features
                   if sum([f in s for s in [filter_features, wrapper_features, embedded_features]]) >= 2)
```

---

## 📋 Evaluation Metrics Comparison

### For Regression:
- **RMSE** (Root Mean Squared Error): Lower is better
  - Penalizes large errors heavily
  - Same unit as target
  
- **MAE** (Mean Absolute Error): Lower is better
  - Average absolute error
  - More interpretable
  
- **R²** (R-squared): Higher is better
  - Proportion of variance explained
  - 0-1 scale (0 = no better than mean, 1 = perfect)

### For Classification:
- **Accuracy**: % correct predictions
- **Precision**: % predicted positive that are correct
- **Recall**: % actual positive that are found
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

---

## 📊 Computational Complexity

| Method | Time | Space | Scalability |
|--------|------|-------|-------------|
| Variance Threshold | O(n) | O(1) | ⭐⭐⭐⭐⭐ Excellent |
| Correlation | O(n×m) | O(m²) | ⭐⭐⭐⭐ Very Good |
| Mutual Information | O(n×m) | O(m) | ⭐⭐⭐⭐ Very Good |
| ANOVA | O(n×m) | O(m) | ⭐⭐⭐⭐ Very Good |
| Forward Selection | O(m³) | O(n×m) | ⭐⭐ Poor |
| Backward Elimination | O(m³) | O(n×m) | ⭐⭐ Poor |
| RFE | O(m²×n) | O(n×m) | ⭐⭐⭐ Good |
| LASSO | O(n×m) | O(n×m) | ⭐⭐⭐⭐⭐ Excellent |

*where n = samples, m = features*

---

## 🎯 Common Scenarios & Recommendations

### Scenario 1: Quick Exploration
**Time**: < 5 minutes
**Methods**: 
1. Variance Threshold → remove constants
2. Correlation → quick ranking
3. Mutual Information → non-linear check
**Result**: Top 5-10 features for initial analysis

### Scenario 2: Kaggle Competition
**Time**: Unlimited, accuracy critical
**Methods**:
1. Filter (identify obvious features)
2. RFE (wrapper - best balance)
3. LASSO (embedded perspective)
4. Ensemble (consensus features)
**Result**: High-accuracy feature subset

### Scenario 3: Production Model
**Time**: Training speed important
**Methods**:
1. LASSO (efficient, regularized)
2. Variance Threshold (preprocessing)
3. Brief validation with RFE
**Result**: Fast, interpretable features

### Scenario 4: High-Dimensional Data (> 10,000 features)
**Time**: Must be fast
**Methods**:
1. Variance Threshold (remove constants)
2. LASSO (scalable embedded)
3. Correlation (for specific features)
**Result**: Reduced feature set for downstream analysis

### Scenario 5: Small Dataset (n < 100 samples)
**Time**: Careful to avoid overfitting
**Methods**:
1. LASSO (built-in regularization)
2. Avoid Wrapper (risk of overfitting)
3. Use Cross-Validation (critical!)
**Result**: Robust feature selection

---

## ✅ Quality Checklist

Before finalizing feature selection:

- [ ] Used train-test split correctly?
- [ ] Standardized features before method?
- [ ] Applied same preprocessing to test data?
- [ ] Avoided data leakage?
- [ ] Compared multiple methods?
- [ ] Used cross-validation?
- [ ] Checked for feature correlation?
- [ ] Visualized results?
- [ ] Documented decisions?
- [ ] Tested on holdout set?

---

## 🚨 Common Mistakes to Avoid

### ❌ Mistake 1: Fitting Scaler on All Data
```python
# WRONG
scaler.fit(X)  # Leaks test data!
X_scaled = scaler.transform(X)

# CORRECT
scaler.fit(X_train)  # Only training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### ❌ Mistake 2: Selecting Features on Full Dataset
```python
# WRONG
feature_selector.fit(X, y)  # Leaks test info!

# CORRECT
feature_selector.fit(X_train, y_train)
```

### ❌ Mistake 3: Comparing Methods Unfairly
```python
# WRONG
method_a_results = evaluate(features_a)
method_b_results = evaluate(features_b)  # Different split!

# CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
for method in methods:
    features = method.fit_transform(X_train)
    results = evaluate(X_test[features], y_test)
```

### ❌ Mistake 4: Ignoring Class Imbalance
```python
# WRONG
train_test_split(X, y, test_size=0.2)  # May split imbalanced!

# CORRECT
train_test_split(X, y, test_size=0.2, stratify=y)
```

### ❌ Mistake 5: Not Using Cross-Validation
```python
# WRONG
Single train-test split (high variance)

# CORRECT
cross_val_score(model, X, y, cv=5)
```

---

## 📚 Mathematical Intuition

### Variance Threshold
```
Variance = E[(X - μ)²]
Remove if Variance < threshold
```

### Mutual Information
```
I(X; Y) = ∑∑ P(x,y) * log(P(x,y) / (P(x)*P(y)))
Higher I(X;Y) = stronger relationship
```

### ANOVA F-Test
```
F = (Between-group variance) / (Within-group variance)
Higher F = features differ significantly
```

### LASSO Regularization
```
Loss = ||Y - Xβ||² + λ|β|
As λ ↑, more coefficients → 0 (feature selection)
```

---

## 🎬 Quick Start Commands

### Filter Methods
```python
# All-in-one filter evaluation
mi_scores = mutual_info_regression(X, y, random_state=42)
f_scores, p_values = f_regression(X, y)
correlations = X.corrwith(y).abs()
```

### Wrapper Methods
```python
# RFE (recommended wrapper)
from sklearn.feature_selection import RFE
rfe = RFE(LinearRegression(), n_features_to_select=5)
rfe.fit(X_train, y_train)
```

### Embedded Methods
```python
# LASSO with CV
from sklearn.linear_model import LassoCV
lasso_cv = LassoCV(cv=5)
lasso_cv.fit(X_train, y_train)
```

---

## 📞 Quick Debugging

**Problem**: Too many features selected
**Solution**: 
- Lower variance threshold
- Select fewer features in SelectKBest
- Increase LASSO regularization (alpha)

**Problem**: Too few features selected  
**Solution**:
- Raise variance threshold
- Select more features
- Decrease LASSO regularization

**Problem**: Contradictory results between methods
**Solution**:
- This is normal! Methods have different assumptions
- Use ensemble voting
- Trust RFE over simple filters

**Problem**: Model overfitting
**Solution**:
- Use LASSO (has regularization)
- Select fewer features
- Use cross-validation

---

## 🎓 Comparison Table Summary

| Aspect | Filter | Wrapper | Embedded |
|--------|--------|---------|----------|
| Speed | ⚡⚡⚡ Fast | ⚡ Slow | ⚡⚡ Medium |
| Accuracy | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Bias | High | Low | Medium |
| Interpretability | High | Medium | Medium |
| Scalability | High | Low | High |
| Interaction Detection | No | Yes | Partial |
| Best For | Quick Start | Best Results | Production |

---

**Remember**: No single method is best for all problems. Use multiple approaches and compare!

*Last Updated: June 2024*
