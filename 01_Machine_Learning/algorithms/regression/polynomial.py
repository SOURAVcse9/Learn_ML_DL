#polinomial
# 📌 Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 📌 Step 2: Load dataset
df = pd.read_csv("regression model/Boston.csv")
df = df.drop(['Unnamed: 0'], axis=1)

# 📌 Step 3: Feature & Target
X = df[['lstat']]   # single feature (better for curve)
y = df['medv']

# 📌 Step 4: Polynomial transform (degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# 📌 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y, test_size=0.2, random_state=42
)

# 📌 Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Step 7: Prediction
y_pred = model.predict(X_test)

# 📌 Step 8: Evaluation
print("R2 Score:", r2_score(y_test, y_pred))

# 📌 Step 9: Get coefficients
b0 = model.intercept_
b1 = model.coef_[1]
b2 = model.coef_[2]

# 📌 Step 10: Print Equation
print("\nPolynomial Equation:")
print(f"medv = {b0:.2f} + ({b1:.2f} * lstat) + ({b2:.2f} * lstat^2)")

# 📌 Step 11: User Input
lstat_value = float(input("\nEnter lstat value: "))

# transform input
input_poly = poly.transform([[lstat_value]])

# 📌 Step 12: Prediction
predicted = model.predict(input_poly)

print(f"Predicted House Price (medv): {predicted[0]:.2f}")

# 📌 Step 13: Visualization (Curve)
plt.scatter(X, y)   # actual data

# sort for smooth curve
X_range = np.linspace(X.min(), X.max(), 100)
X_range_poly = poly.transform(X_range)

plt.plot(X_range, model.predict(X_range_poly))
plt.xlabel("lstat")
plt.ylabel("medv")
plt.title("Polynomial Regression (Curve)")
plt.show()