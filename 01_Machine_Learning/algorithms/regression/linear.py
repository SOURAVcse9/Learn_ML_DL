# 📌 Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 📌 Step 2: Load dataset
df = pd.read_csv("regression model/boston.csv")
print(df.head()) 
df = df.drop(['Unnamed: 0'], axis=1)

# 📌 Step 3: Feature & Target
X = df[['rm']]
y = df['medv']

# 📌 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 Step 5: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate y_pred_test for evaluation
y_pred_test = model.predict(X_test)

# 📌 Step 12: Evaluation
print("R2 Score:", r2_score(y_test, y_pred_test))
# 📌 Step 6: Get coefficients
b0 = model.intercept_
b1 = model.coef_[0]

# 📌 Step 7: Print Equation
print("Regression Equation:")
print(f"medv = {b0:.2f} + ({b1:.2f} * rm)")

# 📌 Step 8: Take user input
rm_value = float(input("Enter number of rooms (rm): "))

# 📌 Step 9: Predict
predicted_price = model.predict([[rm_value]])

print(f"Predicted House Price (medv): {predicted_price[0]:.2f}")