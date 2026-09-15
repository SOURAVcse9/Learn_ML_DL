#corrected multiple regression
# 📌 Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 📌 Step 2: Load dataset
df = pd.read_csv("regression model/Boston.csv")

# 📌 Step 3: Data cleaning
df = df.drop(['Unnamed: 0'], axis=1)

# 📌 Step 4: Feature & Target
X = df.drop('medv', axis=1)
y = df['medv']

# 📌 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 Step 6: Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 📌 Step 7: Get coefficients
b0 = model.intercept_
coeff = model.coef_

# 📌 Step 8: Print Equation
print("\nMultiple Linear Regression Equation:\n")

features = X.columns
equation = f"medv = {b0:.2f}"

for i in range(len(coeff)):
    equation += f" + ({coeff[i]:.2f} * {features[i]})"

print(equation)

# Calculate y_pred for the test set for R2 score
y_pred_test = model.predict(X_test)

# 📌 Step 12: Evaluation
print("R2 Score:", r2_score(y_test, y_pred_test))

# 📌 Step 9: User Input
print("\nEnter values for prediction:")

user_data = []
for feature in features:
    value = float(input(f"{feature}: "))
    user_data.append(value)

# 📌 Step 10: Convert to DataFrame (FIX for warning)
user_df = pd.DataFrame([user_data], columns=features)

# 📌 Step 11: Prediction for user input
predicted = model.predict(user_df)


# 📌 Step 13: Output
print(f"\nPredicted House Price (medv): {predicted[0]:.2f}")