# 📌 Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 📌 Step 2: Load dataset
df = pd.read_csv("regression model/Boston.csv")
df = df.drop(['Unnamed: 0'], axis=1)

# 📌 Step 3: Create target class (classification)
df['price_cat'] = (df['medv'] > 25).astype(int)

# 📌 Step 4: Feature & Target
X = df.drop(['medv', 'price_cat'], axis=1)
y = df['price_cat']

# 📌 Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 📌 Step 6: Train Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 📌 Step 7: Prediction
y_pred = model.predict(X_test)

# 📌 Step 8: Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# 📌 Step 9: Coefficients
b0 = model.intercept_[0]
coeff = model.coef_[0]

# 📌 Step 10: Print Equation
print("\nLogistic Regression Equation:\n")

features = X.columns
equation = f"log(p/(1-p)) = {b0:.2f}"

for i in range(len(coeff)):
    equation += f" + ({coeff[i]:.2f} * {features[i]})"

print(equation)

# 📌 Step 11: User Input
print("\nEnter values for prediction:")

user_data = []
for feature in features:
    value = float(input(f"{feature}: "))
    user_data.append(value)

# 📌 Step 12: Convert to DataFrame (avoid warning)
user_df = pd.DataFrame([user_data], columns=features)

# 📌 Step 13: Prediction
predicted_class = model.predict(user_df)
probability = model.predict_proba(user_df)

# 📌 Step 14: Output
print(f"\nPredicted Class: {predicted_class[0]} (1=Expensive, 0=Cheap)")
print(f"Probability [Cheap, Expensive]: {probability[0]}")