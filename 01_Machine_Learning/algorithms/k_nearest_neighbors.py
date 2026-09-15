import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# CSV load (file path ঠিক করে নিবি)
df = pd.read_csv("KNN/KNNAlgorithmDataset.csv")

# 👉 ধরে নিচ্ছি last column হলো label (target)
X = df.iloc[:, :-1]   # features
y = df.iloc[:, -1]    # label

# Train-Test split0=
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model তৈরি
model = KNeighborsClassifier(n_neighbors=3)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Predictions:", y_pred)
print("Accuracy:", accuracy)
# Example: নতুন data predict
sample = [[5, 5]]   # নিজের dataset অনুযায়ী change করবি

prediction = model.predict(sample)
print("Predicted class:", prediction)