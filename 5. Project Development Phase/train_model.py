from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv("dataset/Crop_recommendation.csv")

print(data.head())
print(data.shape)
print(data.info())
print(data.describe())
print(data['label'].unique())
print(data['label'].nunique())
print(data['label'].value_counts())

plt.figure(figsize=(12,6))
sns.countplot(x='label', data=data)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()

data.hist(figsize=(15,10))
plt.show()

plt.figure(figsize=(10,8))

sns.heatmap(
    data.select_dtypes(include=['number']).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()

X = data.drop('label', axis=1)
y = data['label']

print("Input Features:")
print(X.head())

print("\nTarget Variable:")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================
# Model Training
# ==========================

model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# ==========================
# Save the Trained Model
# ==========================

with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully as model.pkl")