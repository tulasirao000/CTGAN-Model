import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from ctgan import CTGAN

data = pd.read_csv("cancer patient data sets.csv")
print(data.head())
print(data.info())

X = data.drop("Level", axis=1)
y = data["Level"]

encoder = LabelEncoder()

for col in X.select_dtypes(include='object').columns:
    X[col] = encoder.fit_transform(X[col])

train_data = data.copy()


train_data = train_data.dropna()


categorical_columns = train_data.select_dtypes(include='object').columns.tolist()

print("Categorical columns:", categorical_columns)
print("Rows:", len(train_data))

ctgan = CTGAN(
    epochs=50,
    batch_size=min(50, 100),
    verbose=True
)

ctgan.fit(train_data, categorical_columns)

synthetic_data = ctgan.sample(100)
print(synthetic_data.head())

augmented_data = pd.concat([train_data, synthetic_data], ignore_index=True)

X_aug = augmented_data.drop(columns=["Patient Id", "index"])
y_aug = augmented_data["Level"]

le_y = LabelEncoder()
y_aug = le_y.fit_transform(y_aug)

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(0)

assert X.ndim == 2
assert y.ndim == 1
assert X.shape[0] == y.shape[0]
assert X.isnull().sum().sum() == 0
assert not any(X.dtypes == "object")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
