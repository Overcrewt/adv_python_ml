import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder


# ==============================================================
# 1. LOAD DATA
# ==============================================================

df = pd.read_csv("ai4i2020.csv")

print(f"Dataset: {df.shape[0]} records, {df.shape[1]} columns")
print("\nColumns:")
print(df.columns)


# ==============================================================
# 2. CREATE MULTI-CLASS TARGET
# ==============================================================

def get_failure_type(row):
    if row["TWF"] == 1:
        return "TWF"
    elif row["HDF"] == 1:
        return "HDF"
    elif row["PWF"] == 1:
        return "PWF"
    elif row["OSF"] == 1:
        return "OSF"
    elif row["RNF"] == 1:
        return "RNF"
    else:
        return "No Failure"


df["fault_type"] = df.apply(get_failure_type, axis=1)

print("\nClass distribution:")
print(df["fault_type"].value_counts())


# ==============================================================
# 3. PREPARE FEATURES
# ==============================================================

drop_cols = [
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF",
    "fault_type",
]

X = df.drop(columns=drop_cols)
y = df["fault_type"]

# Encode Type column: L, M, H -> numbers
le = LabelEncoder()
X["Type"] = le.fit_transform(X["Type"])


# ==============================================================
# 4. FEATURE ENGINEERING
# ==============================================================

X["temperature_difference"] = (
    X["Process temperature [K]"] - X["Air temperature [K]"]
)

X["power"] = (
    X["Rotational speed [rpm]"] * X["Torque [Nm]"] * 2 * np.pi / 60
)

X["wear_torque"] = X["Tool wear [min]"] * X["Torque [Nm]"]

X["temperature_ratio"] = (
    X["Process temperature [K]"] / X["Air temperature [K]"]
)

print(f"\nFeatures after feature engineering: {X.shape[1]}")
print(X.head())


# ==============================================================
# 5. TRAIN/TEST SPLIT
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTraining set: {X_train.shape[0]} records")
print(f"Test set:     {X_test.shape[0]} records")


# ==============================================================
# 6. MODEL PIPELINE
# ==============================================================

model = Pipeline([
    ("oversampler", RandomOverSampler(random_state=42)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    ))
])


# ==============================================================
# 7. CROSS-VALIDATION
# ==============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
)

print("\nCross-validation macro F1:")
print(f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ==============================================================
# 8. FINAL TEST EVALUATION
# ==============================================================

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\nRandom Forest with feature engineering and oversampling")
print(f"Test macro F1-score: {macro_f1:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ==============================================================
# 9. CONFUSION MATRIX
# ==============================================================

labels = ["No Failure", "HDF", "OSF", "PWF", "TWF", "RNF"]

cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")