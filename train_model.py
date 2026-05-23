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


RANDOM_STATE = 42

LABELS = ["No Failure", "HDF", "OSF", "PWF", "TWF", "RNF"]


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

X_original = df.drop(columns=drop_cols)
y = df["fault_type"]

# Encode Type column: L, M, H -> numbers
le = LabelEncoder()
X_original["Type"] = le.fit_transform(X_original["Type"])


# ==============================================================
# 4. FEATURE ENGINEERING
# ==============================================================

X_engineered = X_original.copy()

X_engineered["temperature_difference"] = (
    X_engineered["Process temperature [K]"] - X_engineered["Air temperature [K]"]
)

X_engineered["power"] = (
    X_engineered["Rotational speed [rpm]"]
    * X_engineered["Torque [Nm]"]
    * 2
    * np.pi
    / 60
)

X_engineered["wear_torque"] = (
    X_engineered["Tool wear [min]"] * X_engineered["Torque [Nm]"]
)

X_engineered["temperature_ratio"] = (
    X_engineered["Process temperature [K]"] / X_engineered["Air temperature [K]"]
)

print(f"\nOriginal features: {X_original.shape[1]}")
print(f"Features after feature engineering: {X_engineered.shape[1]}")


# ==============================================================
# 5. TRAIN/TEST SPLIT
# ==============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_original_train = X_original.loc[train_idx]
X_original_test = X_original.loc[test_idx]

X_engineered_train = X_engineered.loc[train_idx]
X_engineered_test = X_engineered.loc[test_idx]

y_train = y.loc[train_idx]
y_test = y.loc[test_idx]

print(f"\nTraining set: {len(train_idx)} records")
print(f"Test set:     {len(test_idx)} records")

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTest class distribution:")
print(y_test.value_counts())


# ==============================================================
# 6. HELPER FUNCTIONS
# ==============================================================

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(f"Macro F1-score: {macro_f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name,
        "model": model,
        "y_pred": y_pred,
        "macro_f1": macro_f1,
    }


def save_confusion_matrix(y_test, y_pred, filename="confusion_matrix.png"):
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"\nConfusion matrix saved as {filename}")


# ==============================================================
# 7. EXPERIMENT 1: BASELINE RANDOM FOREST
# ==============================================================

baseline_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

baseline_result = evaluate_model(
    "Experiment 1: Baseline Random Forest with original features",
    baseline_model,
    X_original_train,
    X_original_test,
    y_train,
    y_test,
)


# ==============================================================
# 8. EXPERIMENT 2: RANDOM FOREST + FEATURE ENGINEERING
# ==============================================================

feature_model = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1,
)

feature_result = evaluate_model(
    "Experiment 2: Random Forest with feature engineering",
    feature_model,
    X_engineered_train,
    X_engineered_test,
    y_train,
    y_test,
)


# ==============================================================
# 9. EXPERIMENT 3: FEATURE ENGINEERING + OVERSAMPLING
# ==============================================================

final_model = Pipeline([
    ("oversampler", RandomOverSampler(random_state=RANDOM_STATE)),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )),
])

final_result = evaluate_model(
    "Experiment 3: Random Forest with feature engineering and oversampling",
    final_model,
    X_engineered_train,
    X_engineered_test,
    y_train,
    y_test,
)


# ==============================================================
# 10. CROSS-VALIDATION FOR FINAL MODEL
# ==============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_scores = cross_val_score(
    final_model,
    X_engineered_train,
    y_train,
    cv=cv,
    scoring="f1_macro",
)

print("\n" + "=" * 70)
print("Cross-validation for final model")
print("=" * 70)
print(f"Cross-validation macro F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# ==============================================================
# 11. RESULTS SUMMARY
# ==============================================================

results = pd.DataFrame([
    {
        "experiment": "Baseline Random Forest",
        "features": "Original 6 features",
        "imbalance_handling": 'class_weight="balanced"',
        "test_macro_f1": baseline_result["macro_f1"],
    },
    {
        "experiment": "Random Forest + feature engineering",
        "features": "Original features + 4 engineered features",
        "imbalance_handling": 'class_weight="balanced"',
        "test_macro_f1": feature_result["macro_f1"],
    },
    {
        "experiment": "Final model",
        "features": "Original features + 4 engineered features",
        "imbalance_handling": 'RandomOverSampler + class_weight="balanced"',
        "test_macro_f1": final_result["macro_f1"],
    },
])

print("\n" + "=" * 70)
print("Experiment summary")
print("=" * 70)
print(results.to_string(index=False))

results.to_csv("results_summary.csv", index=False)
print("\nResults summary saved as results_summary.csv")


# ==============================================================
# 12. SAVE FINAL CONFUSION MATRIX
# ==============================================================

save_confusion_matrix(
    y_test,
    final_result["y_pred"],
    filename="confusion_matrix.png",
)
