import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ==============================================================
# 1. LOAD DATA
# ==============================================================

# Download from Kaggle:
# https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020
# Save as 'ai4i2020.csv' in your working directory

df = pd.read_csv("ai4i2020.csv")

print(f"Dataset: {df.shape[0]} records, {df.shape[1]} columns")

# ==============================================================
# 2. CREATE MULTI-CLASS TARGET
# ==============================================================

# The dataset has separate binary columns for each failure type.
# We combine them into a single target variable.

def get_failure_type(row):
    if row['TWF'] == 1:
        return 'TWF'
    elif row['HDF'] == 1:
        return 'HDF'
    elif row['PWF'] == 1:
        return 'PWF'
    elif row['OSF'] == 1:
        return 'OSF'
    elif row['RNF'] == 1:
        return 'RNF'
    else:
        return 'No Failure'

df['fault_type'] = df.apply(get_failure_type, axis=1)

# ==============================================================
# 3. PREPARE FEATURES
# ==============================================================

# Drop identifiers and original failure columns
drop_cols = ['UDI', 'Product ID', 'Machine failure',
             'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'fault_type']

X = df.drop(columns=drop_cols)
y = df['fault_type']

# Encode the categorical 'Type' column (L, M, H)
le = LabelEncoder()
X['Type'] = le.fit_transform(X['Type'])

print(f"Features: {X.shape[1]}")
print(f"\nClass distribution:\n{y.value_counts()}")

# ==============================================================
# 4. TRAIN/TEST SPLIT
# ==============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {X_train.shape[0]} records")
print(f"Test set:     {X_test.shape[0]} records")

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================

# 1. Feature Engineering (applied separately to train and test sets to maintain strict leakage prevention)
def engineer_features(data):
    df_feat = data.copy()
    
    # Temperature Difference (Process temperature - Air temperature)
    df_feat['Temp_Diff'] = df_feat['Process temperature [K]'] - df_feat['Air temperature [K]']
    
    # Power = Rotational speed [rpm] * Torque [Nm] * (2 * pi / 60)
    df_feat['Power_W'] = df_feat['Rotational speed [rpm]'] * df_feat['Torque [Nm]'] * (2 * np.pi / 60)
    
    # Tool wear * torque
    df_feat['Wear_Torque'] = df_feat['Tool wear [min]'] * df_feat['Torque [Nm]']
    
    # Binary indicators of rule candidates
    df_feat['HDF_candidate'] = ((df_feat['Temp_Diff'] < 8.6) & (df_feat['Rotational speed [rpm]'] < 1380)).astype(int)
    df_feat['PWF_candidate'] = ((df_feat['Power_W'] < 3500) | (df_feat['Power_W'] > 9000)).astype(int)
    
    # OSF candidate based on Type encoding (L -> 1, M -> 2, H -> 0 based on LabelEncoder alphabetical sorting)
    df_feat['OSF_candidate'] = 0
    df_feat.loc[(df_feat['Type'] == 1) & (df_feat['Wear_Torque'] > 11000), 'OSF_candidate'] = 1
    df_feat.loc[(df_feat['Type'] == 2) & (df_feat['Wear_Torque'] > 12000), 'OSF_candidate'] = 1
    df_feat.loc[(df_feat['Type'] == 0) & (df_feat['Wear_Torque'] > 13000), 'OSF_candidate'] = 1
    
    # TWF candidate (Tool wear between 200 and 240)
    df_feat['TWF_candidate'] = ((df_feat['Tool wear [min]'] >= 200) & (df_feat['Tool wear [min]'] <= 240)).astype(int)
    
    return df_feat

print("\nEngineering features...")
X_train_engineered = engineer_features(X_train)
X_test_engineered = engineer_features(X_test)

# Encode targets (LightGBM requires integer target values)
target_le = LabelEncoder()
y_train_encoded = target_le.fit_transform(y_train)
y_test_encoded = target_le.transform(y_test)

# Build imblearn pipeline to prevent data leakage during scaling/resampling
scaler = StandardScaler()
smote = SMOTE(random_state=42, k_neighbors=3)
lgb_classifier = lgb.LGBMClassifier(
    objective='multiclass',
    class_weight='balanced',
    random_state=42,
    verbosity=-1,
    n_estimators=100
)

pipeline = Pipeline([
    ('scaler', scaler),
    ('smote', smote),
    ('classifier', lgb_classifier)
])

# Train model
print("Training pipeline...")
pipeline.fit(X_train_engineered, y_train_encoded)

# Predict on test set
print("Predicting on test set...")
y_pred_encoded = pipeline.predict(X_test_engineered)

# Map predictions back to string labels for evaluation
y_pred = target_le.inverse_transform(y_pred_encoded)

# Evaluate using Macro F1 Score
macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nHoldout Test Macro F1 Score: {macro_f1:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==============================================================
# CONFUSION MATRIX PLOT
# ==============================================================

labels = ["No Failure", "HDF", "OSF", "PWF", "TWF", "RNF"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Macro F1 Score = {macro_f1:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion matrix heatmap to 'confusion_matrix.png'.")
plt.show()
