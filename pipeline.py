"""
AI4I 2020 Predictive Maintenance Pipeline
Author: Senior Machine Learning Engineer
Description: End-to-end machine learning pipeline to predict machine failures
             using the AI4I 2020 Predictive Maintenance Dataset.
             Uses scikit-learn, imblearn, and LightGBM with strict leakage prevention.
"""

import os
import urllib.request
import zipfile
import io
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def download_and_extract_data(url: str, extract_to: str = ".") -> str:
    """
    Downloads the dataset from the given URL and extracts it.
    """
    csv_filename = "ai4i2020.csv"
    csv_path = os.path.join(extract_to, csv_filename)
    
    if os.path.exists(csv_path):
        print(f"Dataset already exists at: {csv_path}")
        return csv_path
        
    print(f"Downloading dataset from {url}...")
    try:
        # Request with User-Agent to avoid potential blocks
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        with urllib.request.urlopen(req) as response:
            zip_file = zipfile.ZipFile(io.BytesIO(response.read()))
            # Find the CSV file inside the zip
            for name in zip_file.namelist():
                if name.endswith('.csv'):
                    zip_file.extract(name, extract_to)
                    print(f"Extracted: {name}")
                    if name != csv_filename:
                        os.rename(os.path.join(extract_to, name), csv_path)
                    return csv_path
    except Exception as e:
        print(f"Failed to download from official source: {e}")
        # Alternative public backup source if official UCI is down
        backup_url = "https://raw.githubusercontent.com/rstudio/tensorflow-playground/master/data/ai4i2020.csv"
        print(f"Attempting to download from backup URL: {backup_url}")
        req = urllib.request.Request(
            backup_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'}
        )
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            with open(csv_path, 'w') as f:
                f.write(content)
            print(f"Downloaded backup and saved to: {csv_path}")
            return csv_path
            
    raise FileNotFoundError("Could not download or locate the dataset file.")


def load_and_preprocess_data(csv_path: str):
    """
    Loads the CSV dataset and performs feature engineering and column drop steps.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("Initial dataset shape:", df.shape)
    print("Columns in dataset:", df.columns.tolist())
    
    # 1. Drop purely identifier columns (UDI, Product ID)
    # The columns in the dataset are 'UDI' and 'Product ID'
    id_cols = ['UDI', 'Product ID']
    df = df.drop(columns=[col for col in id_cols if col in df.columns], errors='ignore')
    
    # 2. Drop failure mode target-leakage columns (TWF, HDF, PWF, OSF, RNF)
    leakage_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df = df.drop(columns=[col for col in leakage_cols if col in df.columns], errors='ignore')
    
    # 3. Create logically sound engineered features
    # 'Air temperature [K]' and 'Process temperature [K]'
    # 'Rotational speed [rpm]' and 'Torque [Nm]'
    
    # Map raw column names to make them easier to handle if needed, or use exact names
    air_temp_col = 'Air temperature [K]'
    proc_temp_col = 'Process temperature [K]'
    rot_speed_col = 'Rotational speed [rpm]'
    torque_col = 'Torque [Nm]'
    
    if air_temp_col in df.columns and proc_temp_col in df.columns:
        df['Temperature_Difference'] = df[proc_temp_col] - df[air_temp_col]
        print("Engineered feature created: Temperature_Difference")
    else:
        raise KeyError("Temperature columns not found in dataset")
        
    if rot_speed_col in df.columns and torque_col in df.columns:
        df['Power'] = df[rot_speed_col] * df[torque_col]
        print("Engineered feature created: Power")
    else:
        raise KeyError("Rotational speed or Torque columns not found in dataset")
        
    # Separate features and target
    # Target variable: Machine failure
    target_col = 'Machine failure'
    if target_col not in df.columns:
        # Sometimes it might be named differently or check case
        possible_targets = [c for c in df.columns if 'failure' in c.lower()]
        if possible_targets:
            target_col = possible_targets[0]
        else:
            raise KeyError("Target column 'Machine failure' not found.")
            
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y)} ({np.mean(y)*100:.2f}% failures)")
    
    return X, y


def build_pipeline(categorical_cols, numerical_cols):
    """
    Constructs the preprocessor, SMOTE, and LightGBM pipeline.
    """
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(drop='first'), categorical_cols)
        ]
    )
    
    # SMOTE oversampler (strictly inside the pipeline to prevent data leakage during CV)
    smote = SMOTE(random_state=42)
    
    # LightGBM Classifier
    # Since we are using SMOTE to balance the training fold, we might not need is_unbalance=True,
    # or we can use it to give extra weight to failures. Let's configure it.
    classifier = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        verbosity=-1,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        # is_unbalance=True or scale_pos_weight can be set if SMOTE alone is not enough
        # we will use is_unbalance=True as a native handling parameter alongside SMOTE as requested
        is_unbalance=True 
    )
    
    # imblearn pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', smote),
        ('classifier', classifier)
    ])
    
    return pipeline


def main():
    # URL for dataset zip archive on UCI
    dataset_url = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
    
    # Download and load data
    csv_path = download_and_extract_data(dataset_url)
    X, y = load_and_preprocess_data(csv_path)
    
    # Identify categorical and numerical columns
    categorical_cols = ['Type']
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    
    print("\nCategorical columns:", categorical_cols)
    print("Numerical columns:", numerical_cols)
    
    # Build pipeline
    pipeline = build_pipeline(categorical_cols, numerical_cols)
    
    # 1. 5-Fold Stratified Cross-Validation
    print("\n--- 5-Fold Stratified Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    scoring = ['f1_macro', 'recall', 'precision']
    
    cv_results = cross_validate(
        pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False
    )
    
    print(f"Mean Macro F1 Score: {np.mean(cv_results['test_f1_macro']):.4f} +/- {np.std(cv_results['test_f1_macro']):.4f}")
    print(f"Mean Recall Score:   {np.mean(cv_results['test_recall']):.4f} +/- {np.std(cv_results['test_recall']):.4f}")
    print(f"Mean Precision Score: {np.mean(cv_results['test_precision']):.4f} +/- {np.std(cv_results['test_precision']):.4f}")
    
    # 2. Holdout Train/Test Split (80/20) for Final Verification
    print("\n--- Final Train/Test Split Evaluation ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # Fit the pipeline on the training data (SMOTE is applied ONLY to training set)
    print("Fitting pipeline on training set...")
    pipeline.fit(X_train, y_train)
    
    # Predict on holdout test set (SMOTE is NOT applied to test set)
    y_pred = pipeline.predict(X_test)
    
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    print(f"\nHoldout Test Macro F1 Score: {macro_f1:.4f}")
    
    # Standard classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Verification check
    if macro_f1 >= 0.70:
        print("\nSUCCESS: Macro F1 Score is >= 0.70!")
    else:
        print("\nWARNING: Macro F1 Score did not meet the 0.70 target. Tuning may be required.")


if __name__ == "__main__":
    main()
