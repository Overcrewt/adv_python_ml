import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================
# 1. LOAD DATA
# ==============================================================

train_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
test_url = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'class', 'level'
]

print("Loading data...")
df_train = pd.read_csv(train_url, names=columns)
df_test = pd.read_csv(test_url, names=columns)

# Drop difficulty level column (not a feature)
df_train.drop(columns=['level'], inplace=True)
df_test.drop(columns=['level'], inplace=True)

print(f"Training set: {df_train.shape[0]} records, {df_train.shape[1]} columns")
print(f"Test set:     {df_test.shape[0]} records, {df_test.shape[1]} columns")

# ==============================================================
# 2. ENCODE CATEGORICAL FEATURES
# ==============================================================

# Merge temporarily to ensure consistent encoding across train and test
df_full = pd.concat([df_train, df_test])

# Encode categorical columns as integers
cat_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df_full[col] = le.fit_transform(df_full[col])
    label_encoders[col] = le

# ==============================================================
# 3. MAP ATTACKS TO 5 CATEGORIES
# ==============================================================

category_map = {
    'normal': 'Normal',
    # DoS
    'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
    'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS', 'apache2': 'DoS',
    'processtable': 'DoS', 'udpstorm': 'DoS', 'worm': 'DoS',
    # Probe
    'satan': 'Probe', 'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
    'mscan': 'Probe', 'saint': 'Probe',
    # R2L
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'ftp_write': 'R2L',
    'imap': 'R2L', 'phf': 'R2L', 'multihop': 'R2L', 'warezmaster': 'R2L',
    'spy': 'R2L', 'xlock': 'R2L', 'xsnoop': 'R2L', 'snmpguess': 'R2L',
    'snmpgetattack': 'R2L', 'httptunnel': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    # U2R
    'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'rootkit': 'U2R',
    'perl': 'U2R', 'sqlattack': 'U2R', 'xterm': 'U2R', 'ps': 'U2R'
}

df_full['category'] = df_full['class'].map(category_map).fillna('Other')

# ==============================================================
# 4. PREPARE FEATURES AND LABELS
# ==============================================================

# Drop constant column and original class labels
df_full.drop(columns=['num_outbound_cmds', 'class'], inplace=True)

# Split back into train and test
train_len = len(df_train)
df_train_processed = df_full.iloc[:train_len].copy()
df_test_processed = df_full.iloc[train_len:].copy()

X_train = df_train_processed.drop(columns=['category'])
y_train = df_train_processed['category']

X_test = df_test_processed.drop(columns=['category'])
y_test = df_test_processed['category']

print(f"\nFeatures: {X_train.shape[1]}")
print(f"\nTraining set class distribution:")
print(y_train.value_counts())
print(f"\nTest set class distribution:")
print(y_test.value_counts())

# ==============================================================
# YOUR WORK STARTS HERE
# ==============================================================
import os
import datetime
from sklearn.model_selection import GridSearchCV

print("\n" + "="*60)
print("5. MODEL TRAINING AND HYPERPARAMETER TUNING")
print("="*60 + "\n")

run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requires feature scaling for optimal performance and convergence.
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model and parameter grid
# Note: SVM on 125,000 samples can be extremely slow. 
# We use cache_size=2000 to allocate 2GB of RAM for the kernel cache to speed it up.
svm = SVC(random_state=42, class_weight='balanced', cache_size=2000)

param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf'] # 'linear' is omitted as it takes a very long time in SVC. Consider LinearSVC instead for purely linear kernels.
}

print("Initializing GridSearchCV with the following parameter grid:")
for k, v in param_grid.items():
    print(f"  {k}: {v}")

# 3-fold CV is usually faster for a large dataset, but 5-fold is standard.
# We'll use 3 to save time but you can increase it.
grid_search = GridSearchCV(
    estimator=svm,
    param_grid=param_grid,
    cv=3,
    scoring='f1_macro',
    n_jobs=-1,
    verbose=2
)

print("\nStarting Grid Search (this may take a while, SVM on large datasets is slow)...")
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
best_cv_score = grid_search.best_score_

print(f"\nBest Parameters Found: {best_params}")
print(f"Best Cross-Validation Macro F1: {best_cv_score:.4f}")

# Predict on the test set
print("\nEvaluating the best model on the test set...")
y_pred = best_model.predict(X_test_scaled)

final_macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nFinal Test Macro F1-score: {final_macro_f1:.4f}")

clf_report_text = classification_report(y_test, y_pred)
clf_report_dict = classification_report(y_test, y_pred, output_dict=True)
print("\nClassification Report:")
print(clf_report_text)

# ==============================================================
# 6. CONFUSION MATRIX PLOT
# ==============================================================

print("\nGenerating confusion matrix plot...")
labels = ["DoS", "Normal", "Probe", "R2L", "U2R"]

# Ensure classes are in the desired order
cm = confusion_matrix(y_test, y_pred, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Best Model)")
plt.tight_layout()
cm_filename = f"confusion_matrix_{run_timestamp}.png"
plt.savefig(cm_filename, dpi=150)
print(f"Confusion matrix saved as '{cm_filename}'.")

# ==============================================================
# 7. GENERATE MARKDOWN REPORT
# ==============================================================
print("\nGenerating markdown report...")
report_filename = "model_evaluation_report.md"

md_content = f"## Run Details: {run_timestamp}\n\n"
md_content += "**Model:** SVC\n\n"

md_content += "### Best Hyperparameters\n\n"
md_content += "| Parameter | Value |\n| :--- | :--- |\n"
for k, v in best_params.items():
    md_content += f"| `{k}` | `{v}` |\n"
md_content += "\n"

md_content += "### Performance Metrics\n\n"
md_content += f"- **Cross-Validation Macro F1-score:** `{best_cv_score:.4f}`\n"
md_content += f"- **Test Set Macro F1-score:** `{final_macro_f1:.4f}`\n\n"

md_content += "### Test Set Classification Report\n\n"
md_content += "| Class | Precision | Recall | F1-Score | Support |\n"
md_content += "| :--- | :--- | :--- | :--- | :--- |\n"
for target_class in labels:
    if target_class in clf_report_dict:
        metrics = clf_report_dict[target_class]
        md_content += f"| {target_class} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1-score']:.4f} | {metrics['support']:.0f} |\n"

macro_avg = clf_report_dict.get('macro avg', {})
if macro_avg:
    md_content += f"| **Macro Avg** | **{macro_avg['precision']:.4f}** | **{macro_avg['recall']:.4f}** | **{macro_avg['f1-score']:.4f}** | **{macro_avg['support']:.0f}** |\n"
weighted_avg = clf_report_dict.get('weighted avg', {})
if weighted_avg:
    md_content += f"| **Weighted Avg** | **{weighted_avg['precision']:.4f}** | **{weighted_avg['recall']:.4f}** | **{weighted_avg['f1-score']:.4f}** | **{weighted_avg['support']:.0f}** |\n\n"

md_content += "### Confusion Matrix\n\n"
md_content += f"![Confusion Matrix]({cm_filename})\n\n"
md_content += "---\n\n"

is_new_file = not os.path.exists(report_filename)
with open(report_filename, "a") as f:
    if is_new_file:
        f.write("# Network Intrusion Detection Model Evaluation Report\n\n")
    f.write(md_content)

print(f"Report appended to {report_filename}\n")
