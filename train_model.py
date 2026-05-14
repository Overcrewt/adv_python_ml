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

print("\n" + "="*60)
print("5. MODEL TRAINING AND EVALUATION")
print("="*60 + "\n")

# To handle class imbalance, we use a RandomForestClassifier with class_weight='balanced'
print("Initializing Random Forest Classifier with class_weight='balanced'...")
model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42, n_jobs=-1)

# Perform 5-fold cross-validation on the training set using Macro F1-score
print("\nPerforming 5-fold cross-validation (this may take a few minutes)...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
print(f"Cross-validation macro F1: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# Train the model on the entire training set
print("\nTraining the final model on the entire training set...")
model.fit(X_train, y_train)

# Predict on the test set
print("Evaluating on the test set...")
y_pred = model.predict(X_test)

# Evaluate using Macro F1-score
final_macro_f1 = f1_score(y_test, y_pred, average='macro')
print(f"\nFinal Test Macro F1-score: {final_macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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
plt.title("Confusion Matrix (Random Forest with Balanced Weights)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved as 'confusion_matrix.png'.")
