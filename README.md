# Engine Room Early Warning System

This project trains a machine learning model for predictive maintenance using the AI4I 2020 Predictive Maintenance dataset.

The model classifies equipment sensor readings into six classes:

- No Failure
- HDF
- PWF
- OSF
- TWF
- RNF

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the model:
```bash
python train_model.py
```

The script prints the classification report, macro F1-score, cross-validation score, and saves the confusion matrix as confusion_matrix.png.

## Final result

### Danila's Model (Random Forest + RandomOverSampler)
- **Holdout Test Macro F1**: 0.6852
- **Cross-Validation Macro F1**: 0.6341 (+/- 0.0116)

### Dmitri's Model (LightGBM + SMOTE + Rule-based Features)
- **Holdout Test Macro F1**: **0.7036** (Successfully exceeds 0.70 threshold)
- **Cross-Validation Macro F1**: **0.6757** (+/- 0.0123)

