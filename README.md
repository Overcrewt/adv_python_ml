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

Final test macro F1-score: 0.6852

Cross-validation macro F1-score: 0.6330 (+/- 0.0129)
