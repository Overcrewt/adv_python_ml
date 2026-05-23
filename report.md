# Engine Room Early Warning System

## 1. Introduction

The goal of this assignment was to train a machine learning model for predictive maintenance. The model classifies equipment sensor readings into six classes: No Failure, HDF, PWF, OSF, TWF, and RNF.

The main evaluation metric is macro F1-score, because the dataset is highly imbalanced. Most records belong to the No Failure class, while failure classes are rare.

## 2. Dataset and preprocessing

I used the AI4I 2020 Predictive Maintenance dataset with 10,000 records.

The columns UDI and Product ID were removed because they are identifiers and do not describe the machine state. The original binary failure columns were combined into one multiclass target column called fault_type.

The Type column was encoded from categorical values into numeric values.

## 3. Feature engineering

I added four engineered features:

- temperature_difference = Process temperature - Air temperature
- power = Rotational speed * Torque * 2π / 60
- wear_torque = Tool wear * Torque
- temperature_ratio = Process temperature / Air temperature

These features were added because they have physical meaning. Temperature difference can help detect heat dissipation failure. Power is directly related to power failure. Wear-torque product helps detect overstrain failure.

## 4. Models and experiments

First, I trained a Random Forest model with class_weight="balanced". This baseline reached macro F1-score of 0.5548.

After adding feature engineering, the macro F1-score improved to 0.6516.

Then I added RandomOverSampler to handle class imbalance in the training set. The final model used a pipeline with RandomOverSampler and RandomForestClassifier. This improved the test macro F1-score to 0.6852.

## 5. Cross-validation and test result

The final model was evaluated with 5-fold stratified cross-validation on the training set.

Cross-validation macro F1-score:

0.6330 (+/- 0.0129)

Final test macro F1-score:

0.6852

The test score is slightly higher than the cross-validation score. The difference is not very large, so the result does not look suspicious. The model generalizes reasonably well, but rare classes remain difficult.

## 6. Final classification report

```text
              precision    recall  f1-score   support

         HDF       1.00      1.00      1.00        23
  No Failure       0.99      1.00      1.00      1930
         OSF       0.84      1.00      0.91        16
         PWF       1.00      1.00      1.00        18
         RNF       0.00      0.00      0.00         4
         TWF       1.00      0.11      0.20         9

    accuracy                           0.99      2000
   macro avg       0.81      0.69      0.69      2000
weighted avg       0.99      0.99      0.99      2000
```

## 7. Confusion matrix

![Confusion Matrix](confusion_matrix.png)

The confusion matrix shows that the model performs very well on No Failure, HDF, OSF and PWF. However, TWF is detected only partially, and RNF is not detected.

RNF is especially difficult because it has very few examples and represents random failure. With only 18 records in the full dataset and 4 records in the test split, the model does not have enough training examples to learn this class reliably.

## 8. Reflection

The most useful improvement was feature engineering. Adding physically meaningful features improved the macro F1-score more than just using the original dataset columns.

Random oversampling also helped, but it did not fully solve the rare class problem. If I had more time, I would try other models such as LightGBM or XGBoost, compare them with Random Forest, and test different resampling strategies. I would also investigate whether RNF should be treated separately because it is random and extremely rare.

