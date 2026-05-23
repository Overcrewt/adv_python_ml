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

I tested the model in three stages. The goal was to start with a simple baseline and then add improvements one by one.

| Experiment | Features | Imbalance handling | Test macro F1 |
|---|---|---|---|
| Baseline Random Forest | Original 6 features | class_weight="balanced" | 0.5548 |
| Random Forest + feature engineering | Original features + 4 engineered physical features | class_weight="balanced" | 0.6516 |
| Final model | Original features + 4 engineered physical features | RandomOverSampler + class_weight="balanced" | 0.6852 |

The baseline model already performed well on the majority class, but the macro F1-score was limited by rare failure types. Adding feature engineering produced the largest improvement. This makes sense because the engineered features describe real physical failure mechanisms: power for PWF, temperature difference for HDF, and wear-torque product for OSF.

Random oversampling gave a smaller additional improvement. It helped the model see rare classes more often during training, but it did not fully solve the problem of extremely rare or random failures.

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

## 8. Limitations

This model should be treated as an educational predictive maintenance experiment, not as a production-ready maritime safety system.

The main limitation is the extreme class imbalance. The No Failure class dominates the dataset, while TWF and RNF have very few examples. RandomOverSampler helps by duplicating minority class samples in the training set, but it does not create genuinely new information.

RNF is especially difficult because it represents random failure. If the failure is random, there may be no clear sensor pattern for the model to learn. This explains why RNF was not detected in the final test evaluation.

Another limitation is that the dataset is small and synthetic. In a real engine room, sensor noise, maintenance history, operating modes, and equipment-specific thresholds would need to be considered.

## 9. Reflection

The most useful improvement was feature engineering. Adding physically meaningful features improved the macro F1-score more than just using the original dataset columns.

Random oversampling also helped, but it did not fully solve the rare class problem. If I had more time, I would try other models such as LightGBM or XGBoost, compare them with Random Forest, and test different resampling strategies. I would also investigate whether RNF should be treated separately because it is random and extremely rare.

