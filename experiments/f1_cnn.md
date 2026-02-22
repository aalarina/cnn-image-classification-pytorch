# F1-Score Analysis — Custom CNN Model

## Experiment Overview

The model was trained for 10 epochs on a highly imbalanced dataset (class 0: minority, class 1: majority).
Due to class imbalance (~1:9 ratio), the primary evaluation metric is F1-score, as it better reflects performance than accuracy alone.

<img width="981" height="451" alt="image" src="https://github.com/user-attachments/assets/2d43249d-c9cd-454b-90be-8384b4ca4b58" />

## Training Performance

**Loss**

- Decreases steadily: 0.2214 → 0.0684

- Indicates stable optimization

- No instability during training

**F1 Score**

- Improves consistently: 0.9361 → 0.9764

- High training F1 suggests strong learning capacity

**Confusion Matrix Insights**

- Majority class (Class 1) is predicted almost perfectly.

- Minority class (Class 0) improves over time but remains under-predicted.

- The model clearly learns dominant patterns in the dataset.

**Conclusion:**
The model fits the training data very well. However, the class imbalance still influences the predictions.

## Validation Performance

**Loss**

- Fluctuates between 0.7167 and 1.4429

- Not consistently decreasing

- Suggests early signs of overfitting

**F1 Score**

- Starts high: 0.8972

- Gradually drops to 0.5528 by epoch 10

This decline indicates that the model increasingly memorizes training data rather than learning generalizable features.

## Best Validation Epoch

**Epoch 1 (Highest Validation F1)**

Validation Confusion Matrix:

[[  2  34]
 [  3 321]]

Interpretation:

- Class 0: Only 2 out of 36 correctly classified

- Class 1: 321 out of 324 correctly classified

Even at peak validation F1:

- Minority class recall remains extremely low

- Model heavily favors the majority class

## Test Set Performance

- Strong performance on majority class (clean images)

- Minority class (artifact images) remains poorly recognized

- Model bias toward class 1 persists

## Key Observations

- High training F1 confirms model capacity.

- Validation F1 degradation suggests overfitting.

- Severe class imbalance impacts minority class detection.

Model is biased toward predicting the majority class.



