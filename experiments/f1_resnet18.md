# F1-Score Analysis — ResNet-18 Model

## Experiment Overview

- This experiment evaluates a ResNet-18 model fine-tuned for binary artifact classification on a highly imbalanced dataset:

- Class 0 — Artifact (minority class)

- Class 1 — Clean image (majority class)

Due to class imbalance, F1-score is used as the primary evaluation metric.

## Training Performance

**Loss**

- Decreases steadily: 0.6640 → 0.0631

- Indicates stable optimization

- Strong convergence behavior

**F1 Score**

- Improves significantly: 0.8090 → 0.9910

- Near-perfect training F1 by final epoch

This demonstrates that ResNet-18 has high representational capacity and easily fits the training data.

**Confusion Matrix Insights (Training)**

- Majority class (Class 1) is predicted very accurately.

- Minority class (Class 0) predictions improve slightly across epochs.

However, minority samples remain underrepresented in some epochs.

**Conclusion:** ResNet-18 learns dominant patterns extremely well. But strong training performance does not guarantee generalization.

## Validation Performance

**Loss**

- Fluctuates significantly: 0.6083 → 2.4190

- No stable downward trend

- Clear signs of overfitting

**F1 Score**

- Peaks at 0.9056 (epoch 9)

- Drops sharply to 0.5028 (epoch 10)

The sharp drop in the final epoch indicates overfitting to the training data.

## Test Set Performance

- All minority class samples (Class 0) were misclassified as Class 1.

- Majority class predictions are perfect.

This highlights:

- Strong bias toward majority class

- Persistent impact of class imbalance

Deeper architecture alone does not solve imbalance issues.

## Key Observations

- ResNet-18 achieves near-perfect training performance.

- Validation instability suggests overfitting.

- Peak F1 before final epoch indicates early stopping necessity.

Class imbalance remains the limiting factor despite deeper architecture.

## Future improvements may include:

- Weighted loss or Focal Loss

- Oversampling / undersampling

- Stronger augmentation

- Early stopping

- Threshold tuning

- Balanced batch sampling
