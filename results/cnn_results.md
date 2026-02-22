# F1 metrics

The model was trained for 10 epochs on a highly imbalanced dataset (class 0: minority, class 1: majority). The reported metrics include train/validation loss, F1 scores, and confusion matrices.

## Training Performance

Loss steadily decreases on the training set: Starts at 0.2214 → ends at 0.0684.

F1 score on training improves: Starts at 0.9361 → ends at 0.9764.

Confusion matrices show that:

The model predicts the majority class (class 1) very well.
Minority class (class 0) predictions improve gradually but remain challenging.
Model is learning effectively on the training data.

High F1 score indicates strong overall performance, but the class imbalance still influences the predictions.

## Validation Performance

Validation loss fluctuates: Between 0.7167 and 1.4429. Not strictly decreasing, suggesting overfitting to the training set.

Validation F1 score drops over epochs: Starts high at 0.8972 → drops to 0.5528 by epoch 10.

Confusion matrices reveal:

Many false negatives for the minority class (class 0).
