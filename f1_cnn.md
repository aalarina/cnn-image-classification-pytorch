# F1 metrics (CNN model)

The model was trained for 10 epochs on a highly imbalanced dataset (class 0: minority, class 1: majority). The reported metrics include train/validation loss, F1 scores, and confusion matrices.

## Training Performance

<img width="981" height="451" alt="image" src="https://github.com/user-attachments/assets/2d43249d-c9cd-454b-90be-8384b4ca4b58" />
<img width="981" height="451" alt="image" src="https://github.com/user-attachments/assets/2d43249d-c9cd-454b-90be-8384b4ca4b58" />


- Loss steadily decreases on the training set: Starts at 0.2214 → ends at 0.0684.

- F1 score on training improves: Starts at 0.9361 → ends at 0.9764.

- Confusion matrices show that the model predicts the majority class (class 1) very well.
Minority class (class 0) predictions improve gradually but remain challenging.
Model is learning effectively on the training data.

High F1 score indicates strong overall performance, but the class imbalance still influences the predictions.

## Validation Performance

- Validation loss fluctuates: Between 0.7167 and 1.4429. Not strictly decreasing, suggesting overfitting to the training set.

- Validation F1 score drops over epochs: Starts high at 0.8972 → drops to 0.5528 by epoch 10.

- Confusion matrices reveal many false negatives for the minority class (class 0).

## Epoch 1 (best F1 on validation):

Val Confusion Matrix:

[[ 2 34]

 [ 3 321]]
 
Class 0 is under-predicted (2/36 correct), class 1 is mostly correct.

Even at best F1, the minority class is poorly recognized.

## Test

- Model generalizes reasonably well to the majority class.

- Minority class (class 0) is still poorly recognized on the test set.

Despite limitations, the model provides a baseline benchmark for artifact detection on this dataset.


