# Model Comparison

This document compares two architectures trained for binary artifact classification on a highly imbalanced dataset:

- Custom CNN

- ResNet-18 (Transfer Learning)

Primary evaluation metric: F1-score

## Training Behavior

**Custom CNN**

- Stable training curve

- Gradual F1 improvement

- Slower overfitting

**ResNet-18**

- Faster convergence

- Near-perfect training F1

- Stronger overfitting behavior

Observation: ResNet-18 has higher representational capacity and fits the training data more aggressively.

## Minority Class Performance

- Both models struggle with the minority class (Class 0).

However:

- CNN shows slight gradual improvement across epochs.

- ResNet-18 improves mid-training but collapses in later epochs.

- On test set, ResNet-18 misclassifies all minority samples.

## Key Takeaways

- Model depth alone does not solve imbalance.

- Both architectures overfit due to limited minority samples.

- Early stopping is essential for ResNet-18.

- Balanced sampling may improve minority recall.

## Future Improvements

To improve performance:

- Weighted CrossEntropy

- Oversampling minority class

- Stratified batch sampling

- Stronger augmentation

- Threshold tuning

- Early stopping

- Ensemble methods
