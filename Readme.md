# Car Evaluation Classifier

For [CS550: Machine Learning](https://github.com/gagan-iitb/CS550) Course, created a classifier to classify different car deals based on the attributes of the car like cost, number of doors, maintainance cost, safety, etc.

We tested with different baseline models, but found SVC to be working best. Classifier performance is reported below.
But we were unable to cross 90% accuracy with the training data, as the target distribution was not uniform. Thus we applied [SMOTE](https://arxiv.org/abs/1106.1813) to create synthetic data to handle the target imbalance problem.

After SMOTE, the number of data-points have increased, but the baseline was performing poorly due to the synthetic data. Thus we applied Ensemble models and seen an increase in accuracy.

## Results

|                | No. of Datapoints | Bad Deal Percentage | Train Accuracy | Test Accuracy |
| -------------- | ----------------- | ------------------- |---------------- |-------------- |
| SVC                         | 1554.0              | 0.07722        | 0.997426     | 0.896552 |
| SVC (SMOT)                  | 2868.0              | 0.50000        | 0.996513     | 0.879310 |
| Ensemble (SMOT)             | 2868.0              | 0.50000        | 1.000000     | 0.954023 |


## Other Contributers:
- [Shahid](https://github.com/sowdagar3)
