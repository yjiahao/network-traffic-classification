# Network Traffic Classification

Dataset Description: https://www.cse.wustl.edu/~jain/iiot2/index.html

Research paper: https://www.sciencedirect.com/science/article/pii/S2214212625000511

## Plausible approach to why we try each model

1. Perform some feature engineering and apply traditional ML models (with methods accounting for class imbalance)
2. We say that the ML models do not take into account the time aspect so we will try to account for time using sequence models
3. Say that sequence models do not account for the relationships between the ip addresses so we will attempt to use graphs

### Plausible method for prediction

1. Predict directly whether traffic is normal or not normal (binary classification)
2. Predict directly what type of traffic it is (6 class classification)
3. First predict whether traffic is normal or not normal, then for the not normal ones, add another classifier to predict what abnormal traffic it is (this should probably yield the best performance)

Paper on ways we can use the traditional ML or sequence models: https://www.mdpi.com/2076-3417/15/6/3121
