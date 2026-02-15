## ROLL NO : 2025AB05282
## BITS ID : 2025ab05282@wilp.bits-pilani.ac.in
## Machine Learning Assignment - 2
 Implement multiple classification models - Build an interactive Streamlit web application to demonstrate your models - Deploy the app on Streamlit Community Cloud  (FREE) - Share clickable links for evaluation

## a) Problem Statement
 
The objective of this assignment is to implement and compare multiple machine learning models for a classification problem. 

- Implement and compare six different classification algorithms
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost

- Evaluate them using multiple performance metrics
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)

- Deploy an interactive Streamlit web application
 
This project demonstrates an end-to-end machine learning workflow including model training, evaluation, comparison, and deployment.
 
---
 
## b) Dataset Description
 
Dataset Name: Breast Cancer Wisconsin Dataset

Dataset Source : Breast Cancer Wisconsin (Diagnostic) Data Set

Link : https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
 
Dataset Shape: (569, 32)
 
After removing the `id` and 'diagnosis' column, 30 numerical features were used for modeling.
 
Target Variable: diagnosis
- B (Benign) → 0
- M (Malignant) → 1
 
Train-Test Split: 80/20 split
- Training Set: (455, 30)
- Test Set: (114, 30)
 
Class Distribution:
- Train: [285 Benign, 170 Malignant]
- Test: [72 Benign, 42 Malignant]
 
About dataset : 
Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image, including radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.
 
---
 
## c) Models Used and Evaluation Metrics
 
The following classification models were implemented:
 
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifi er - Gaussian or Multinomial
5. Ensemble Model - Random Forest
6. Ensemble Model - XGBoost
 
Evaluation Metrics Used:
1. Accuracy
2. AUC Score
3. Precision
4. Recall
5. F1 Score
6. Matthews Correlation Coeffi cient (MCC Score)
 
---


## 📊 Model Performance Comparison

| ML Model Name               | Accuracy | AUC    | Precision | Recall  | F1 Score | MCC    |
|-----------------------------|----------|--------|-----------|---------|----------|--------|
| Logistic Regression         | 0.9824   | 0.9974 | 0.9951    | 0.9575  | 0.9760   | 0.9626 |
| Decision Trees              | 0.9859   | 0.9850 | 0.9811    | 0.9811  | 0.9811   | 0.9699 |
| kNN                         | 0.9912   | 0.9974 | 0.9952    | 0.9811  | 0.9881   | 0.9812 |
| Naïve Bayes                 | 0.9402   | 0.9888 | 0.9406    | 0.8962  | 0.9179   | 0.8716 |
| Random Forest (Ensemble)    | 0.9947   | 0.9998 | 1.0000    | 0.9858  | 0.9929   | 0.9888 |
| XGBoost (Ensemble)          | 0.9947   | 0.9988 | 1.0000    | 0.9858  | 0.9929   | 0.9888 |



## 📊 Model Performance Observations (Binary Classification)

| ML Model | Observation |
|----------|------------|
| **Logistic Regression** | Performed extremely well with a high AUC (0.9974), indicating strong class separability. The model effectively ranks positive instances higher than negative ones, making it suitable for cancer diagnosis. Slightly lower Recall (0.9575) suggests some minority cases are missed. |
| **Decision Tree** | Demonstrated balanced Precision and Recall (0.9811) with strong MCC (0.9699), but lower AUC (0.9850) compared to other models. May suffer from overfitting and lacks the stability provided by ensemble methods. |
| **KNN** | Achieved high Accuracy (0.9912) and AUC (0.9974), with strong Recall (0.9811) and high MCC (0.9812). Performs well due to clear class clustering, but performance may degrade if class imbalance increases or dataset size grows. |
| **Gaussian Naïve Bayes** | Recorded the lowest Recall (0.8962) and MCC (0.8716). Struggles with minority class detection due to the feature independence assumption. Not ideal for imbalanced datasets with correlated features. |
| **Random Forest** | Achieved the highest AUC (0.9998), perfect Precision (1.0), very high Recall (0.9858), and highest MCC (0.9888). Provides the best balance between false positives and false negatives. Handles imbalance effectively through bootstrapping and feature randomness, making it the most reliable model. |
| **XGBoost** | Delivered performance nearly identical to Random Forest, with high Recall (0.9858) and MCC (0.9888). Boosting improves focus on difficult (minority) samples. Strong candidate for imbalanced binary classification, especially with proper hyperparameter tuning. |




---
 
## Final Conclusion
 
Most Reliable Models:
Random Forest
XGBoost

Ensemble models (Random Forest and XGBoost) performed best overall.
Random Forest achieved the highest AUC and demonstrated strong generalization capability. Keeping false positives extremely low, ensuring correct diagnosis. 
 
Given the medical nature of the dataset, high recall for malignant cases is critical. Both Random Forest and XGBoost maintained high recall (0.9858) while also achieving perfect precision.


---
 
## Streamlit Application Features
 
- Dataset upload feature
- Dropdown to select the model 
- Display of Evaluation metric
- Visualization of Confusion matrix 
- Classification report
 
---


## Deployment
 
Streamlit App Link: https://mlassignment2-zbsnv9bulpghea8nuhnfqz.streamlit.app/


---



