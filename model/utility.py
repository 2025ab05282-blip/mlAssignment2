import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.datasets import load_breast_cancer


# =====================================
# 1️⃣ Load Dataset
# =====================================

def load_dataset():
    """
    Loads dataset.
    Replace this function if using custom CSV dataset.
    """
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return X, y


# =====================================
# 2️⃣ Train-Test Split & Scaling
# =====================================

def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset and applies Standard Scaling.
    Returns scaled train/test sets and scaler.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


# =====================================
# 3️⃣ Evaluate Model
# =====================================

def evaluate_model(model, X_test, y_test):
    """
    Returns dictionary of all required evaluation metrics.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    return metrics, y_pred


# =====================================
# 4️⃣ Print Metrics
# =====================================

def print_metrics(model_name, metrics):
    print(f"\n===== {model_name} Results =====")

    for key, value in metrics.items():
        if value is not None:
            print(f"{key}: {value:.4f}")


# =====================================
# 5️⃣ Plot Confusion Matrix
# =====================================

def plot_confusion_matrix(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


# =====================================
# 6️⃣ Print Classification Report
# =====================================

def print_classification_report(y_test, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


# =====================================
# 7️⃣ Save Model
# =====================================

def save_model(model, scaler, filename):
    os.makedirs("../model", exist_ok=True)

    pickle.dump(model, open(f"../model/{filename}", "wb"))
    pickle.dump(scaler, open(f"../model/scaler.pkl", "wb"))

    print(f"\nModel saved as ../model/{filename}")
