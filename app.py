import streamlit as st
import pandas as pd
import pickle
import os

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

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# =====================================
# Page Config
# =====================================

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification App")
st.write("Upload a test CSV file. The last column must be the target variable. 'diagnosis' ")


# =====================================
# Load Scaler (Cached)
# =====================================

@st.cache_resource
def load_scaler():
    return pickle.load(open("model/scaler.pkl", "rb"))


@st.cache_resource
def load_model(path):
    return pickle.load(open(path, "rb"))

@st.cache_resource
def load_feature_names():
    return pickle.load(open("model/feature_names.pkl", "rb"))


# =====================================
# File Upload
# =====================================

uploaded_file = st.file_uploader("Upload Test CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # =====================================
    # Load Feature Names
    # =====================================

    feature_names = load_feature_names()

    # Check if target column exists
    if "diagnosis" not in data.columns:
        st.error("Target column 'diagnosis' not found in uploaded file.")
        st.stop()

    # Remove any accidental index columns
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]

    # Keep only training features
    feature_names = load_feature_names()
    X = data[feature_names]

    # Separate target
    y_raw = data["diagnosis"]

    # Split features & target
    #X = data.iloc[:, :-1]
    #y = data.iloc[:, -1]

    if "id" in data.columns:
        data = data.drop(columns=["id"])

    X = data.drop(columns=["diagnosis"])

    y_raw = data["diagnosis"]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    #print("Target mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
    # Typically: {'B': 0, 'M': 1}

    # =====================================
    # Model Selection Dropdown
    # =====================================

    model_choice = st.selectbox(
        "Select Model",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    model_files = {
        "Logistic Regression": "model/logistic_regression.pkl",
        "Decision Tree": "model/decision_tree.pkl",
        "KNN": "model/knn.pkl",
        "Naive Bayes": "model/naive_bayes.pkl",
        "Random Forest": "model/random_forest.pkl",
        "XGBoost": "model/xgboost.pkl"
    }

    model_path = model_files[model_choice]

    if not os.path.exists(model_path):
        st.error("Model not found. Please run training scripts first.")
        st.stop()

    # =====================================
    # Load Model & Scaler
    # =====================================

    model = load_model(model_path)
    scaler = load_scaler()

    # Scale features
    X_scaled = scaler.transform(X)

    # =====================================
    # Predict
    # =====================================



    y_pred = model.predict(X_scaled)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
    else:
        y_prob = None


    # =====================================
    # Evaluation Metrics
    # =====================================

    st.header("Selected Model : "+model_choice)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    if y_prob is not None:
        auc = roc_auc_score(y, y_prob)
        st.metric("AUC Score", f"{auc:.4f}")
    col1.metric("Precision", f"{precision:.4f}")

    col2.metric("Recall", f"{recall:.4f}")
    col2.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")




    # =====================================
    # Confusion Matrix
    # =====================================

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)


    # =====================================
    # Classification Report
    # =====================================

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

else:
    st.info("Please upload a CSV file to begin.")
