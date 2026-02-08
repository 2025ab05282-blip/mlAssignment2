import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    matthews_corrcoef
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC


st.title("📊 Multi-Model Classification Benchmark App")

# ----------------------------------------------------
# Upload dataset
# ----------------------------------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical
    X = pd.get_dummies(X)

    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    test_size = st.slider("Test Size (%)", 10, 40, 20) / 100

    # ----------------------------------------------------
    # All 6 required models
    # ----------------------------------------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric="logloss")
    }

    if st.button("Train & Evaluate All Models"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        results = []
        predictions = {}

        n_classes = len(np.unique(y_test))

        # ----------------------------------------------------
        # Train each model and compute ALL metrics
        # ----------------------------------------------------
        for name, model in models.items():

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)

            # AUC (binary vs multi-class)
            if n_classes == 2:
                auc = roc_auc_score(y_test, y_proba[:, 1])
            else:
                auc = roc_auc_score(
                    y_test, y_proba,
                    multi_class="ovr",
                    average="weighted"
                )

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")
            mcc = matthews_corrcoef(y_test, y_pred)

            predictions[name] = y_pred

            results.append([name, acc, auc, prec, rec, f1, mcc])

        # ----------------------------------------------------
        # Metrics table
        # ----------------------------------------------------
        st.subheader("📈 Model Performance Comparison")

        results_df = pd.DataFrame(
            results,
            columns=[
                "Model", "Accuracy", "AUC",
                "Precision", "Recall", "F1 Score", "MCC"
            ]
        )

        st.dataframe(results_df.sort_values("Accuracy", ascending=False))

        # ----------------------------------------------------
        # Detailed evaluation
        # ----------------------------------------------------
        st.subheader("Detailed Evaluation")

        selected_model = st.selectbox(
            "Choose model",
            list(models.keys())
        )

        y_pred = predictions[selected_model]

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))
