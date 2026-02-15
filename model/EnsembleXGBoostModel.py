from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

from utility import (
    load_dataset,
    preprocess_data,
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    print_classification_report,
    save_model
)

# =====================================
# 1️⃣ Load Dataset
# =====================================

X, y = load_dataset()

# =====================================
# 2️⃣ Preprocess Data
# =====================================

X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# =====================================
# 3️⃣ Define XGBoost Pipeline
# =====================================

xgb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    ))
])

# =====================================
# 4️⃣ Train Model
# =====================================

xgb.fit(X_train, y_train)

# =====================================
# 5️⃣ Evaluate Model
# =====================================

metrics, y_pred = evaluate_model(xgb, X_test, y_test)

print_metrics("XGBoost (Ensemble)", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "XGBoost (Ensemble)")

# =====================================
# 6️⃣ Save Model
# =====================================

save_model(xgb, scaler, "xgboost.pkl")
