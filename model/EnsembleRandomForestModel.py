from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
# 3️⃣ Define Random Forest Pipeline
# =====================================

rf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ))
])

# =====================================
# 4️⃣ Train Model
# =====================================

rf.fit(X_train, y_train)

# =====================================
# 5️⃣ Evaluate Model
# =====================================

metrics, y_pred = evaluate_model(rf, X_test, y_test)

print_metrics("Random Forest (Ensemble)", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "Random Forest (Ensemble)")

# =====================================
# 6️⃣ Save Model
# =====================================

save_model(rf, scaler, "random_forest.pkl")
