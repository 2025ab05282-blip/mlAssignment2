from sklearn.tree import DecisionTreeClassifier
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
# 3️⃣ Define Decision Tree Pipeline
# =====================================

prep_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

dt = Pipeline(steps=[
    ("prep", prep_tree),
    ("model", DecisionTreeClassifier(
        random_state=42
    ))
])

# =====================================
# 4️⃣ Train Model
# =====================================

dt.fit(X_train, y_train)

# =====================================
# 5️⃣ Evaluate Model
# =====================================

metrics, y_pred = evaluate_model(dt, X_test, y_test)

print_metrics("Decision Tree", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "Decision Tree")

# =====================================
# 6️⃣ Save Model
# =====================================

save_model(dt, scaler, "decision_tree.pkl")
