from sklearn.naive_bayes import GaussianNB
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
# 3️⃣ Define Gaussian Naive Bayes Pipeline
# =====================================

gnb = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("model", GaussianNB())
])

# =====================================
# 4️⃣ Train Model
# =====================================

gnb.fit(X_train, y_train)

# =====================================
# 5️⃣ Evaluate Model
# =====================================

metrics, y_pred = evaluate_model(gnb, X_test, y_test)

print_metrics("Gaussian Naive Bayes", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "Gaussian Naive Bayes")

# =====================================
# 6️⃣ Save Model
# =====================================

save_model(gnb, scaler, "naive_bayes.pkl")
