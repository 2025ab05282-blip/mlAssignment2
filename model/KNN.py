from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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
# 3️⃣ Define KNN Pipeline
# =====================================

knn = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        p=2
    ))
])

# =====================================
# 4️⃣ Train Model
# =====================================

knn.fit(X_train, y_train)

# =====================================
# 5️⃣ Evaluate Model
# =====================================

metrics, y_pred = evaluate_model(knn, X_test, y_test)

print_metrics("K-Nearest Neighbors", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "K-Nearest Neighbors")

# =====================================
# 6️⃣ Save Model
# =====================================

save_model(knn, scaler, "knn.pkl")
