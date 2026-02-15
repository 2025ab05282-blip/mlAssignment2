from sklearn.linear_model import LogisticRegression
from utility import (
    load_dataset,
    preprocess_data,
    evaluate_model,
    print_metrics,
    plot_confusion_matrix,
    print_classification_report,
    save_model
)

# Load data
X, y = load_dataset()

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)

# Train
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
metrics, y_pred = evaluate_model(model, X_test, y_test)

print_metrics("Logistic Regression", metrics)
print_classification_report(y_test, y_pred)
plot_confusion_matrix(y_test, y_pred, "Logistic Regression")

# Save
save_model(model, scaler, "logistic_regression.pkl")
