import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

#load the prepared data from session 2
df = pd.read_csv("attrition_prepared.csv")

# Split into features (X) and target (y)
X = df.drop(columns=["Attrition"])
y = df["Attrition"]

#split into train and test sets -80/20, with aa fixed random_state for for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
   X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale all features to the same numeric range
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train the logistic regression model
model =  LogisticRegression(max_iter=2000, class_weight={0: 1, 1: 2})
model.fit(X_train_scaled, y_train)

#Evaluate on the held-out test set
y_pred = model.predict(X_test_scaled)

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Stayed", "Left"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# --- NEW: print the model's coefficients, ranked by influence ---
coefficients = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
})
coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
coefficients = coefficients.sort_values("abs_coefficient", ascending=False)

print("\nTop 15 most influential features:")
print(coefficients[["feature", "coefficient"]].head(15).to_string(index=False))


#Save the trained model AND the exact column order it expects
joblib.dump(model, "attrition_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(X.columns.tolist(), "model_columns.joblib")
print("\nmodels and column list saved")