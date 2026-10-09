import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CSV_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

df = pd.read_csv(CSV_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop("customerID", axis=1)

df = pd.get_dummies(
    df,
    columns=[
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod",
    ],
    drop_first=True,
)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
model.fit(X_train_scaled, y_train)

print(classification_report(y_test, model.predict(X_test_scaled)))
print("ROC AUC:", round(roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]), 3))

joblib.dump(
    {"model": model, "scaler": scaler, "columns": list(X.columns)},
    "churn_model.joblib",
)
print("Saved churn_model.joblib")
