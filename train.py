from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "multiclass_dataset.csv"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["label"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(n_estimators=300, random_state=42))
])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)

metrics = {
    "accuracy": float(accuracy_score(y_test, pred)),
    "precision_macro": float(precision_score(y_test, pred, average="macro")),
    "recall_macro": float(recall_score(y_test, pred, average="macro")),
    "f1_macro": float(f1_score(y_test, pred, average="macro")),
}

joblib.dump(pipe, MODELS / "multiclass_model.joblib")
joblib.dump(metrics, MODELS / "metrics.joblib")
joblib.dump({
    "X_test": X_test,
    "y_test": y_test,
    "pred": pred,
    "proba": proba,
    "feature_names": X.columns.tolist(),
    "importances": pipe.named_steps["model"].feature_importances_.tolist()
}, MODELS / "evaluation_bundle.joblib")

print("Training complete.")
