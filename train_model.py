
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "soil_dataset.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}\n"
        "Place soil_dataset.csv in the project folder."
    )

df = pd.read_csv(DATA_FILE)

NUMERIC_FEATURES = [
    "nitrogen_mg_kg",
    "phosphorus_mg_kg",
    "potassium_mg_kg",
    "ph",
    "electrical_conductivity_ds_m",
    "soil_temperature_c",
    "moisture_percent",
]

CATEGORICAL_FEATURES = [
    "crop_selected",
    "soil_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "soil_condition"

required_columns = FEATURES + [TARGET]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Dataset is missing required columns: {missing_columns}\n"
        f"Actual columns: {df.columns.tolist()}"
    )

# Only use model inputs and target.
# Other columns, if present, are not automatically included.
df = df[required_columns].copy()

# --------------------------------------------------
# 2. Validate and clean basic data
# --------------------------------------------------

for column in NUMERIC_FEATURES:
    df[column] = pd.to_numeric(df[column], errors="raise")

if df[required_columns].isnull().any().any():
    null_counts = df[required_columns].isnull().sum()
    raise ValueError(
        "Missing values found:\n"
        + null_counts[null_counts > 0].to_string()
    )

df[TARGET] = df[TARGET].astype(str).str.strip().str.title()

EXPECTED_LABELS = {"Good", "Moderate", "Bad"}

if set(df[TARGET].unique()) != EXPECTED_LABELS:
    raise ValueError(
        f"Expected labels {EXPECTED_LABELS}, "
        f"found {set(df[TARGET].unique())}"
    )

if not df[NUMERIC_FEATURES].map(
    lambda value: pd.notna(value) and abs(value) != float("inf")
).all().all():
    raise ValueError("Numeric features contain infinite values.")

for column in CATEGORICAL_FEATURES:
    df[column] = df[column].astype(str).str.strip()

if (df[CATEGORICAL_FEATURES] == "").any().any():
    raise ValueError("Empty categorical values found.")

X = df[FEATURES]
y = df[TARGET]

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())

# --------------------------------------------------
# 3. Split data into training and test sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        CATEGORICAL_FEATURES,
    ),
    ("numbers", "passthrough", NUMERIC_FEATURES),
])

# --------------------------------------------------
# 5. Compare candidate models
# --------------------------------------------------

candidates = {
    "Dummy baseline": DummyClassifier(
        strategy="most_frequent"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

cv_results = {}

for name, classifier in candidates.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])

    scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
    )

    cv_results[name] = scores.mean()

    print(
        f"\n{name}: mean CV macro F1 = "
        f"{scores.mean():.4f} "
        f"+/- {scores.std():.4f}"
    )

# Choose the best actual classifier, not the dummy baseline.
eligible_results = {
    name: score
    for name, score in cv_results.items()
    if name != "Dummy baseline"
}

best_name = max(eligible_results, key=eligible_results.get)
print("\nSelected model:", best_name)

# --------------------------------------------------
# 6. Train selected model
# --------------------------------------------------

final_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", candidates[best_name]),
])

final_model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Evaluate once on the held-out test set
# --------------------------------------------------

predictions = final_model.predict(X_test)

print("\n========== TEST RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y_test, predictions), 4),
)
print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro"), 4),
)

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

label_order = ["Bad", "Good", "Moderate"]

print("\nConfusion matrix (Bad, Good, Moderate):")
print(confusion_matrix(y_test, predictions, labels=label_order))

# --------------------------------------------------
# 8. Save model and metadata
# --------------------------------------------------

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

joblib.dump(
    final_model,
    ARTIFACT_DIR / "soil_condition_model.joblib",
)

joblib.dump(
    {
        "model_name": best_name,
        "version": "2.0-prototype",
        "features": FEATURES,
        "classes": label_order,
        "label_type": "provisional",
    },
    ARTIFACT_DIR / "model_metadata.joblib",
)

print("\nTraining completed.")
print("Saved model:", ARTIFACT_DIR / "soil_condition_model.joblib")
print("Saved metadata:", ARTIFACT_DIR / "model_metadata.joblib")
