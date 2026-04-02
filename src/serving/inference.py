import os
import pandas as pd
import joblib

# === BASE DIRECTORY ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === MODEL PATH ===
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "3b1a41221fc44548aed629fa42b762e0",
    "artifacts",
    "model",
    "model.pkl"
)

# === PREPROCESSOR PATH ===
PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "model",
    "3b1a41221fc44548aed629fa42b762e0",
    "artifacts",
    "preprocessing.pkl"
)

# === FEATURE COLUMNS PATH ===
FEATURE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "3b1a41221fc44548aed629fa42b762e0",
    "artifacts",
    "feature_columns.txt"
)

# === LOAD MODEL ===
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from {MODEL_PATH}")
except Exception as e:
    raise Exception(f"❌ Failed to load model: {e}")

# === LOAD PREPROCESSOR ===
try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("✅ Preprocessor loaded")
except Exception as e:
    raise Exception(f"❌ Failed to load preprocessor: {e}")

# === LOAD FEATURE COLUMNS ===
try:
    with open(FEATURE_PATH) as f:
        FEATURE_COLS = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(FEATURE_COLS)} features")
except Exception as e:
    raise Exception(f"❌ Failed to load feature columns: {e}")

# === CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# === TRANSFORMATION FUNCTION ===
def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Numeric conversion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    # Binary encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    # One-hot encoding
    obj_cols = df.select_dtypes(include=["object"]).columns
    if len(obj_cols) > 0:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # Boolean → int
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # Align features
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

# === PREDICT FUNCTION ===
def predict(input_dict: dict) -> str:
    df = pd.DataFrame([input_dict])

    # Transform
    df_enc = _serve_transform(df)

    # Predict
    try:
        pred = model.predict(df_enc)[0]
    except Exception as e:
        raise Exception(f"❌ Prediction failed: {e}")

    # Output
    return "Likely to churn" if pred == 1 else "Not likely to churn"