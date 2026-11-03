from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
import xgboost as xgb

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean" / "dataset_ml_horario.csv"

# 1. Cargar el dataset maestro
print("⏳ Cargando dataset maestro para ML...")
df = pd.read_csv(DATA_PATH, parse_dates=["time"])

# 2. Ingeniería de Características (Features)
print("⚙️ Creando variables predictoras (lags y tiempo)...")
df["hora"] = df["time"].dt.hour
df["mes"] = df["time"].dt.month

# Creamos lags (variables de horas anteriores) para que la atmósfera tenga memoria
for col in ["precipitation", "relative_humidity_2m", "surface_pressure", "dew_point_2m"]:
    df[f"{col}_lag1"] = df[col].shift(1)
    df[f"{col}_lag2"] = df[col].shift(2)

# Limpiamos los nulos generados por los lags
df = df.dropna().reset_index(drop=True)

# Definir características (X) y objetivo (y)
features = [
    "precipitation", "dew_point_2m", "cloud_cover", "wind_direction_10m",
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
    "hora", "mes",
    "precipitation_lag1", "relative_humidity_2m_lag1", "surface_pressure_lag1",
    "precipitation_lag2", "relative_humidity_2m_lag2", "surface_pressure_lag2"
]

X = df[features]
y = df["target_lluvia_binaria"]

# 3. División temporal
# 2016-2021 entrena, 2022 calibra las probabilidades y 2023-2024 prueba.
# La calibración evita que el desbalance (solo unas pocas horas con lluvia)
# haga que las probabilidades sean artificialmente tacañas.
train_mask = df["time"] < "2022-01-01"
calibration_mask = (df["time"] >= "2022-01-01") & (df["time"] < "2023-01-01")
test_mask = df["time"] >= "2023-01-01"
X_train, X_calibration, X_test = (
    X[train_mask],
    X[calibration_mask],
    X[test_mask],
)
y_train, y_calibration, y_test = (
    y[train_mask],
    y[calibration_mask],
    y[test_mask],
)

print(
    f"📊 Entrenamiento: {len(X_train)} horas | "
    f"Calibración: {len(X_calibration)} horas | "
    f"Prueba general: {len(X_test)} horas"
)

# ==========================================
# MODELO 1: XGBOOST CLASSIFIER
# ==========================================
print("\n🚀 Entrenando XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    eval_metric="logloss"
)
xgb_model.fit(X_train, y_train)

calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(xgb_model),
    method="sigmoid",
)
calibrated_model.fit(X_calibration, y_calibration)

preds_xgb = calibrated_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, preds_xgb)
print(f"🎯 AUC XGBoost calibrado (Test General): {auc_xgb:.4f}")

# ==========================================
# MODELO 2: REGRESIÓN LOGÍSTICA
# ==========================================
print("\n🚀 Entrenando Regresión Logística...")
lr_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
lr_model.fit(X_train, y_train)

preds_lr = lr_model.predict_proba(X_test)[:, 1]
auc_lr = roc_auc_score(y_test, preds_lr)
print(f"🎯 AUC Regresión Logística (Test General): {auc_lr:.4f}")

# ==========================================
# PRUEBA DE FUEGO: EL DÍA DEL DILUVIO
# ==========================================
# Reemplaza 'YYYY-MM-DD' con la fecha exacta del diluvio que mencionas (ej: "2026-10-25" o la fecha que corresponda)
# Nota: Si el diluvio ocurrió en un día reciente que está en el dataset, lo filtramos:
fecha_diluvio = "2022-06-07"  # <--- Cambia esto por la fecha real del diluvio si está en el dataset histórico
print(f"\n⛈️ Evaluando el día de prueba específico: {fecha_diluvio}")

df_diluvio = df[df["time"].dt.date == pd.to_datetime(fecha_diluvio).date()].copy()

if len(df_diluvio) > 0:
    X_d = df_diluvio[features]
    df_diluvio["prob_xgb"] = calibrated_model.predict_proba(X_d)[:, 1]
    df_diluvio["prob_lr"] = lr_model.predict_proba(X_d)[:, 1]
    
    print(df_diluvio[["time", "lluvia_ideam", "target_lluvia_binaria", "prob_xgb", "prob_lr"]].to_string(index=False))
else:
    print(f"⚠️ La fecha {fecha_diluvio} no está en el rango del dataset analizado (2016-2024). Ajusta 'fecha_diluvio' a una fecha dentro de ese rango para ver la hora del diluvio.")


# Guardar el modelo calibrado, que conserva la interfaz predict_proba.
joblib.dump(calibrated_model, "modelo_xgb_valledupar.pkl")
print("💾 ¡Modelo XGBoost calibrado guardado exitosamente!")