from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, classification_report
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

# 3. División temporal (Entrenar con pasado, probar con los últimos años)
# Usaremos datos antes de 2023 para entrenar y 2023-2024 para test general
train_mask = df["time"] < "2023-01-01"
X_train, X_test = X[train_mask], X[~train_mask]
y_train, y_test = y[train_mask], y[~train_mask]

print(f"📊 Datos de entrenamiento: {len(X_train)} horas | Datos de prueba general: {len(X_test)} horas")

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

preds_xgb = xgb_model.predict_proba(X_test)[:, 1]
auc_xgb = roc_auc_score(y_test, preds_xgb)
print(f"🎯 AUC XGBoost (Test General): {auc_xgb:.4f}")

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
    df_diluvio["prob_xgb"] = xgb_model.predict_proba(X_d)[:, 1]
    df_diluvio["prob_lr"] = lr_model.predict_proba(X_d)[:, 1]
    
    print(df_diluvio[["time", "lluvia_ideam", "target_lluvia_binaria", "prob_xgb", "prob_lr"]].to_string(index=False))
else:
    print(f"⚠️ La fecha {fecha_diluvio} no está en el rango del dataset analizado (2016-2024). Ajusta 'fecha_diluvio' a una fecha dentro de ese rango para ver la hora del diluvio.")


# Guardar el modelo XGBoost entrenado
joblib.dump(xgb_model, "modelo_xgb_valledupar.pkl")
print("💾 ¡Modelo XGBoost guardado exitosamente como 'modelo_xgb_valledupar.pkl'!")