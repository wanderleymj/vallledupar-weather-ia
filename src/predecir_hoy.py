from datetime import datetime
import joblib
import pandas as pd
import requests

# 1. Cargar el modelo entrenado
MODELO_PATH = "modelo_xgb_valledupar.pkl"
try:
    model = joblib.load(MODELO_PATH)
    print("✅ Modelo cargado correctamente.")
except FileNotFoundError:
    print(
        "⚠️ No se encontró el archivo del modelo. ¡Asegúrate de guardarlo primero con joblib.dump()!"
    )
    exit()

# 2. Consultar la API en vivo de Open-Meteo para Valledupar (Lat: 10.44, Lon: -73.25)
print("🌍 Consultando datos meteorológicos actuales en Valledupar...")
URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 10.44,
    "longitude": -73.25,
    "hourly": [
        "precipitation",
        "dew_point_2m",
        "cloud_cover",
        "wind_direction_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "surface_pressure",
        "wind_speed_10m",
    ],
    "timezone": "America/Bogota",
    "past_days": 1,  # Trae ayer y hoy para calcular bien los lags
}

response = requests.get(URL, params=params, timeout=30)
response.raise_for_status()
data = response.json()

# Convertir a DataFrame
df = pd.DataFrame(data["hourly"])
df["time"] = pd.to_datetime(df["time"])

# 3. Ingeniería de características (exactamente la misma del entrenamiento)
df["hora"] = df["time"].dt.hour
df["mes"] = df["time"].dt.month

for col in [
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "dew_point_2m",
]:
  df[f"{col}_lag1"] = df[col].shift(1)
  df[f"{col}_lag2"] = df[col].shift(2)

# Definir las características exactas
features = [
    "precipitation",
    "dew_point_2m",
    "cloud_cover",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "hora",
    "mes",
    "precipitation_lag1",
    "relative_humidity_2m_lag1",
    "surface_pressure_lag1",
    "precipitation_lag2",
    "relative_humidity_2m_lag2",
    "surface_pressure_lag2",
]

# Filtrar específicamente para el día de ayer (30 de octubre de 2026) para cazar el diluvio
fecha_objetivo = "2026-10-30"
df_ayer = df[df["time"].dt.date == pd.to_datetime(fecha_objetivo).date()].copy()

if not df_ayer.empty:
  X_live = df_ayer[features]
  df_ayer["probabilidad_lluvia"] = model.predict_proba(X_live)[:, 1] * 100

  print(f"\n⛈️ Pronóstico y comportamiento para el día del diluvio ({fecha_objetivo}):")
  print(
      df_ayer[[
          "time",
          "probabilidad_lluvia",
          "temperature_2m",
          "relative_humidity_2m",
          "precipitation",
      ]].to_string(index=False)
  )
else:
  print("⚠️ No se encontraron datos para esa fecha.")