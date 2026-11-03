from datetime import datetime
from zoneinfo import ZoneInfo
import joblib
import pandas as pd
import requests

TIMEZONE = ZoneInfo("America/Bogota")

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
    "current": [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "precipitation",
        "cloud_cover",
        "wind_direction_10m",
        "surface_pressure",
        "wind_speed_10m",
    ],
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

# Calcular la hora actual con la zona de Valledupar, no con la zona del servidor.
ahora = datetime.now(TIMEZONE)
hora_actual = ahora.replace(
    minute=0, second=0, microsecond=0, tzinfo=None
)
df["es_hora_actual"] = df["time"] == hora_actual

# Usa las condiciones actuales de Open-Meteo para la fila actual.
datos_actuales = data.get("current", {})
if datos_actuales.get("time"):
  tiempo_actual = pd.to_datetime(datos_actuales["time"]).to_pydatetime()
  tiempo_actual = tiempo_actual.replace(tzinfo=None)
  if tiempo_actual == hora_actual:
    for col in features[:8]:
      if col in datos_actuales:
        df.loc[df["es_hora_actual"], col] = datos_actuales[col]

for col in [
    "precipitation",
    "relative_humidity_2m",
    "surface_pressure",
    "dew_point_2m",
]:
  df[f"{col}_lag1"] = df[col].shift(1)
  df[f"{col}_lag2"] = df[col].shift(2)

df_hoy = df[
    (df["time"].dt.date == hora_actual.date())
    & (df["time"] >= hora_actual)
].copy()
df_hoy = df_hoy.dropna(subset=features)

if not df_hoy.empty:
  X_live = df_hoy[features]
  df_hoy["probabilidad_lluvia"] = model.predict_proba(X_live)[:, 1] * 100

  print(
      f"\n⛈️ Condiciones actuales ({ahora.strftime('%Y-%m-%d %H:%M')}) "
      "y pronóstico restante del día:"
  )
  print(
      df_hoy[[
          "time",
          "probabilidad_lluvia",
          "temperature_2m",
          "relative_humidity_2m",
          "precipitation",
      ]].to_string(index=False)
  )
else:
  print("⚠️ No se encontraron datos para la hora actual o las horas restantes.")