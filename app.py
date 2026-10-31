from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import requests

app = FastAPI(title="Valledupar Rain Debug")

# Intentar cargar el modelo con un bloque seguro
MODELO_PATH = "modelo_xgb_valledupar.pkl"
try:
  model = joblib.load(MODELO_PATH)
  print("✅ Modelo cargado exitosamente.")
except Exception as e:
  print(f"⚠️ Error cargando modelo: {e}")
  model = None

FEATURES = [
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


@app.get("/", response_class=HTMLResponse)
def home():
  return "<h1>Servidor activo. Prueba /test-api o /predecir</h1>"


# 1. Ruta para probar solo si internet y Open-Meteo funcionan en Render
@app.get("/test-api")
def test_api():
  try:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 10.44,
        "longitude": -73.25,
        "hourly": ["temperature_2m"],
        "timezone": "America/Bogota",
        "past_days": 1,
    }
    r = requests.get(url, params=params, timeout=10)
    return {"status": "API OK", "data_sample": r.json()["hourly"]["time"][:3]}
  except Exception as e:
    return {"status": "Error", "detalle": str(e)}


@app.get("/predecir")
def predecir_clima():
  if model is None:
    raise HTTPException(
        status_code=500, detail="El modelo no está cargado en el servidor."
    )

  url = "https://api.open-meteo.com/v1/forecast"
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
      "past_days": 1,
  }

  headers = {
      "User-Agent": (
          "ValleduparWeatherApp-StudentProject/1.0"
          " (contacto@valledupar-ai.com)"
      )
  }

  try:
    response = requests.get(url, params=params, headers=headers, timeout=30)
    data = response.json()

    # Validar si Open-Meteo devolvió un error
    if "hourly" not in data:
      print(f"⚠️ Respuesta extraña de Open-Meteo: {data}")
      raise HTTPException(
          status_code=502,
          detail=f"Open-Meteo no devolvió datos horarios: {data}",
      )

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

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

    hoy_str = datetime.now().strftime("%Y-%m-%d")
    df["solo_fecha"] = df["time"].dt.strftime("%Y-%m-%d")
    df_hoy = df[df["solo_fecha"] == hoy_str].copy()

    if df_hoy.empty:
      raise HTTPException(
          status_code=404, detail="No se encontraron datos para el día actual."
      )

    X_live = df_hoy[FEATURES]
    df_hoy["probabilidad_lluvia"] = model.predict_proba(X_live)[:, 1] * 100

    resultado = []
    for _, row in df_hoy.iterrows():
      resultado.append({
          "tiempo": str(row["time"]),
          "probabilidad_lluvia_porcentaje": round(
              row["probabilidad_lluvia"], 2
          ),  # type: ignore
          "temperatura_celsius": row["temperature_2m"],
          "humedad_relativa": row["relative_humidity_2m"],
          "nubosidad": row["cloud_cover"],
      })

    return {"ciudad": "Valledupar", "pronostico_horario": resultado}

  except Exception as e:
    print(f"❌ Error crítico en predecir_clima: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))