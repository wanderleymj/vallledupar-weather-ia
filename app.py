from datetime import datetime
from pathlib import Path
import time
from zoneinfo import ZoneInfo
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
import requests

app = FastAPI(
    title="Valledupar Rain Predictor API",
    description=(
        "API de Machine Learning con XGBoost para predecir lluvia en tiempo"
        " real."
    ),
)

TIMEZONE = ZoneInfo("America/Bogota")
BASE_DIR = Path(__file__).resolve().parent
OPEN_METEO_CACHE_SECONDS = 300
_open_meteo_cache = {"expires_at": 0.0, "data": None}
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

# Cargar el modelo
MODELO_PATH = BASE_DIR / "modelo_xgb_valledupar.pkl"
try:
  model = joblib.load(MODELO_PATH)
  print("✅ Modelo cargado exitosamente en la API.")
except Exception as e:
  print(f"⚠️ Error al cargar el modelo: {e}")
  model = None

@app.get("/", response_class=HTMLResponse)
def home():
  html_content = """
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Radar de Lluvia - Valledupar (XGBoost)</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-blue-950 text-white min-h-screen p-4 md:p-8">
        <div class="max-w-4xl mx-auto">
            <header class="text-center mb-8">
                <span class="bg-blue-600/30 text-blue-400 text-xs font-semibold px-3 py-1 rounded-full uppercase tracking-wider border border-blue-500/30">ML Powered Forecast</span>
                <h1 class="text-3xl md:text-5xl font-extrabold mt-3 tracking-tight">🌧️ Valledupar Weather AI</h1>
                <p class="text-slate-400 mt-2 text-sm md:text-base">Predicción de probabilidad de lluvia horaria basada en Machine Learning (XGBoost + Open-Meteo)</p>
            </header>

            <!-- Pantalla de Carga -->
            <div id="loading" class="text-center py-16">
                <div class="inline-block animate-spin rounded-full h-12 w-12 border-4 border-blue-500 border-t-transparent"></div>
                <p class="mt-4 text-slate-300 font-medium">Consultando la atmósfera en tiempo real...</p>
            </div>

            <!-- Contenedor de Error -->
            <div id="error-container" class="hidden text-center py-12 bg-red-950/30 border border-red-500/30 rounded-2xl p-6">
                <p class="text-red-400 font-bold text-lg">⚠️ ¡Ups! Hubo un problema al consultar el pronóstico.</p>
                <p id="error-details" class="text-slate-400 text-sm mt-2 font-mono"></p>
                <button onclick="cargarPronostico()" class="mt-5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-semibold px-5 py-2.5 rounded-xl transition shadow-lg">
                    🔄 Reintentar
                </button>
            </div>

            <!-- Cuadrícula de Pronóstico -->
            <div id="grid-container" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 hidden"></div>
        </div>

        <script>
            async function cargarPronostico() {
                const loading = document.getElementById('loading');
                const errorContainer = document.getElementById('error-container');
                const gridContainer = document.getElementById('grid-container');
                
                loading.style.display = 'block';
                errorContainer.classList.add('hidden');
                gridContainer.classList.add('hidden');
                gridContainer.innerHTML = '';

                try {
                    const response = await fetch('/predecir', { cache: 'no-store' });
                    if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
                    
                    const data = await response.json();
                    
                    loading.style.display = 'none';
                    gridContainer.classList.remove('hidden');

                    const actual = data.actual;
                    const actualCard = `
                        <div class="md:col-span-2 lg:col-span-3 bg-emerald-950/40 border border-emerald-500/40 rounded-2xl p-5 shadow-xl">
                            <div class="flex flex-wrap justify-between gap-2 items-center">
                                <div>
                                    <p class="text-emerald-300 text-xs uppercase tracking-wider font-semibold">Condiciones actuales</p>
                                    <p class="text-slate-200 text-sm mt-1">Consulta: ${data.fecha_consulta} ${data.hora_consulta}</p>
                                    <p class="text-slate-400 text-xs mt-1">Datos de Open-Meteo: ${data.datos_open_meteo_hora}</p>
                                </div>
                                <span class="text-4xl font-black text-white">${actual.probabilidad_lluvia_porcentaje}%</span>
                            </div>
                            <div class="mt-4 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm text-slate-300">
                                <span>🌡️ <strong>${actual.temperatura_celsius}°C</strong></span>
                                <span>💧 Humedad: <strong>${actual.humedad_relativa}%</strong></span>
                                <span>☁️ Nubosidad: <strong>${actual.nubosidad}%</strong></span>
                                <span>🌧️ Precipitación: <strong>${actual.precipitacion_mm} mm</strong></span>
                            </div>
                        </div>
                    `;
                    gridContainer.innerHTML = actualCard;

                    data.pronostico_horario.forEach(item => {
                        let prob = item.probabilidad_lluvia_porcentaje;
                        let bgColor = "bg-slate-800/60 border-slate-700/50";
                        let badgeColor = "bg-blue-500/20 text-blue-300 border-blue-500/30";
                        let alertaTexto = "Normal";

                        if (prob > 5.0) {
                            bgColor = "bg-amber-950/40 border-amber-500/40";
                            badgeColor = "bg-amber-500/20 text-amber-300 border-amber-500/30";
                            alertaTexto = "⚠️ Alta Probabilidad";
                        } else if (prob > 2.5) {
                            bgColor = "bg-blue-900/40 border-blue-500/40";
                            badgeColor = "bg-blue-500/20 text-blue-300 border-blue-500/30";
                            alertaTexto = "⚡ Posible Lluvia";
                        }

                        const card = `
                            <div class="${bgColor} backdrop-blur-md border rounded-2xl p-5 shadow-xl transition hover:scale-102">
                                <div class="flex justify-between items-center mb-3">
                                    <span class="text-xs font-mono text-slate-400">${item.tiempo}</span>
                                    <span class="text-xs px-2.5 py-0.5 rounded-full border ${badgeColor} font-medium">${alertaTexto}</span>
                                </div>
                                <div class="flex items-baseline justify-between mt-2">
                                    <div>
                                        <span class="text-3xl font-black text-white">${prob}%</span>
                                        <span class="text-xs text-slate-400 block mt-0.5">Prob. Lluvia</span>
                                    </div>
                                    <div class="text-right">
                                        <span class="text-xl font-bold text-slate-200">${item.temperatura_celsius}°C</span>
                                        <span class="text-xs text-slate-400 block">Temp.</span>
                                    </div>
                                </div>
                                <div class="mt-4 pt-3 border-t border-slate-700/40 flex justify-between text-xs text-slate-400">
                                    <span>💧 Humedad: <strong>${item.humedad_relativa}%</strong></span>
                                    <span>☁️ Nubosidad: <strong>${item.nubosidad}%</strong></span>
                                </div>
                            </div>
                        `;
                        gridContainer.innerHTML += card;
                    });
                } catch (error) {
                    loading.style.display = 'none';
                    errorContainer.classList.remove('hidden');
                    document.getElementById('error-details').innerText = error.message;
                }
            }
            cargarPronostico();
        </script>
    </body>
    </html>
    """
  return html_content


@app.head("/")
def health_check():
  return Response(status_code=200)


@app.get("/predecir")
def predecir_clima(response: Response):
  response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
  response.headers["Pragma"] = "no-cache"
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
      "forecast_days": 1,
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

  headers = {
      "User-Agent": (
          "ValleduparWeatherApp-StudentProject/1.0"
          " (contacto@valledupar-ai.com)"
      )
  }

  now_timestamp = time.monotonic()
  cached_data = _open_meteo_cache["data"]
  if cached_data is not None and now_timestamp < _open_meteo_cache["expires_at"]:
    data = cached_data
  else:
    try:
      response = requests.get(url, params=params, headers=headers, timeout=30)
      if response.status_code == 429:
        if cached_data is not None:
          data = cached_data
        else:
          raise HTTPException(
              status_code=503,
              detail="Open-Meteo alcanzó temporalmente su límite de consultas. Intenta de nuevo en unos minutos.",
          )
      else:
        response.raise_for_status()
        data = response.json()
        _open_meteo_cache["data"] = data
        _open_meteo_cache["expires_at"] = (
            now_timestamp + OPEN_METEO_CACHE_SECONDS
        )
    except HTTPException:
      raise
    except requests.RequestException as error:
      raise HTTPException(
          status_code=502,
          detail=f"Error al conectar con Open-Meteo: {error}",
      ) from error
    except ValueError as error:
      raise HTTPException(
          status_code=502,
          detail=f"Open-Meteo devolvió una respuesta inválida: {error}",
      ) from error

  if data is None:
    raise HTTPException(
        status_code=503,
        detail="No hay datos meteorológicos disponibles temporalmente.",
    )

  df = pd.DataFrame(data["hourly"])
  df["time"] = pd.to_datetime(df["time"])

  df["hora"] = df["time"].dt.hour
  df["mes"] = df["time"].dt.month

  now = datetime.now(TIMEZONE)
  current_hour = now.replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)

  # Las dos horas anteriores son necesarias para los lags del modelo.
  df["es_hora_actual"] = df["time"] == current_hour
  current_rows = df[df["es_hora_actual"]].copy()
  if current_rows.empty:
    raise HTTPException(
        status_code=404,
        detail=f"Open-Meteo no devolvió datos para la hora actual ({current_hour}).",
    )

  # Para la hora actual usamos también el bloque `current` de Open-Meteo,
  # en vez de mostrar una hora de pronóstico como si fuera una observación.
  current_data = data.get("current", {})
  if current_data.get("time"):
    current_time = pd.to_datetime(current_data["time"]).to_pydatetime()
    current_time = current_time.replace(tzinfo=None)
    if current_time == current_hour:
      for column in FEATURES[:8]:
        if column in current_data:
          df.loc[df["es_hora_actual"], column] = current_data[column]

  for column in [
      "precipitation",
      "relative_humidity_2m",
      "surface_pressure",
      "dew_point_2m",
  ]:
    df[f"{column}_lag1"] = df[column].shift(1)
    df[f"{column}_lag2"] = df[column].shift(2)

  current_rows = df[df["es_hora_actual"]].copy()
  current_rows = current_rows.dropna(subset=FEATURES)
  if current_rows.empty:
    raise HTTPException(
        status_code=502,
        detail="No hay suficientes datos horarios anteriores para calcular la probabilidad actual.",
    )

  X_current = current_rows[FEATURES]
  current_rows["probabilidad_lluvia"] = (
      model.predict_proba(X_current)[:, 1] * 100
  )

  today = current_hour.date()
  df_future = df[
      (df["time"].dt.date == today) & (df["time"] >= current_hour)
  ].copy()
  df_future = df_future.dropna(subset=FEATURES)
  if not df_future.empty:
    X_live = df_future[FEATURES]
    df_future["probabilidad_lluvia"] = model.predict_proba(X_live)[:, 1] * 100

  def serializar_fila(row):
    return {
        "tiempo": str(row["time"]),
        "probabilidad_lluvia_porcentaje": round(
            float(row["probabilidad_lluvia"]), 2
        ),
        "temperatura_celsius": float(row["temperature_2m"]),
        "humedad_relativa": float(row["relative_humidity_2m"]),
        "nubosidad": float(row["cloud_cover"]),
        "precipitacion_mm": float(row["precipitation"]),
    }

  actual = serializar_fila(current_rows.iloc[0])
  resultado = [serializar_fila(row) for _, row in df_future.iterrows()]
  datos_open_meteo_hora = actual["tiempo"]

  return {
      "ciudad": "Valledupar",
      "fecha_consulta": now.strftime("%Y-%m-%d"),
      "hora_consulta": now.strftime("%H:%M"),
      "datos_open_meteo_hora": datos_open_meteo_hora,
      "actual": actual,
      "pronostico_horario": resultado,
  }