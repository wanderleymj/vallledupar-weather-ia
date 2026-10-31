from datetime import datetime
from fastapi import FastAPI, HTTPException
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

# Cargar el modelo
MODELO_PATH = "modelo_xgb_valledupar.pkl"
try:
  model = joblib.load(MODELO_PATH)
  print("✅ Modelo cargado exitosamente en la API.")
except Exception as e:
  print(f"⚠️ Error al cargar el modelo: {e}")
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
                    const response = await fetch('/predecir');
                    if (!response.ok) throw new Error(`Error HTTP: ${response.status}`);
                    
                    const data = await response.json();
                    
                    loading.style.display = 'none';
                    gridContainer.classList.remove('hidden');

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
    response.raise_for_status()
    data = response.json()
  except Exception as e:
    raise HTTPException(
        status_code=502,
        detail=f"Error al conectar con Open-Meteo: {str(e)}",
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