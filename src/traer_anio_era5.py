import os
import time
import requests
import pandas as pd

# 1. Cargar el archivo limpio del IDEAM para extraer el rango exacto de fechas
ideam_path = "data/clean/precipitacion_limpia.csv"
if not os.path.exists(ideam_path):
    raise FileNotFoundError(f"No se encontró {ideam_path}. ¡Asegúrate de correr primero el script de limpieza!")

df_ideam = pd.read_csv(ideam_path, parse_dates=["Fecha"], index_col="Fecha")

fecha_min = df_ideam.index.min()
fecha_max = df_ideam.index.max()

print(f"📅 Rango exacto detectado en el IDEAM:")
print(f"   Desde: {fecha_min}")
print(f"   Hasta: {fecha_max}")

URL = "https://archive-api.open-meteo.com/v1/archive"
CARPETA = "data/raw/open_meteo_era5"
VARIABLES = [
    "precipitation", "dew_point_2m", "cloud_cover", "wind_direction_10m",
    "temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m",
]

def traer_rango_fechas(start_date, end_date):
    params = {
        "latitude": 10.44,
        "longitude": -73.25,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "models": "",
        "hourly": ",".join(VARIABLES),
        "timezone": "America/Bogota",
    }
    respuesta = requests.get(URL, params=params, timeout=60)
    if respuesta.status_code != 200:
        print(f"⚠️ Error {respuesta.status_code}: {respuesta.text}")
    respuesta.raise_for_status()
    return pd.DataFrame(respuesta.json()["hourly"])

os.makedirs(CARPETA, exist_ok=True)

# 2. Descargar año por año respetando estrictamente los límites del IDEAM
anos = range(fecha_min.year, fecha_max.year + 1)

for anio in anos:
    ruta = f"{CARPETA}/{anio}.csv"
    if os.path.exists(ruta):
        print(f"⏭️ El año {anio} ya existe en ERA5, se omite.")
        continue
    
    # Acotar las fechas para que no pidan días fuera del rango real del IDEAM
    inicio_anio = max(pd.Timestamp(f"{anio}-01-01"), fecha_min)
    fin_anio = min(pd.Timestamp(f"{anio}-12-31"), fecha_max)
    
    print(f"📥 Descargando ERA5 ({inicio_anio.date()} al {fin_anio.date()})...")
    df_era5 = traer_rango_fechas(inicio_anio, fin_anio)
    df_era5.to_csv(ruta, index=False)
    print(f"✅ Guardado: {ruta} con {len(df_era5)} registros horarias.")
    time.sleep(1)

print("\n🎉 ¡Descarga de ERA5 completada y perfectamente sincronizada con el IDEAM!")