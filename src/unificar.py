import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
IDEAM_PATH = BASE_DIR / "data" / "clean" / "precipitacion_limpia.csv"
ERA5_DIR = BASE_DIR / "data" / "raw" / "open_meteo_era5"

# 1. Cargar el IDEAM horario
print("⏳ Cargando datos horarios del IDEAM...")
df_ideam = pd.read_csv(IDEAM_PATH, parse_dates=["Fecha"])
# Renombramos para estandarizar la llave de tiempo con ERA5 ('time')
df_ideam = df_ideam.rename(columns={"Fecha": "time", "Valor": "lluvia_ideam"})
df_ideam["time"] = pd.to_datetime(df_ideam["time"])

# 2. Cargar y concatenar todos los años de ERA5
print("⏳ Cargando y uniendo los años de ERA5...")
partes_era5 = []
for anio in range(2016, 2025):
    archivo = ERA5_DIR / f"{anio}.csv"
    if archivo.exists():
        df_ano = pd.read_csv(archivo, parse_dates=["time"])
        partes_era5.append(df_ano)

df_era5 = pd.concat(partes_era5, ignore_index=True)

# 3. Hacer el merge por hora exacta
print("🔗 Cruzando ERA5 con el IDEAM por marca de tiempo...")
df_final = pd.merge(df_era5, df_ideam, on="time", how="inner")

# Creamos una columna binaria o continua según prefieras para el objetivo
# Por ejemplo, definimos si llovió (>= 0.1 mm) o mantenemos el valor continuo
df_final["target_lluvia_binaria"] = (df_final["lluvia_ideam"] >= 0.1).astype(int)

# 4. Guardar el dataset maestro para Machine Learning
OUTPUT_PATH = BASE_DIR / "data" / "clean" / "dataset_ml_horario.csv"
os.makedirs(OUTPUT_PATH.parent, exist_ok=True)
df_final.to_csv(OUTPUT_PATH, index=False)

print(f"\n🎉 ¡Dataset maestro listo!")
print(f"📁 Guardado en: {OUTPUT_PATH}")
print(f"📊 Dimensiones totales: {df_final.shape[0]} horas y {df_final.shape[1]} columnas.")
print(df_final.head(3))