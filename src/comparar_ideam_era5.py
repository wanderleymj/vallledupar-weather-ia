from datetime import timedelta
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parents[1]
IDEAM_PATH = BASE_DIR / "data" / "clean" / "precipitacion_limpia.csv"
ERA5_DIR = BASE_DIR / "data" / "raw" / "open_meteo_era5"   # Carpeta correcta de ERA5

UMBRAL_IDEAM = 1.0   # mm; umbral base para considerar que llovió en el IDEAM

# 1. IDEAM
ideam = pd.read_csv(IDEAM_PATH)
ideam["fecha"] = pd.to_datetime(ideam["Fecha"]).dt.date
ideam = ideam[ideam["tiene_dato"] == True][["fecha", "Valor"]]
ideam = ideam.rename(columns={"Valor": "lluvia_ideam"})
ideam["ideam_si"] = ideam["lluvia_ideam"] >= UMBRAL_IDEAM
print("IDEAM:", ideam["fecha"].min(), "->", ideam["fecha"].max(), "|", len(ideam), "días")

# 2. ERA5: lluvia total por día (desde 2016 hasta 2024)
partes = []
for anio in range(2016, 2025):
    archivo = ERA5_DIR / f"{anio}.csv"
    if archivo.exists():
        df = pd.read_csv(archivo)
        df["fecha"] = pd.to_datetime(df["time"]).dt.date
        partes.append(df.groupby("fecha")["precipitation"].sum())

ifs = pd.concat(partes).rename("lluvia_ifs").reset_index()

# 3. Unir y comparar cantidades
m = pd.merge(ideam, ifs, on="fecha")
print("Días comparados:", len(m))
print("Total IDEAM:", round(m["lluvia_ideam"].sum()), "mm | ERA5:", round(m["lluvia_ifs"].sum()), "mm")
print("Correlación diaria:", round(m["lluvia_ideam"].corr(m["lluvia_ifs"]), 2))
print("Días con lluvia según el IDEAM:", round(m["ideam_si"].mean(), 2))

# 4. ¿Qué umbral le sirve a ERA5? (Como ERA5 suaviza la lluvia, a veces un umbral menor a 1mm funciona mejor)
print("\n--- Evaluación de Umbrales para ERA5 ---")
for u in [0.1, 0.5, 1, 2, 3, 5, 8]:
    ifs_si = m["lluvia_ifs"] >= u
    aciertos_lluvia = (ifs_si & m["ideam_si"]).sum()
    print("Umbral", u, "mm | Días con lluvia (ERA5):", round(ifs_si.mean(), 2),
          "| Detectadas:", round(aciertos_lluvia / max(m["ideam_si"].sum(), 1), 2),
          "| Precisión:", round(aciertos_lluvia / max(ifs_si.sum(), 1), 2),
          "| Aciertos globales:", round((ifs_si == m["ideam_si"]).mean(), 2))

# 5. ¿Están desfasados los días?
print("\n--- Análisis de Desfase Temporal ---")
for d in [-1, 0, 1]:
    i = ideam.copy()
    i["fecha"] = i["fecha"].apply(lambda f: f + timedelta(days=d))
    mm = pd.merge(i, ifs, on="fecha")
    print("Desfase", d, "| Correlación:", round(mm["lluvia_ideam"].corr(mm["lluvia_ifs"]), 2))

# 6. AUC (Capacidad predictiva)
m["fecha"] = pd.to_datetime(m["fecha"])
prueba = m[m["fecha"] >= "2023-01-01"]
if len(prueba) > 0:
    print("\n--- Poder Predictivo ---")
    print("AUC con la lluvia de ERA5 como predictor (2023-2025):",
          round(roc_auc_score(prueba["ideam_si"], prueba["lluvia_ifs"]), 3))