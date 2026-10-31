from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DATOS_DIR = BASE_DIR / "data" / "raw" / "open_meteo_ifs"
REPORTE_PATH = BASE_DIR / "data" / "reporte_calidad.csv"

COLUMNAS_BASE = [
    "time",
    "precipitation",
    "temperature_2m",
    "dew_point_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
    "cloud_cover",
]

def revisar(anio: int):
    ruta = DATOS_DIR / f"{anio}.csv"
    if not ruta.exists():
        raise FileNotFoundError(f"No existe el archivo para el año {anio}: {ruta}")

    df = pd.read_csv(ruta)
    columnas_faltantes = [col for col in COLUMNAS_BASE if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en {anio}: {columnas_faltantes}")

    df_original = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    duplicadas = int(df["time"].duplicated().sum())
    df = df.dropna(subset=["time"]).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    horas_esperadas = pd.date_range(f"{anio}-01-01 00:00:00", f"{anio}-12-31 23:00:00", freq="h")
    horas_presentes = pd.DatetimeIndex(df["time"])
    faltantes = len(horas_esperadas.difference(horas_presentes))

    resultado = {
        "anio": anio,
        "filas": len(df_original),
        "filas_validas": len(df),
        "esperadas": len(horas_esperadas),
        "duplicadas": duplicadas,
        "faltantes": faltantes,
        "nulos": int(df_original.isna().sum().sum()),
    }

    # Calcula min y max dinámicamente de todas las variables base (excepto time)
    for columna in COLUMNAS_BASE[1:]:
        resultado[f"{columna}_min"] = df[columna].min()
        resultado[f"{columna}_max"] = df[columna].max()

    return resultado

if __name__ == "__main__":
    filas = []

    for anio in range(2017, 2026):
        try:
            filas.append(revisar(anio))
        except (FileNotFoundError, ValueError) as exc:
            print(f"Advertencia para {anio}: {exc}")

    if not filas:
        raise RuntimeError("No se encontraron archivos válidos para revisar.")

    tabla = pd.DataFrame(filas)
    pd.set_option("display.max_columns", None)
    print(tabla.T)

    REPORTE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(REPORTE_PATH, index=False)
    print(f"\nReporte guardado en: {REPORTE_PATH}")