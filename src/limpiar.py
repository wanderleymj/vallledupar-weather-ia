import pandas as pd

# 1. Leer el CSV omitiendo los comentarios y configurando separadores y decimales
df = pd.read_csv("data/raw/ideam/precipitacion_diaria.csv", comment="#", sep=";", decimal=",")

# 2. Renombrar las columnas de AQUARIUS a los nombres cortos que usa tu script
df = df.rename(columns={
    "Sello de tiempo (UTC-05:00)": "Fecha",
    "Valor (mm)": "Valor"
})

df = df[["Fecha", "Valor"]]
df["Fecha"] = pd.to_datetime(df["Fecha"])

calendario_completo = pd.date_range(df["Fecha"].min(), df["Fecha"].max(), freq="h")

df = df.set_index("Fecha").reindex(calendario_completo)
df.index.name = "Fecha"

df["tiene_dato"] = df["Valor"].notna()

df.to_csv("data/clean/precipitacion_limpia.csv", index=True)

total = len(df)
con_dato = df["tiene_dato"].sum()

print("Dias totales = ", total)
print("Dias con dato = ", con_dato)
print("Dias sin dato = ", total - con_dato)