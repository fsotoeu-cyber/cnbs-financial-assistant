# ============================================================
# CNBS → ETL → QUALITY GATE → DELTA CHECK → PRODUCCIÓN
# Compatible 100% con el esquema de app.py
# ============================================================
import os
from pathlib import Path
import pandas as pd

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================
URL_CNBS = (
    "https://datos.cnbs.gob.hn/datastore/dump/"
    "509e19c4-09d1-4f3d-9ec4-f7a6e874bb78"
)
ARCHIVO_SALIDA = "indicadores_financieros_CNBS.csv"

TIPOS_OK = [
    "BANCOS COMERCIALES",
    "BANCOS ESTATALES",
    "SOCIEDADES FINANCIERAS",
]

BANCOS_CLAVE = [
    "BANCATLAN",
    "BAC CREDOMATIC",
    "FICOHSA",
    "BANPAIS",
    "BANCOCCI",
    "LAFISE",
    "AZTECA",
    "FICENSA",
    "BANHCAFE",
]


# ============================================================
# 2. EXTRACCIÓN
# ============================================================
def extraer_datos():
    print("=" * 70)
    print("1) EXTRACCIÓN — CNBS")
    print("=" * 70)
    df_raw = pd.read_csv(URL_CNBS)
    print(f"Filas descargadas : {len(df_raw):,}")
    print(f"Columnas          : {len(df_raw.columns)}")
    print(f"Columnas detectadas: {list(df_raw.columns)}")
    return df_raw


# ============================================================
# 3. VALIDACIÓN DE ESTRUCTURA DE FUENTE
# ============================================================
def validar_estructura_fuente(df_raw):
    print("\n" + "=" * 70)
    print("2) VALIDACIÓN DE ESTRUCTURA DE FUENTE")
    print("=" * 70)
    columnas_requeridas = [
        "TipoInstitucion",
        "Logo",
        "FechaReporte",
        "Indicador",
        "TipoIndicador",
        "Saldo",
    ]
    faltantes = [c for c in columnas_requeridas if c not in df_raw.columns]
    if faltantes:
        raise ValueError(
            f"Quality Gate FAILED — columnas faltantes en origen: {faltantes}"
        )
    print("OK — Estructura CNBS compatible.")


# ============================================================
# 4. NORMALIZACIÓN Y TRANSFORMACIÓN
# ============================================================
def normalizar_datos(df_raw):
    print("\n" + "=" * 70)
    print("3) NORMALIZACIÓN Y TRANSFORMACIÓN")
    print("=" * 70)
    df = df_raw.copy()

    for col in ["TipoInstitucion", "Logo", "Indicador", "TipoIndicador"]:
        df[col] = df[col].astype("string").str.strip()

    df = df[df["TipoInstitucion"].isin(TIPOS_OK)]

    df["FechaReporte"] = pd.to_datetime(df["FechaReporte"], errors="coerce")
    df["FechaReporte"] = df["FechaReporte"].dt.to_period("M").dt.to_timestamp("M")
    df["Saldo"] = pd.to_numeric(df["Saldo"], errors="coerce")

    df = df.rename(
        columns={
            "Logo": "Banco",
            "TipoIndicador": "CategoriaIndicador",
        }
    )

    df = df.dropna(
        subset=["Banco", "FechaReporte", "Indicador", "TipoInstitucion"]
    )

    df = df.drop_duplicates(
        subset=["Banco", "FechaReporte", "Indicador"], keep="last"
    )

    df = df.sort_values(
        ["FechaReporte", "TipoInstitucion", "Banco", "Indicador"]
    ).reset_index(drop=True)

    columnas_app = [
        "Banco",
        "FechaReporte",
        "Indicador",
        "CategoriaIndicador",
        "Saldo",
        "TipoInstitucion",
    ]
    df = df[columnas_app].copy()
    print(f"Filas procesadas y ordenadas: {len(df):,}")
    return df


# ============================================================
# 5. EDA
# ============================================================
def ejecutar_eda(df):
    print("\n" + "=" * 70)
    print("4) EDA — DATASET TRANSFORMADO")
    print("=" * 70)
    print("\nTipos de institución:")
    print(df["TipoInstitucion"].value_counts(dropna=False).to_string())
    print("\nInstituciones por tipo:")
    print(
        df.groupby("TipoInstitucion")["Banco"]
        .nunique()
        .sort_values(ascending=False)
        .to_string()
    )
    print(f"\nIndicadores únicos   : {df['Indicador'].nunique()}")
    print(f"Instituciones únicas : {df['Banco'].nunique()}")
    print(
        f"Periodo              : {df['FechaReporte'].min().date()} → {df['FechaReporte'].max().date()}"
    )
    pct = df["Saldo"].isna().mean() * 100
    print(f"Nulos en Saldo       : {df['Saldo'].isna().sum():,} ({pct:.2f}%)")


# ============================================================
# 6. QUALITY GATE
# ============================================================
def quality_gate(df):
    print("\n" + "=" * 70)
    print("5) QUALITY GATE")
    print("=" * 70)
    errores = []

    if df.empty:
        errores.append("Dataset vacío")

    pct = df["Saldo"].isna().mean() if not df.empty else 1.0
    if pct > 0.05:
        errores.append(f"Nulos en Saldo > 5%: {pct * 100:.2f}%")

    instituciones = set(df["Banco"].dropna().unique())
    for banco in BANCOS_CLAVE:
        if banco not in instituciones:
            errores.append(f"Falta institución crítica: {banco}")

    if df["TipoInstitucion"].nunique() < 2:
        errores.append("Menos de 2 tipos de institución")

    if df["FechaReporte"].isna().any():
        errores.append("Existen fechas inválidas")

    if df["Indicador"].nunique() < 5:
        errores.append("Número de indicadores demasiado bajo")

    if errores:
        print("\n❌ QUALITY GATE FAILED")
        for e in errores:
            print(f" - {e}")
        raise ValueError("El dataset no pasó el Quality Gate.")

    print("✅ QUALITY GATE PASSED")
    print(f"Instituciones : {df['Banco'].nunique()}")
    print(f"Tipos         : {df['TipoInstitucion'].nunique()}")
    print(f"Indicadores   : {df['Indicador'].nunique()}")
    print(
        f"Periodo       : {df['FechaReporte'].min().date()} → {df['FechaReporte'].max().date()}"
    )


# ============================================================
# 7. DELTA CHECK
# ============================================================
def auditar_cambios_delta(df):
    print("\n" + "=" * 70)
    print("6) AUDITORÍA DE CAMBIOS (DELTA CHECK)")
    print("=" * 70)

    if not os.path.exists(ARCHIVO_SALIDA):
        print(f"ℹ️ {ARCHIVO_SALIDA} no encontrado. Generando dataset inicial.")
        return

    df_actual = pd.read_csv(ARCHIVO_SALIDA)
    df_actual["FechaReporte"] = pd.to_datetime(df_actual["FechaReporte"])
    prev, new = len(df_actual), len(df)
    f_prev = df_actual["FechaReporte"].max().date()
    f_new = df["FechaReporte"].max().date()

    if f_new > f_prev or new > prev:
        print("🟢 [NUEVOS DATOS DETECTADOS EN LA CNBS]")
        print(f" - Fecha máxima previa : {f_prev}")
        print(f" - Nueva fecha máxima  : {f_new}")
        print(f" - Incremento filas    : +{new - prev:,}")
    else:
        print("🟡 [SISTEMA AL DÍA]")
        print(f" - Fecha máxima actual : {f_new}")
        print(" - Se sobrescribe el CSV local para mantener orden y formato.")


# ============================================================
# 8. EXPORTACIÓN
# ============================================================
def exportar_dataset(df):
    print("\n" + "=" * 70)
    print("7) EXPORTACIÓN")
    print("=" * 70)
    df_export = df.copy()
    df_export["FechaReporte"] = df_export["FechaReporte"].dt.strftime("%Y-%m-%d")
    df_export.to_csv(ARCHIVO_SALIDA, index=False, encoding="utf-8-sig")
    size = Path(ARCHIVO_SALIDA).stat().st_size
    print(f"Archivo generado : {ARCHIVO_SALIDA}")
    print(f"Tamaño           : {size:,} bytes")
    print(f"Registros        : {len(df_export):,}")
    try:
        from google.colab import files  # type: ignore

        files.download(ARCHIVO_SALIDA)
        print("🚀 Descarga iniciada (Google Colab).")
    except ImportError:
        print("ℹ️ CSV guardado en el directorio actual.")
    return df_export


# ============================================================
# 9. RESUMEN
# ============================================================
def resumen_final(df):
    print("\n" + "=" * 70)
    print("8) RESUMEN DEL DATASET DE PRODUCCIÓN")
    print("=" * 70)
    print(f"Registros        : {len(df):,}")
    print(f"Instituciones    : {df['Banco'].nunique():,}")
    print(f"Tipos            : {df['TipoInstitucion'].nunique():,}")
    print(f"Indicadores      : {df['Indicador'].nunique():,}")
    print("\nPor tipo:")
    print(
        df.groupby("TipoInstitucion")
        .agg(Instituciones=("Banco", "nunique"), Registros=("Banco", "size"))
        .sort_values("Instituciones", ascending=False)
        .to_string()
    )
    print("\n✅ Pipeline OK. Dataset listo para producción.")


# ============================================================
# 10. MAIN
# ============================================================
def run_pipeline():
    print("\n🏦 PIPELINE FINANCIERO CNBS")
    print("Extracción → ETL → Quality Gate → Delta Check → Producción\n")
    df_raw = extraer_datos()
    validar_estructura_fuente(df_raw)
    df = normalizar_datos(df_raw)
    ejecutar_eda(df)
    quality_gate(df)
    auditar_cambios_delta(df)
    exportar_dataset(df)
    resumen_final(df)
    return df


if __name__ == "__main__":
    run_pipeline()
