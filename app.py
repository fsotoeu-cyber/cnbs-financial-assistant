import streamlit as st
import pandas as pd
import unicodedata
import re
import json
import os
import time
import plotly.express as px
from collections import defaultdict
from abc import ABC, abstractmethod
import logging

# ============================================================
# CONFIGURACIÓN
# ============================================================
class Config:
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.0
    TOP_RESULTS = 14
    ENTIDADES_AGREGADAS = ["BANCOS", "HONDURAS", "Sistema", "SISTEMA", "SISTEMA BANCARIO"]
    PRIORIDAD_TEMAS = ["credito", "riesgo", "salud", "rentabilidad", "solvencia", "liquidez"]
    SCORE_TRIPLE_EXCELENTE = 1.5
    SCORE_TRIPLE_BUENO = 0.8
    FORMULA_SCORE_TRIPLE = "(ROE / Morosidad) × (Capital / 100)"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente_cnbs")

# ============================================================
# AGENTES PARALELOS (opcional)
# ============================================================
try:
    from agentes_paralelos import analisis_paralelo_simple
    AGENTES_PARALELOS_DISPONIBLE = True
except ImportError:
    AGENTES_PARALELOS_DISPONIBLE = False
    def analisis_paralelo_simple(llm, df, ctx):
        return "⚠️ Módulo de agentes paralelos no disponible. Instala agentes_paralelos.py"

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def nivel_score(valor, excelente=None, bueno=None):
    excelente = Config.SCORE_TRIPLE_EXCELENTE if excelente is None else excelente
    bueno = Config.SCORE_TRIPLE_BUENO if bueno is None else bueno
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "N/D", "⚪"
    if v >= excelente:
        return "Excelente", "🟢"
    if v >= bueno:
        return "Bueno", "🟡"
    return "Bajo", "🔴"

def columnas_ranking(df_res):
    if df_res is None or df_res.empty:
        return None
    cols = set(df_res.columns)
    if "Score_Triple" in cols:
        return "triple"
    if "Ratio_ROE_Mora" in cols and "Ranking" in cols:
        return "roe_mora"
    if "Score_ROE_Capital" in cols and "Ranking" in cols:
        return "roe_capital"
    return None

def conclusion_ranking(df_res, tipo=None):
    if df_res is None or df_res.empty or "Banco" not in df_res.columns:
        return ""
    tipo = tipo or columnas_ranking(df_res)
    top = df_res.iloc[0]
    banco = str(top["Banco"])
    anio = ""
    if "Año" in df_res.columns and pd.notna(top.get("Año")):
        anio = f" en {int(top['Año'])}"
    n = len(df_res)
    if tipo == "triple":
        sc = float(top["Score_Triple"])
        nivel, _ = nivel_score(sc)
        return (
            f"**Conclusión:** **{banco}** presenta el mejor equilibrio entre rentabilidad, "
            f"riesgo crediticio y solvencia entre los {n} bancos analizados{anio} "
            f"(Score = {sc:.2f}, nivel {nivel}; fórmula: {Config.FORMULA_SCORE_TRIPLE})."
        )
    if tipo == "roe_mora":
        ratio = float(top["Ratio_ROE_Mora"])
        return (
            f"**Conclusión:** **{banco}** presenta la mejor relación rentabilidad–riesgo "
            f"entre los {n} bancos analizados{anio} (Ratio ROE/Mora = {ratio:.2f})."
        )
    if tipo == "roe_capital":
        sc = float(top["Score_ROE_Capital"])
        return (
            f"**Conclusión:** **{banco}** presenta el mejor equilibrio rentabilidad–solvencia "
            f"entre los {n} bancos analizados{anio} (Score ROE×Capital/100 = {sc:.2f})."
        )
    return f"**Conclusión:** **{banco}** ocupa el primer lugar del ranking calculado por Pandas{anio}."

def aplanar_lista(lista):
    if lista is None:
        return []
    if not isinstance(lista, list):
        return [lista]
    out = []
    for item in lista:
        out.extend(aplanar_lista(item) if isinstance(item, list) else [item])
    return out

def normalizar_texto(texto):
    if not texto:
        return ""
    t = unicodedata.normalize("NFD", str(texto))
    return "".join(c for c in t if unicodedata.category(c) != "Mn").lower()

def extraer_anios(pregunta):
    return sorted(set(int(a) for a in re.findall(r"\b(20\d{2})\b", str(pregunta))))

ALIAS_BANCOS = {
    "atlantida": ["BANCATLAN"], "bancatlan": ["BANCATLAN"], "atlántida": ["BANCATLAN"],
    "occidente": ["BANCOCCI"], "bancocci": ["BANCOCCI"], "occi": ["BANCOCCI"],
    "banpais": ["BANPAIS"], "pais": ["BANPAIS"], "banpaís": ["BANPAIS"],
    "ficohsa": ["FICOHSA"], "ficocsa": ["FICOHSA"],
    "bac": ["BAC CREDOMATIC"], "credomatic": ["BAC CREDOMATIC"],
    "davivienda": ["BANCO DAVIVIENDA"], "davivenda": ["BANCO DAVIVIENDA"],
    "popular": ["BANCO POPULAR"],
    "cuscatlan": ["CUSCATLAN HONDURAS"], "cuscatlán": ["CUSCATLAN HONDURAS"],
    "promerica": ["PROMERICA"], "promérica": ["PROMERICA"],
    "lafise": ["LAFISE"], "banrural": ["BANRURAL"], "rural": ["BANRURAL"],
    "azteca": ["AZTECA"], "asteca": ["AZTECA"],
    "banhcafe": ["BANHCAFE"], "banhcafé": ["BANHCAFE"], "ficensa": ["FICENSA"],
}

def extraer_bancos(pregunta):
    q = normalizar_texto(pregunta)
    found = []
    for alias, reales in ALIAS_BANCOS.items():
        if alias in q:
            found.extend(reales)
    return aplanar_lista(list(dict.fromkeys(found)))

SINONIMOS_INDICADORES = {
    "roa": "RENTABILIDAD SOBRE ACTIVOS (ROA): RESULTADOS EL EJERCICIO (ANUALIZADOS)/ACTIVOS TOTALES PROMEDIO",
    "rentabilidad sobre activos": "RENTABILIDAD SOBRE ACTIVOS (ROA): RESULTADOS EL EJERCICIO (ANUALIZADOS)/ACTIVOS TOTALES PROMEDIO",
    "roe": "RENTABILIDAD SOBRE EL PATRIMONIOS (ROE): RESULTADOS EL EJERCICIO (ANUALIZADOS)/CAPITAL Y RESERVAS PROMEDIO",
    "rentabilidad sobre patrimonio": "RENTABILIDAD SOBRE EL PATRIMONIOS (ROE): RESULTADOS EL EJERCICIO (ANUALIZADOS)/CAPITAL Y RESERVAS PROMEDIO",
    "rentabilidad sobre el patrimonio": "RENTABILIDAD SOBRE EL PATRIMONIOS (ROE): RESULTADOS EL EJERCICIO (ANUALIZADOS)/CAPITAL Y RESERVAS PROMEDIO",
    "morosidad directa": "INDICE DE MOROSIDAD SOBRE CARTERA CREDITICIA DIRECTA (SIN CONTINGENTES)",
    "mora directa": "INDICE DE MOROSIDAD SOBRE CARTERA CREDITICIA DIRECTA (SIN CONTINGENTES)",
    "morosidad total": "ÍNDICE DE MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL",
    "morosidad": "ÍNDICE DE MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL",
    "mora": "ÍNDICE DE MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL",
    "adecuacion de capital": "INDICE DE ADECUACIÓN DE CAPITAL",
    "adecuacion": "INDICE DE ADECUACIÓN DE CAPITAL",
    "indice de adecuacion": "INDICE DE ADECUACIÓN DE CAPITAL",
    "capital": "INDICE DE ADECUACIÓN DE CAPITAL",
    "solvencia": "INDICE DE ADECUACIÓN DE CAPITAL",
    "spread de intermediacion": "SPREAD DE INTERMEDIACIÓN",
    "spread": "SPREAD DE INTERMEDIACIÓN",
    "tasa activa": "TASA ACTIVA DE INTERMEDIACIÓN",
    "tasa pasiva": "TASA PASIVA PARA INTERMEDIACIÓN",
    "liquidez": "ÍNDICE DE LIQUIDEZ: COBERTURA DE DEPÓSITOS DEL PÚBLICO CON ACTIVOS LÍQUIDOS",
    "cobertura de mora": "ÍNDICE DE COBERTURA DE LA MORA DE CARTERA",
    "cartera de tarjetas": "CARTERA DE TARJETAS DE CRÉDITO/CARTERA CREDITICIA TOTAL",
    "tarjetas de credito": "CARTERA DE TARJETAS DE CRÉDITO/CARTERA CREDITICIA TOTAL",
    "tarjeta de credito": "CARTERA DE TARJETAS DE CRÉDITO/CARTERA CREDITICIA TOTAL",
    "exposicion a tarjetas": "CARTERA DE TARJETAS DE CRÉDITO/CARTERA CREDITICIA TOTAL",
    "tarjetas": "CARTERA DE TARJETAS DE CRÉDITO/CARTERA CREDITICIA TOTAL",
    "cobertura de la mora": "ÍNDICE DE COBERTURA DE LA MORA DE CARTERA",
}

TEMAS_FINANCIEROS = {
    "riesgo": ["morosidad", "cobertura de mora", "capital", "liquidez", "roa", "roe"],
    "credito": ["morosidad", "cobertura de mora", "cartera de tarjetas"],
    "rentabilidad": ["roa", "roe", "spread"],
    "solvencia": ["capital", "liquidez"],
    "liquidez": ["liquidez"],
    "salud": ["roa", "roe", "capital", "liquidez", "morosidad", "spread"],
}

PALABRAS_CLAVE_TEMAS = {
    "riesgo": ["riesgo", "riesgos", "riesgoso", "estabilidad"],
    "credito": ["credito", "crediticio", "cartera", "morosidad", "mora"],
    "rentabilidad": ["rentabilidad", "rentable", "roa", "roe", "spread", "margen"],
    "solvencia": ["solvencia", "capital", "adecuacion", "patrimonio"],
    "liquidez": ["liquidez", "liquido", "depositos", "cobertura"],
    "salud": ["salud", "sistema", "general", "panorama", "estado", "contexto"],
}

def validar_dataframe(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return False, "DataFrame vacío"
    req = {"FechaReporte", "Banco", "Indicador", "Saldo"}
    if req - set(df.columns):
        return False, f"Faltan: {req - set(df.columns)}"
    return True, None

def validar_resultado(df):
    if df is None or df.empty:
        return False
    if "Ratio_ROE_Mora" in df.columns:
        return not df["Ratio_ROE_Mora"].isna().all()
    if "Saldo" in df.columns:
        return not df["Saldo"].isna().all()
    return True

def construir_catalogo(df):
    cat = {normalizar_texto(i): i for i in df["Indicador"].dropna().unique()}
    for a, o in SINONIMOS_INDICADORES.items():
        cat[normalizar_texto(a)] = o
    return cat

def buscar_indicador(palabra_clave, catalogo):
    if not palabra_clave:
        return None
    clave = normalizar_texto(palabra_clave)
    if clave in catalogo:
        return catalogo[clave]
    for k in sorted(catalogo.keys(), key=len):
        if k == clave or (len(clave) > 3 and (clave in k or k in clave)):
            if clave in SINONIMOS_INDICADORES or normalizar_texto(clave) in {normalizar_texto(x) for x in SINONIMOS_INDICADORES}:
                if k == clave or clave in SINONIMOS_INDICADORES:
                    return catalogo.get(clave) or catalogo[k]
            return catalogo[k]
    for k, v in catalogo.items():
        if clave in k or k in clave:
            return v
    return None

def obtener_indicadores_por_tema(tema, catalogo):
    if tema not in TEMAS_FINANCIEROS:
        return []
    out, seen = [], set()
    for c in TEMAS_FINANCIEROS[tema]:
        ind = buscar_indicador(c, catalogo)
        if ind and ind not in seen:
            out.append(ind)
            seen.add(ind)
    return out

def extraer_claves_indicador(pregunta):
    q = normalizar_texto(pregunta)
    peso = {
        "morosidad directa": 10, "mora directa": 10, "morosidad total": 9,
        "rentabilidad sobre patrimonio": 9,
        "rentabilidad sobre el patrimonio": 9,
        "roe": 9,
        "rentabilidad sobre activos": 6,
        "roa": 6,
        "adecuacion de capital": 8,
        "indice de adecuacion": 8,
        "spread de intermediacion": 6,
        "cartera de tarjetas": 7, "tarjetas de credito": 7, "tarjeta de credito": 7,
        "exposicion a tarjetas": 7, "tarjetas": 6,
        "tasa activa": 6, "tasa pasiva": 6, "cobertura de mora": 6, "cobertura de la mora": 6,
        "morosidad": 5, "rentabilidad": 4, "cobertura": 4,
        "mora": 3, "adecuacion": 3, "capital": 2, "solvencia": 2, "spread": 2, "liquidez": 2,
    }
    candidatos = sorted([c for c in peso if c in q], key=lambda x: peso[x], reverse=True)
    claves = []
    for c in candidatos:
        if c in claves:
            continue
        if c == "capital" and any(x in claves for x in ("adecuacion", "adecuacion de capital", "solvencia")):
            continue
        if c == "mora" and any(x.startswith("morosidad") for x in claves):
            continue
        if c in ("roa", "rentabilidad sobre activos") and ("roe" in q or "patrimonio" in q) and "roa" not in q and "activos" not in q:
            continue
        if c in ("roe", "rentabilidad sobre patrimonio", "rentabilidad sobre el patrimonio") and "roa" in q and "roe" not in q and "patrimonio" not in q:
            pass
        claves.append(c)
    if ("roe" in q or "patrimonio" in q) and "roa" not in q and "activos" not in q:
        claves = [c for c in claves if c not in ("roa", "rentabilidad sobre activos")]
    if "roa" in q and "roe" not in q and "patrimonio" not in q:
        claves = [c for c in claves if c not in ("roe", "rentabilidad sobre patrimonio", "rentabilidad sobre el patrimonio")]
    return claves

def resolver_indicadores(claves, catalogo):
    res, seen = [], set()
    for c in claves:
        ind = buscar_indicador(c, catalogo)
        if ind and ind not in seen:
            res.append(ind)
            seen.add(ind)
    return res

class ContextoConversacional:
    def __init__(self):
        self.historial = []
        self.ultimo_tema = None
        self.ultimo_banco = None
        self.ultimo_anio = None
        self.ultimos_indicadores = []
        self.preferencias = defaultdict(lambda: defaultdict(int))
    def actualizar(self, pregunta, respuesta, tema=None, banco=None, anio=None, indicadores=None):
        banco = aplanar_lista(banco) if banco else []
        self.historial.append({"pregunta": pregunta, "tema": tema, "banco": banco, "anio": anio, "indicadores": indicadores})
        if tema:
            self.ultimo_tema = tema
        if banco:
            self.ultimo_banco = banco
        if anio:
            self.ultimo_anio = anio
        if indicadores:
            self.ultimos_indicadores = indicadores
    def obtener_contexto(self):
        if not self.historial:
            return "No hay contexto previo."
        u = self.historial[-1]
        parts = []
        if u.get("tema"):
            parts.append(f"Tema: {u['tema']}")
        if u.get("banco"):
            parts.append(f"Banco: {', '.join(u['banco']) if isinstance(u['banco'], list) else u['banco']}")
        if u.get("anio"):
            parts.append(f"Año: {u['anio']}")
        return " | ".join(parts) if parts else "Sin contexto"
    def limpiar(self):
        self.__init__()

def detectar_tema(query):
    q = normalizar_texto(query)
    for tema in Config.PRIORIDAD_TEMAS:
        if any(k in q for k in PALABRAS_CLAVE_TEMAS.get(tema, [])):
            return tema
    return None

def detectar_tipo_consulta(query):
    q = normalizar_texto(query)
    if any(w in q for w in ("compara", "comparacion", "versus", " vs ", "entre")):
        return "comparar"
    if any(w in q for w in (
        "ranking", "top", "mayor", "peor", "menor",
        "preocupante", "mayor riesgo", "mas riesgoso", "peor perfil",
        "que bancos", "que banco", "identifica", "cuáles bancos", "cuales bancos",
    )):
        return "ranking"
    if "mejor" in q and not any(w in q for w in ("compara", "entre", "versus")):
        return "ranking"
    if any(w in q for w in ("historico", "serie", "evolucion", "tendencia")):
        return "serie"
    return "promedio"

def es_analisis_riesgo_crediticio(query):
    q = normalizar_texto(query)
    habla_riesgo = any(p in q for p in (
        "riesgo crediticio", "riesgo de credito", "morosidad", "cobertura de mora",
        "calidad de activos", "cartera de tarjetas",
    ))
    pide_detalle_bancos = any(p in q for p in (
        "banco", "bancos", "preocupante", "mayor riesgo", "peor", "ranking",
        "identifica", "perfil", "detallado", "profundo", "compar", "evalua",
        "evaluación", "evaluacion", "implicaciones", "recomenda",
    ))
    if "informe ejecutivo" in q and not any(p in q for p in ("banco", "bancos", "preocupante", "mayor riesgo")):
        return False
    return habla_riesgo and pide_detalle_bancos

def detectar_orden_ranking(query):
    q = normalizar_texto(query)
    return any(w in q for w in ("menor", "peor", "mas bajo", "seguro", "menos riesgoso"))

def es_consulta_rentabilidad_riesgo(query):
    q = normalizar_texto(query)
    if any(p in q for p in ("roa", "adecuacion", "capital", "spread", "liquidez")):
        return False
    return any(p in q for p in (
        "rentabilidad-riesgo", "rentabilidad riesgo", "relacion rentabilidad",
        "roe y morosidad", "roe y mora", "mejor relacion",
    ))

def es_consulta_conversacional(query):
    q = normalizar_texto(query or "").strip()
    if not q:
        return True
    marcas_fin = (
        "roa", "roe", "morosidad", "mora", "capital", "adecuacion", "banco",
        "sistema", "ranking", "compara", "evolucion", "spread", "liquidez",
        "cobertura", "tarjetas", "cnbs", "indicador", "rentabilidad", "solvencia",
        "equilibrio", "score", "2024", "2025", "2026",
    )
    if any(m in q for m in marcas_fin):
        return False
    patrones = (
        "que dia", "qué dia", "que fecha", "qué fecha", "hoy es", "fecha de hoy",
        "que hora", "qué hora", "hora actual",
        "hola", "buenos dias", "buenas tardes", "buenas noches", "buen dia",
        "gracias", "muchas gracias",
        "quien eres", "qué eres", "que eres", "como te llamas", "ayuda", "help",
        "que puedes hacer", "qué puedes hacer",
    )
    return any(p in q for p in patrones)

def respuesta_conversacional(query):
    from datetime import datetime
    q = normalizar_texto(query or "")
    ahora = datetime.now()
    dias = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
    meses = (
        "", "enero", "febrero", "marzo", "abril", "mayo", "junio",
        "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
    )
    fecha = f"{dias[ahora.weekday()]} {ahora.day} de {meses[ahora.month]} de {ahora.year}"
    hora = ahora.strftime("%H:%M")
    if any(p in q for p in ("que dia", "qué dia", "que fecha", "qué fecha", "hoy es", "fecha de hoy")):
        return f"Hoy es **{fecha}**."
    if any(p in q for p in ("que hora", "qué hora", "hora actual")):
        return f"Son las **{hora}**."
    if any(p in q for p in ("hola", "buenos dias", "buenas tardes", "buenas noches", "buen dia")):
        return f"Hola. Soy el **Agente Financiero CNBS**. Puedes consultar ROA, ROE, morosidad y capital. _Hoy es {fecha}._"
    if any(p in q for p in ("gracias", "muchas gracias")):
        return "Con gusto. Cuando quieras, haz otra consulta sobre indicadores CNBS."
    if any(p in q for p in ("quien eres", "que eres", "qué eres", "como te llamas")):
        return "Soy el **Agente Financiero CNBS**: analizo indicadores del sistema bancario hondureño con Pandas y, si hace falta, un LLM solo para redactar."
    if any(p in q for p in ("ayuda", "help", "que puedes hacer", "qué puedes hacer")):
        return "Puedo ayudarte con rankings, comparaciones entre bancos, evolución temporal y riesgo crediticio. Ejemplo: *¿Qué banco tiene mejor relación ROE/morosidad en 2025?*"
    return f"Esta consulta no parece referirse a indicadores CNBS. Prueba con ROA, ROE, morosidad o un banco. _Hoy es {fecha}._"

def planificar(query, contexto, catalogo):
    q = normalizar_texto(query)
    bancos = aplanar_lista(contexto.ultimo_banco) if contexto.ultimo_banco else []
    anios = extraer_anios(query) or ([contexto.ultimo_anio] if contexto.ultimo_anio else [])

    claves = extraer_claves_indicador(query)
    if claves:
        inds = resolver_indicadores(claves, catalogo)
        pide_rent_solv = any(c in claves for c in (
            "roe", "roa", "rentabilidad sobre patrimonio", "rentabilidad sobre el patrimonio",
            "rentabilidad sobre activos", "adecuacion de capital", "indice de adecuacion",
            "capital", "solvencia",
        ))
        if (not pide_rent_solv) and any(p in q for p in (
            "riesgo crediticio", "riesgo de credito", "calidad de activos",
        )):
            for extra in obtener_indicadores_por_tema("credito", catalogo):
                if extra not in inds:
                    inds.append(extra)
        if inds:
            return inds, None, detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    if any(p in q for p in ("invertir", "recomendar", "inversion")):
        inds = obtener_indicadores_por_tema("rentabilidad", catalogo)
        if inds:
            return inds, "rentabilidad", detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    if es_analisis_riesgo_crediticio(query):
        inds = obtener_indicadores_por_tema("credito", catalogo)
        if inds:
            return inds, "credito", "ranking", bancos, anios, False

    tema = detectar_tema(query)
    if tema:
        inds = obtener_indicadores_por_tema(tema, catalogo)
        if inds:
            return inds, tema, detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    if contexto.ultimo_tema:
        inds = obtener_indicadores_por_tema(contexto.ultimo_tema, catalogo)
        if inds:
            return inds, contexto.ultimo_tema, detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    return obtener_indicadores_por_tema("salud", catalogo), "salud", detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

def filtrar_anio(df, anio):
    if anio:
        d = df[df["FechaReporte"].dt.year == anio].copy()
        return d if not d.empty else pd.DataFrame()
    m = df["FechaReporte"].max()
    return df[df["FechaReporte"] == m].copy()

def _base_filtrada(df, bancos, indicadores, anio):
    d = filtrar_anio(df, anio)
    if d.empty:
        return d
    d = d[~d["Banco"].isin(Config.ENTIDADES_AGREGADAS)]
    if bancos:
        d = d[d["Banco"].isin(aplanar_lista(bancos))]
    if indicadores:
        d = d[d["Indicador"].isin(indicadores)]
    return d

def promedio(df, bancos, indicadores, anio):
    d = _base_filtrada(df, bancos, indicadores, anio)
    if d.empty:
        return d
    if not bancos:
        rows = []
        for ind in (indicadores or list(d["Indicador"].unique())):
            sub = d[d["Indicador"] == ind]
            if sub.empty:
                continue
            val = sub.groupby("Banco")["Saldo"].mean().mean()
            rows.append({"Banco": "Sistema", "Indicador": ind, "Saldo": float(val)})
        return pd.DataFrame(rows)
    return d.groupby(["Banco", "Indicador"], dropna=False)["Saldo"].mean().reset_index()

def ranking(df, indicadores, anio, top=10, ascending=False):
    if indicadores and len(indicadores) > 1:
        d = _base_filtrada(df, None, indicadores, anio)
        if d.empty:
            return d
        res = d.groupby(["Banco", "Indicador"], dropna=False)["Saldo"].mean().reset_index()
        mora_mask = res["Indicador"].str.contains("MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL", case=False, na=False)
        if mora_mask.any():
            orden = (
                res[mora_mask]
                .sort_values("Saldo", ascending=False)["Banco"]
                .tolist()
            )
            res["__ord"] = res["Banco"].apply(lambda b: orden.index(b) if b in orden else 999)
            res = res.sort_values(["__ord", "Indicador"]).drop(columns="__ord")
            if top and top < res["Banco"].nunique():
                top_bancos = orden[:top]
                res = res[res["Banco"].isin(top_bancos)]
        return res.reset_index(drop=True)
    d = _base_filtrada(df, None, indicadores, anio)
    if d.empty:
        return d
    res = d.groupby(["Banco", "Indicador"], dropna=False)["Saldo"].mean().reset_index()
    return res.sort_values(by="Saldo", ascending=ascending).head(top)

def comparar_bancos(df, bancos, indicadores, anio):
    return promedio(df, aplanar_lista(bancos), indicadores, anio)

def serie_temporal(df, bancos, indicadores):
    d = df[~df["Banco"].isin(Config.ENTIDADES_AGREGADAS)].copy()
    if bancos:
        d = d[d["Banco"].isin(aplanar_lista(bancos))]
    if indicadores:
        d = d[d["Indicador"].isin(indicadores)]
    return d[["FechaReporte", "Banco", "Indicador", "Saldo"]].sort_values("FechaReporte")

def ranking_rentabilidad_riesgo(df, anio, top=10, bancos=None):
    base = df[(df["FechaReporte"].dt.year == anio) & (~df["Banco"].isin(Config.ENTIDADES_AGREGADAS))]
    base = base[~base["Banco"].astype(str).str.upper().isin({"SISTEMA", "BANCOS", "HONDURAS"})]
    if bancos:
        base = base[base["Banco"].isin(aplanar_lista(bancos))]
    roe = base[base["Indicador"].str.contains("ROE", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    mora = base[base["Indicador"].str.contains(
        "MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL", case=False, na=False
    )].groupby("Banco")["Saldo"].mean()
    j = pd.DataFrame({"ROE": roe, "Morosidad": mora}).dropna()
    j = j[j["Morosidad"] > 0].copy()
    j["Ratio_ROE_Mora"] = j["ROE"] / j["Morosidad"]
    j = j.sort_values("Ratio_ROE_Mora", ascending=False).head(top).reset_index()
    j.insert(0, "Ranking", range(1, len(j) + 1))
    j["Ranking"] = j["Ranking"].astype(int)
    j["Año"] = int(anio)
    j["ROE"] = j["ROE"].round(2)
    j["Morosidad"] = j["Morosidad"].round(2)
    j["Ratio_ROE_Mora"] = j["Ratio_ROE_Mora"].round(2)
    return j

def es_consulta_roe_solvencia(query):
    q = normalizar_texto(query)
    habla_roe = "roe" in q or "rentabilidad" in q
    habla_cap = any(p in q for p in ("adecuacion", "solvencia", "capital", "solvencia"))
    pide_rank = any(p in q for p in ("ranking", "mejores", "mejor equilibrio", "equilibrio", "top", "primer lugar", "bancos con mejor"))
    if "morosidad" in q or "mora" in q:
        return False
    return habla_roe and habla_cap and (pide_rank or "equilibrio" in q)

def ranking_roe_solvencia(df, anio, top=10, bancos=None):
    base = df[(df["FechaReporte"].dt.year == anio)].copy()
    base = base[~base["Banco"].isin(Config.ENTIDADES_AGREGADAS)]
    base = base[~base["Banco"].astype(str).str.upper().str.contains("SISTEMA")]
    if bancos:
        base = base[base["Banco"].isin(aplanar_lista(bancos))]
    roe = base[base["Indicador"].str.contains("ROE", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    cap = base[base["Indicador"].str.contains("ADECUACI", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    j = pd.DataFrame({"ROE": roe, "Capital": cap}).dropna()
    j = j[(j["Capital"] > 0)].copy()
    if j.empty:
        return j
    j["Score_ROE_Capital"] = (j["ROE"] * j["Capital"] / 100.0).round(2)
    j = j.sort_values("Score_ROE_Capital", ascending=False).head(top).reset_index()
    j.insert(0, "Ranking", range(1, len(j) + 1))
    j["Ranking"] = j["Ranking"].astype(int)
    j["Año"] = int(anio)
    j["ROE"] = j["ROE"].round(2)
    j["Capital"] = j["Capital"].round(2)
    return j

def es_consulta_equilibrio_triple(query):
    q = normalizar_texto(query)
    habla_roe = "roe" in q or "rentabilidad" in q
    habla_mora = "morosidad" in q or "mora" in q or "riesgo" in q
    habla_cap = any(p in q for p in ("adecuacion", "capital", "solvencia"))
    pide_eq = any(p in q for p in ("equilibrio", "equilibrado", "mejor perfil", "perfil mas", "ranking", "mejores", "mejor banco"))
    return habla_roe and habla_mora and habla_cap and pide_eq

def ranking_equilibrio_triple(df, anio, top=10, bancos=None):
    base = df[(df["FechaReporte"].dt.year == anio)].copy()
    base = base[~base["Banco"].isin(Config.ENTIDADES_AGREGADAS)]
    base = base[~base["Banco"].astype(str).str.upper().str.contains("SISTEMA")]
    if bancos:
        base = base[base["Banco"].isin(aplanar_lista(bancos))]
    roe = base[base["Indicador"].str.contains("ROE", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    mora = base[base["Indicador"].str.contains("MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    cap = base[base["Indicador"].str.contains("ADECUACI", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    j = pd.DataFrame({"ROE": roe, "Morosidad": mora, "Capital": cap}).dropna()
    j = j[j["Morosidad"] > 0].copy()
    if j.empty:
        return j
    j["Ratio_ROE_Mora"] = (j["ROE"] / j["Morosidad"]).round(2)
    j["Score_Triple"] = ((j["ROE"] / j["Morosidad"]) * (j["Capital"] / 100.0)).round(2)
    j = j.sort_values("Score_Triple", ascending=False).head(top).reset_index()
    j.insert(0, "Ranking", range(1, len(j) + 1))
    j["Ranking"] = j["Ranking"].astype(int)
    j["Año"] = int(anio)
    j["ROE"] = j["ROE"].round(2)
    j["Morosidad"] = j["Morosidad"].round(2)
    j["Capital"] = j["Capital"].round(2)
    return j

def calcular_confianza(df_res, indicadores=None, bancos=None, anios=None):
    if df_res is None or df_res.empty:
        return "Baja", "Sin filas de resultado"
    if "Ranking" in df_res.columns and "Score_Triple" in df_res.columns:
        return "Alta", f"{len(df_res)} bancos; score (ROE/Mora)×(Capital/100)"
    if "Ranking" in df_res.columns and "Score_ROE_Capital" in df_res.columns:
        return "Alta", f"{len(df_res)} bancos con ROE y capital; ranking Pandas"
    if "Ranking" in df_res.columns and "Ratio_ROE_Mora" in df_res.columns:
        return "Alta", f"{len(df_res)} bancos con ROE y morosidad; ranking Pandas"
    if indicadores and "Indicador" in df_res.columns:
        presentes = set(df_res["Indicador"].unique())
        faltan = [i for i in indicadores if i not in presentes]
        if faltan:
            return "Media", f"Faltan {len(faltan)} indicador(es) de los pedidos"
    if bancos and "Banco" in df_res.columns:
        ped = set(aplanar_lista(bancos))
        if ped - set(df_res["Banco"].unique()):
            return "Media", "Algunos bancos pedidos sin dato"
    parts = [f"{len(df_res)} filas", "Datos CNBS"]
    if indicadores:
        parts.insert(0, f"{len(indicadores)} indicadores pedidos")
    if anios:
        parts.append(f"Años: {', '.join(map(str, anios))}")
    return "Alta", "; ".join(parts)

def usuario_pide_detalle(query):
    q = normalizar_texto(query)
    return any(p in q for p in (
        "explica", "explicalo", "explícalo", "por que", "porqué", "porque",
        "detalla por que", "detalla porqué",
        "analiza en detalle", "analisis profundo", "análisis profundo",
        "informe narrativo", "extenso", "desarrolla", "redacta un ensayo",
        "menciona al menos", "factores", "implica",
    ))

def consulta_simple_sistema(query, df_res):
    if df_res is None or df_res.empty:
        return False
    if usuario_pide_detalle(query):
        return False
    n_ind = df_res["Indicador"].nunique() if "Indicador" in df_res.columns else 0
    n_banco = df_res["Banco"].nunique() if "Banco" in df_res.columns else 0
    q = normalizar_texto(query)
    if n_ind <= 3 and n_banco <= 2 and not any(p in q for p in ("equilibrado", "compara", "versus")):
        return True
    return False

class ConsultaStrategy(ABC):
    @abstractmethod
    def ejecutar(self, df, indicadores, bancos, anio, **kwargs):
        pass

class RankingStrategy(ConsultaStrategy):
    def ejecutar(self, df, indicadores, bancos, anio, **kwargs):
        return ranking(df, indicadores, anio, kwargs.get("top", 10), kwargs.get("asc", False))

class CompararStrategy(ConsultaStrategy):
    def ejecutar(self, df, indicadores, bancos, anio, **kwargs):
        if not bancos:
            return promedio(df, [], indicadores, anio)
        return comparar_bancos(df, bancos, indicadores, anio)

class SerieStrategy(ConsultaStrategy):
    def ejecutar(self, df, indicadores, bancos, anio, **kwargs):
        return serie_temporal(df, bancos, indicadores)

class PromedioStrategy(ConsultaStrategy):
    def ejecutar(self, df, indicadores, bancos, anio, **kwargs):
        return promedio(df, bancos, indicadores, anio)

class StrategyFactory:
    _s = {
        "ranking": RankingStrategy(),
        "comparar": CompararStrategy(),
        "serie": SerieStrategy(),
        "promedio": PromedioStrategy(),
    }
    @classmethod
    def get(cls, tipo):
        return cls._s.get(tipo, cls._s["promedio"])

def ejecutar_consulta(df, indicadores, bancos, anios, tipo, **kwargs):
    query = kwargs.get("query", "")
    bancos = aplanar_lista(bancos) if bancos else []
    if not anios:
        anios = [int(df["FechaReporte"].dt.year.max())]

    if es_consulta_rentabilidad_riesgo(query):
        partes = []
        for anio in anios:
            sub = ranking_rentabilidad_riesgo(df, anio, top=kwargs.get("top", 10), bancos=bancos or None)
            if not sub.empty:
                partes.append(sub)
        if partes:
            return pd.concat(partes, ignore_index=True)

    if es_consulta_equilibrio_triple(query):
        partes = []
        for anio in anios:
            sub = ranking_equilibrio_triple(df, anio, top=kwargs.get("top", 10), bancos=bancos or None)
            if not sub.empty:
                partes.append(sub)
        if partes:
            return pd.concat(partes, ignore_index=True)
        return pd.DataFrame()

    if es_consulta_roe_solvencia(query):
        partes = []
        for anio in anios:
            sub = ranking_roe_solvencia(df, anio, top=kwargs.get("top", 10), bancos=bancos or None)
            if not sub.empty:
                partes.append(sub)
        if partes:
            return pd.concat(partes, ignore_index=True)
        return pd.DataFrame()

    if not indicadores:
        return pd.DataFrame()

    if bancos and tipo == "ranking" and "compara" in normalizar_texto(query):
        tipo = "comparar"

    qn = normalizar_texto(query)
    if tipo in ("promedio", "comparar") and not bancos and any(
        p in qn for p in ("ranking", "mejores bancos", "top ", "mejor equilibrio", "5 bancos", "cinco bancos")
    ):
        tipo = "ranking"

    out = []
    for anio in anios:
        res = StrategyFactory.get(tipo).ejecutar(df, indicadores, bancos, anio, **kwargs)
        if not res.empty:
            res = res.copy()
            res["Año"] = int(anio)
            out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def necesita_llm(df_res, query, meta_info):
    if df_res is None or df_res.empty:
        return False
    if "Ranking" in df_res.columns and "Ratio_ROE_Mora" in df_res.columns:
        return usuario_pide_detalle(query)
    if consulta_simple_sistema(query, df_res) and not usuario_pide_detalle(query):
        return False
    if (
        "Indicador" in df_res.columns
        and df_res["Indicador"].nunique() >= 2
        and "Banco" in df_res.columns
        and df_res["Banco"].nunique() >= 2
        and not usuario_pide_detalle(query)
    ):
        return False
    if "Año" in df_res.columns and df_res["Año"].nunique() > 1 and not usuario_pide_detalle(query):
        return False
    q = normalizar_texto(query)
    if any(p in q for p in ("equilibrado", "relacion", "relativa")) and not usuario_pide_detalle(query):
        return False
    if (
        "Indicador" in df_res.columns
        and "Banco" in df_res.columns
        and df_res["Banco"].nunique() >= 3
        and df_res["Indicador"].nunique() >= 2
        and not usuario_pide_detalle(query)
    ):
        return False
    if any(p in q for p in ("recomendar", "explica", "informe narrativo")):
        return True
    if any(p in q for p in ("analiza", "riesgo", "salud", "evolucion", "informe")) and not (
        "Indicador" in df_res.columns and df_res["Indicador"].nunique() >= 2
    ):
        return True
    if len(df_res) > 12:
        return True
    return False

def preparar_datos_para_redactor(df_res):
    if df_res is None or df_res.empty:
        return []
    if "Ranking" in df_res.columns and "Score_Triple" in df_res.columns:
        datos = []
        for _, row in df_res.iterrows():
            item = {
                "Ranking": int(row["Ranking"]),
                "Banco": row.get("Banco"),
                "ROE": float(row["ROE"]) if pd.notna(row.get("ROE")) else None,
                "Morosidad": float(row["Morosidad"]) if pd.notna(row.get("Morosidad")) else None,
                "Capital": float(row["Capital"]) if pd.notna(row.get("Capital")) else None,
                "Score_Triple": float(row["Score_Triple"]) if pd.notna(row.get("Score_Triple")) else None,
            }
            if "Año" in row.index and pd.notna(row.get("Año")):
                item["Año"] = int(row["Año"])
            datos.append(item)
        return datos
    if "Ranking" in df_res.columns and "Score_ROE_Capital" in df_res.columns:
        datos = []
        for _, row in df_res.iterrows():
            item = {
                "Ranking": int(row["Ranking"]),
                "Banco": row.get("Banco"),
                "ROE": float(row["ROE"]) if pd.notna(row.get("ROE")) else None,
                "Capital": float(row["Capital"]) if pd.notna(row.get("Capital")) else None,
                "Score_ROE_Capital": float(row["Score_ROE_Capital"]) if pd.notna(row.get("Score_ROE_Capital")) else None,
            }
            if "Año" in row.index and pd.notna(row.get("Año")):
                item["Año"] = int(row["Año"])
            datos.append(item)
        return datos
    if "Ranking" in df_res.columns and "Ratio_ROE_Mora" in df_res.columns:
        datos = []
        for _, row in df_res.iterrows():
            item = {
                "Ranking": int(row["Ranking"]),
                "Banco": row.get("Banco"),
                "ROE": float(row["ROE"]) if pd.notna(row.get("ROE")) else None,
                "Morosidad": float(row["Morosidad"]) if pd.notna(row.get("Morosidad")) else None,
                "Ratio_ROE_Mora": float(row["Ratio_ROE_Mora"]) if pd.notna(row.get("Ratio_ROE_Mora")) else None,
            }
            if "Año" in row.index and pd.notna(row.get("Año")):
                item["Año"] = int(row["Año"])
            datos.append(item)
        return datos
    datos = []
    for _, row in df_res.iterrows():
        item = {
            "banco": row.get("Banco"),
            "indicador": row.get("Indicador") if "Indicador" in row.index else None,
            "valor_pct": round(float(row["Saldo"]), 2) if "Saldo" in row.index and pd.notna(row.get("Saldo")) else None,
        }
        if "Ranking" in row.index and pd.notna(row.get("Ranking")):
            item["Ranking"] = int(row["Ranking"])
        if "Año" in row.index and pd.notna(row.get("Año")):
            item["anio"] = int(row["Año"])
        datos.append(item)
    return datos

def extraer_resultado(df_res):
    if df_res is None or df_res.empty or "Banco" not in df_res.columns:
        return {}

    row = None
    if "Ranking" in df_res.columns:
        try:
            row = df_res.sort_values("Ranking").iloc[0]
        except Exception:
            row = None
    if row is None and "Ratio_ROE_Mora" in df_res.columns:
        try:
            row = df_res.sort_values("Ratio_ROE_Mora", ascending=False).iloc[0]
        except Exception:
            row = None
    if row is None and "Score_ROE_Capital" in df_res.columns:
        try:
            row = df_res.sort_values("Score_ROE_Capital", ascending=False).iloc[0]
        except Exception:
            row = None
    if row is None and "Score_Triple" in df_res.columns:
        try:
            row = df_res.sort_values("Score_Triple", ascending=False).iloc[0]
        except Exception:
            row = None
    if row is None:
        row = df_res.iloc[0]

    banco = row.get("Banco")
    if pd.isna(banco):
        return {}
    if str(banco).strip().upper() in {"SISTEMA", "BANCOS", "HONDURAS", "SISTEMA BANCARIO"}:
        for _, alt in df_res.iterrows():
            b2 = alt.get("Banco")
            if pd.notna(b2) and str(b2).strip().upper() not in {"SISTEMA", "BANCOS", "HONDURAS", "SISTEMA BANCARIO"}:
                row = alt
                banco = b2
                break
        else:
            return {}

    def _num(v, nd=2):
        if pd.isna(v):
            return None
        try:
            return round(float(v), nd)
        except Exception:
            return None

    def _int(v):
        if pd.isna(v):
            return None
        try:
            return int(v)
        except Exception:
            return None

    resultado = {
        "ganador": str(banco),
        "ranking": _int(row.get("Ranking")) if "Ranking" in row.index else 1,
        "ratio": _num(row.get("Ratio_ROE_Mora")) if "Ratio_ROE_Mora" in row.index else None,
        "score_roe_capital": _num(row.get("Score_ROE_Capital")) if "Score_ROE_Capital" in row.index else None,
        "score_triple": _num(row.get("Score_Triple")) if "Score_Triple" in row.index else None,
        "capital": _num(row.get("Capital")) if "Capital" in row.index else None,
        "roe": _num(row.get("ROE")) if "ROE" in row.index else None,
        "morosidad": _num(row.get("Morosidad")) if "Morosidad" in row.index else None,
        "anio": _int(row.get("Año")) if "Año" in row.index else None,
        "saldo": _num(row.get("Saldo")) if "Saldo" in row.index else None,
        "indicador": str(row["Indicador"]) if "Indicador" in row.index and pd.notna(row.get("Indicador")) else None,
    }
    try:
        resultado["bancos_en_resultado"] = sorted({str(b) for b in df_res["Banco"].dropna().unique().tolist()})
    except Exception:
        resultado["bancos_en_resultado"] = [resultado["ganador"]]
    return {k: v for k, v in resultado.items() if v is not None or k in ("ganador",)}

def extraer_ganador(df_res):
    r = extraer_resultado(df_res)
    return r.get("ganador")

def extraer_metricas_ganador(df_res, ganador):
    r = extraer_resultado(df_res)
    if not r or (ganador and r.get("ganador") != ganador):
        if ganador:
            r = extraer_resultado(df_res)
        else:
            return {}
    out = {"Banco": r.get("ganador")}
    if "ranking" in r:
        out["Ranking"] = r["ranking"]
    if "roe" in r:
        out["ROE"] = r["roe"]
    if "morosidad" in r:
        out["Morosidad"] = r["morosidad"]
    if "ratio" in r:
        out["Ratio_ROE_Mora"] = r["ratio"]
    if "anio" in r:
        out["Año"] = r["anio"]
    if "saldo" in r:
        out["Saldo"] = r["saldo"]
    if "indicador" in r:
        out["Indicador"] = r["indicador"]
    return out

def construir_contexto_llm(query, df_res, meta_info, contexto_texto="", resultado=None):
    datos = preparar_datos_para_redactor(df_res)
    if resultado is None:
        resultado = extraer_resultado(df_res)
    ganador = resultado.get("ganador") if resultado else None

    bloque_ganador = ""
    if ganador:
        extra_score = ""
        if resultado and resultado.get("score_triple") is not None:
            extra_score = (
                f"- Score_Triple del ganador: {resultado.get('score_triple')}\n"
                f"- Capital del ganador: {resultado.get('capital')}\n"
                f"- Justifica SIEMPRE con Score_Triple (y Capital), no con otro ratio.\n"
            )
        bloque_ganador = f"""
RESULTADO DETERMINÍSTICO DE PANDAS (INMUTABLE):
{json.dumps(resultado, ensure_ascii=False)}

- Banco ganador: {ganador}
{extra_score}- Si la pregunta pide "mejor", "lidera", "equilibrado" o "mejor relación":
  la única respuesta correcta es {ganador}.
- NO propongas otro banco como ganador.
- NO digas que otro banco "presenta el perfil más equilibrado" ni "lidera".
"""

    bloque_ranking = ""
    if isinstance(datos, list) and datos and "Ranking" in datos[0]:
        cols_disp = list(datos[0].keys())
        cols_txt = " | ".join(str(c) for c in cols_disp)
        tiene_score = "Score_Triple" in datos[0]
        if tiene_score:
            bloque_ranking = f"""
REGLAS DE RANKING — EQUILIBRIO TRIPLE (OBLIGATORIAS):
- Pandas ordenó por Score_Triple = (ROE/Morosidad)*(Capital/100). Ranking=1 es el ganador.
- Columnas en DATOS: {cols_txt}
- La tabla DEBE usar exactamente: Ranking | Banco | ROE | Morosidad | Capital | Score_Triple
- NO inventes ni menciones Ratio_ROE_Mora como criterio de este ranking.
- Justifica el #1 con Score_Triple y Capital de DATOS (no solo con ROE o mora).
- Para bancos siguientes: cita un punto débil usando solo cifras de DATOS (p. ej. menor capital o mayor mora).
- NUNCA reordenes ni cambies el ganador.
"""
        else:
            bloque_ranking = f"""
REGLAS DE RANKING (OBLIGATORIAS):
- El DataFrame YA viene ordenado por Pandas con columna Ranking.
- NUNCA cambies el orden ni el ganador (Ranking = 1).
- Columnas disponibles: {cols_txt}
- Copia la tabla con exactamente esas columnas; no inventes cifras.
"""

    return f"""Eres un redactor financiero especializado en el sistema bancario de Honduras (CNBS).

IMPORTANTE — GOBIERNO DE CÁLCULOS:
Los cálculos ya fueron realizados por Pandas.
NO recalcules.
NO cambies rankings.
NO cambies el banco ganador.
NO inventes indicadores ni cifras.
NO deduzcas un ganador distinto al indicado por Pandas.
Tu única tarea: resumir, explicar y redactar (máximo ~150 palabras de prosa + tabla si aplica).

CONSULTA DEL USUARIO:
{query}

CONTEXTO DE SESIÓN:
{contexto_texto or "(ninguno)"}

META:
{json.dumps(meta_info, ensure_ascii=False, default=str)}
{bloque_ganador}
{bloque_ranking}
DATOS EXACTOS CALCULADOS POR PANDAS ({len(datos)} filas):
{json.dumps(datos, ensure_ascii=False)}

REGLAS DE REDACCIÓN:
1. Usa EXCLUSIVAMENTE los números de DATOS.
2. Si un indicador/banco aparece en DATOS, no digas que faltan datos de ese ítem.
3. No inventes causas macroeconómicas ni recomendaciones regulatorias de inversión.
4. Si piden mayor riesgo: prioriza mayor morosidad; cita cobertura/tarjetas del mismo banco si están en DATOS.
5. Markdown, 2 decimales, %. Preferir UNA tabla (filas=Banco).
6. Conclusión breve (2-4 líneas) solo con números de DATOS.
7. Si META indica modo_corto, usa viñetas compactas.
"""

def construir_prompt(query, df_res, meta_info, contexto_texto):
    return construir_contexto_llm(query, df_res, meta_info, contexto_texto)

def validar_respuesta_llm(respuesta, resultado):
    if not resultado or not resultado.get("ganador"):
        return True
    if not respuesta or str(respuesta).startswith("⚠️"):
        return False

    ganador = resultado["ganador"]
    r = normalizar_texto(respuesta)
    g = normalizar_texto(ganador)
    tokens_g = [t for t in g.replace("-", " ").split() if len(t) >= 3]
    if tokens_g:
        if not any(t in r for t in tokens_g):
            return False
    elif g not in r:
        return False

    frases_victoria = (
        "presenta el perfil mas equilibrado",
        "presenta el mejor perfil",
        "tiene el mejor perfil",
        "es el mas equilibrado",
        "es el mejor",
        "ocupa el primer lugar",
        "ocupa el puesto 1",
        "lidera el ranking",
        "lidera con",
        "mejor relacion rentabilidad",
        "mejor ratio",
        "ranking 1",
        "ranking #1",
        "puesto #1",
        "el #1 es",
        "el numero 1 es",
    )
    otros = resultado.get("bancos_en_resultado") or []
    for banco in otros:
        if normalizar_texto(banco) == g:
            continue
        bn = normalizar_texto(banco)
        tokens_b = [t for t in bn.replace("-", " ").split() if len(t) >= 4]
        if not tokens_b:
            continue
        for frase in frases_victoria:
            for tok in tokens_b:
                if tok in r and frase in r:
                    idx_f = r.find(frase)
                    idx_b = r.find(tok)
                    if idx_f >= 0 and idx_b >= 0 and abs(idx_f - idx_b) < 80:
                        idx_g = min((r.find(t) for t in tokens_g if r.find(t) >= 0), default=-1)
                        if idx_g < 0 or abs(idx_f - idx_b) < abs(idx_f - idx_g):
                            return False
    return True

def _langsmith_config(run_name, resultado=None, extra=None):
    meta = {
        "app": "Agente-CNBS",
        "version": "6.3",
        "motor": "Pandas+LLM",
    }
    if resultado:
        if resultado.get("ganador"):
            meta["ganador"] = resultado["ganador"]
        if resultado.get("ranking") is not None:
            meta["ranking"] = resultado["ranking"]
        if resultado.get("ratio") is not None:
            meta["ratio"] = resultado["ratio"]
        if resultado.get("anio") is not None:
            meta["anio"] = resultado["anio"]
    if extra:
        meta.update(extra)
    return {
        "run_name": run_name,
        "tags": ["cnbs", "redactor", "gobernado"],
        "metadata": meta,
    }

def redactar_respuesta(llm, prompt, resultado=None, ganador=None):
    if resultado is None and ganador:
        resultado = {"ganador": ganador, "bancos_en_resultado": [ganador]}
    meta_g = {
        "validacion": True,
        "reintentos": 0,
        "ganador": (resultado or {}).get("ganador"),
        "langsmith": os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
    }
    cfg = _langsmith_config("cnbs-redactor", resultado, {"intento": 1})
    try:
        r = llm.invoke(prompt, config=cfg)
        texto = r.content if hasattr(r, "content") else str(r)
    except TypeError:
        try:
            r = llm.invoke(prompt)
            texto = r.content if hasattr(r, "content") else str(r)
        except Exception as e:
            meta_g["validacion"] = False
            return f"⚠️ Error al redactar: {e}", meta_g
    except Exception as e:
        meta_g["validacion"] = False
        return f"⚠️ Error al redactar: {e}", meta_g

    if not resultado or not resultado.get("ganador") or validar_respuesta_llm(texto, resultado):
        meta_g["validacion"] = True
        return texto, meta_g

    meta_g["reintentos"] = 1
    g = resultado["ganador"]
    correccion = (
        prompt
        + "\n\nCORRECCIÓN OBLIGATORIA:\n"
        + "Tu respuesta anterior no respetó el resultado determinado por Pandas.\n"
        + f"El único ganador válido es: {g}.\n"
        + f"Reescribe mencionando explícitamente a {g} como #1 / mejor perfil.\n"
        + "No digas que otro banco presenta el perfil más equilibrado ni que lidera.\n"
    )
    cfg2 = _langsmith_config("cnbs-redactor-reintento", resultado, {"intento": 2})
    try:
        try:
            r2 = llm.invoke(correccion, config=cfg2)
        except TypeError:
            r2 = llm.invoke(correccion)
        texto2 = r2.content if hasattr(r2, "content") else str(r2)
        if validar_respuesta_llm(texto2, resultado):
            meta_g["validacion"] = True
            return texto2, meta_g
        meta_g["validacion"] = False
        texto_safe = (
            f"**Ganador (Pandas): {g}**\n\n"
            + texto2
            + "\n\n_Nota: el ranking y el ganador fueron fijados por el motor determinístico._"
        )
        return texto_safe, meta_g
    except Exception:
        meta_g["validacion"] = False
        return f"**Ganador determinado por Pandas: {g}**\n\n" + texto, meta_g

def formatear_tabla_ranking(df_res):
    if df_res is None or df_res.empty:
        return "No se encontraron datos."
    top = df_res.iloc[0]
    medallas = {1: "🥇", 2: "🥈", 3: "🥉"}
    anio = int(top["Año"]) if "Año" in df_res.columns and pd.notna(top.get("Año")) else ""
    titulo_anio = f" ({anio})" if anio else ""
    lineas = [
        f"### 🏆 Mejor relación Rentabilidad–Riesgo{titulo_anio}",
        "",
        "| | |",
        "|:--|--:|",
        f"| **Banco** | **{top['Banco']}** |",
        f"| **Ratio ROE/Mora** | **{float(top['Ratio_ROE_Mora']):.2f}** |",
        f"| ROE | {float(top['ROE']):.2f}% |",
        f"| Morosidad | {float(top['Morosidad']):.2f}% |",
        "",
        "---",
        "",
        "#### 📊 Ranking completo",
        "",
        "| | # | Banco | ROE % | Mora % | Ratio |",
        "|:--|--:|:------|------:|------:|------:|",
    ]
    for _, r in df_res.iterrows():
        rk = int(r["Ranking"])
        medal = medallas.get(rk, "")
        lineas.append(
            f"| {medal} | {rk} | {r['Banco']} | {float(r['ROE']):.2f} | "
            f"{float(r['Morosidad']):.2f} | {float(r['Ratio_ROE_Mora']):.2f} |"
        )
    return "\n".join(lineas)

def render_respuesta_ui(output, df_res, meta_html):
    tipo = columnas_ranking(df_res)
    if tipo == "triple":
        top = df_res.iloc[0]
        anio = int(top["Año"]) if "Año" in df_res.columns and pd.notna(top.get("Año")) else ""
        sc = float(top["Score_Triple"])
        nivel, emoji = nivel_score(sc)
        st.markdown(
            f'<div class="winner-card">'
            f'<div class="title">🏆 Banco recomendado (equilibrio triple)'
            f'{" · " + str(anio) if anio else ""}</div>'
            f'<div class="bank">{top["Banco"]}</div>'
            f'<div style="margin-top:8px;font-size:0.95rem;">{emoji} Score {sc:.2f} · {nivel}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score triple", f"{sc:.2f}")
        c2.metric("ROE", f"{float(top['ROE']):.2f}%")
        c3.metric("Morosidad", f"{float(top['Morosidad']):.2f}%")
        c4.metric("Capital", f"{float(top['Capital']):.2f}%")
        st.markdown("---")
        st.markdown("#### 📊 Ranking completo")
        show = df_res[["Ranking", "Banco", "ROE", "Morosidad", "Capital", "Score_Triple"]].copy()
        show["Nivel"] = show["Score_Triple"].map(lambda v: f"{nivel_score(v)[1]} {nivel_score(v)[0]}")
        show = show.rename(columns={
            "Ranking": "#", "Morosidad": "Mora %", "ROE": "ROE %",
            "Capital": "Capital %", "Score_Triple": "Score",
        })
        medallas = {1: "🥇", 2: "🥈", 3: "🥉"}
        show.insert(0, "", show["#"].map(lambda x: medallas.get(int(x), "")))
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            height=min(420, 38 * len(show) + 40),
        )
        st.caption(f"Fórmula: {Config.FORMULA_SCORE_TRIPLE}")
        st.markdown(conclusion_ranking(df_res, "triple"))
    elif tipo == "roe_mora":
        top = df_res.iloc[0]
        anio = int(top["Año"]) if "Año" in df_res.columns and pd.notna(top.get("Año")) else ""
        st.markdown(
            f'<div class="winner-card">'
            f'<div class="title">🏆 Mejor relación Rentabilidad–Riesgo'
            f'{" · " + str(anio) if anio else ""}</div>'
            f'<div class="bank">{top["Banco"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Ratio ROE/Mora", f"{float(top['Ratio_ROE_Mora']):.2f}")
        c2.metric("ROE", f"{float(top['ROE']):.2f}%")
        c3.metric("Morosidad", f"{float(top['Morosidad']):.2f}%")
        st.markdown("---")
        st.markdown("#### 📊 Ranking completo")
        show = df_res[["Ranking", "Banco", "ROE", "Morosidad", "Ratio_ROE_Mora"]].copy()
        show = show.rename(columns={
            "Ranking": "#", "Morosidad": "Mora %", "ROE": "ROE %", "Ratio_ROE_Mora": "Ratio",
        })
        medallas = {1: "🥇", 2: "🥈", 3: "🥉"}
        show.insert(0, "", show["#"].map(lambda x: medallas.get(int(x), "")))
        st.dataframe(
            show, use_container_width=True, hide_index=True,
            height=min(420, 38 * len(show) + 40),
        )
        st.markdown(conclusion_ranking(df_res, "roe_mora"))
    elif (
        df_res is not None and not df_res.empty
        and "Indicador" in df_res.columns
        and "Banco" in df_res.columns
        and df_res["Banco"].nunique() >= 3
    ):
        st.markdown(output)
        piv = pivot_resultados(df_res)
        if piv is not None and not piv.empty:
            st.dataframe(piv.reset_index(), use_container_width=True, hide_index=True)
    else:
        st.markdown(output)
    st.markdown(meta_html, unsafe_allow_html=True)

def formatear_sistema_corto(df_res, anios=None):
    titulo = "Sistema bancario"
    if anios:
        titulo += f" {', '.join(map(str, anios))}"
    lineas = [f"**{titulo}**", ""]
    if "Indicador" in df_res.columns:
        for _, row in df_res.iterrows():
            ind = str(row.get("Indicador", ""))
            if "MOROSIDAD" in ind.upper() and "TOTAL" in ind.upper():
                label = "Morosidad"
            elif "COBERTURA" in ind.upper() and "MORA" in ind.upper():
                label = "Cobertura de mora"
            elif "TARJETA" in ind.upper():
                label = "Cartera de tarjetas"
            elif "ROA" in ind.upper():
                label = "ROA"
            elif "ROE" in ind.upper():
                label = "ROE"
            else:
                label = ind[:40]
            anio = f" ({int(row['Año'])})" if "Año" in row.index and pd.notna(row.get("Año")) else ""
            lineas.append(f"- **{label}{anio}:** {row['Saldo']:.2f}%")
        lineas.append("")
        lineas.append(f"_Indicadores analizados: {df_res['Indicador'].nunique()}_")
    return "\n".join(lineas)

def etiqueta_corta_indicador(nombre):
    u = str(nombre).upper()
    if "ROA" in u and "ROE" not in u:
        return "ROA"
    if "ROE" in u:
        return "ROE"
    if "MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL" in u:
        return "Mora"
    if "MOROSIDAD SOBRE CARTERA CREDITICIA DIRECTA" in u:
        return "Mora directa"
    if "COBERTURA DE LA MORA" in u:
        return "Cobertura"
    if "TARJETAS" in u:
        return "Tarjetas"
    if "ADECUACIÓN DE CAPITAL" in u or "ADECUACION DE CAPITAL" in u:
        return "Capital"
    if "SPREAD" in u:
        return "Spread"
    if "TASA ACTIVA" in u:
        return "Tasa activa"
    if "TASA PASIVA" in u:
        return "Tasa pasiva"
    if "LIQUIDEZ" in u:
        return "Liquidez"
    return str(nombre)[:18]

def pivot_resultados(df_res):
    if df_res is None or df_res.empty or "Indicador" not in df_res.columns:
        return None
    d = df_res.copy()
    d["Ind"] = d["Indicador"].map(etiqueta_corta_indicador)
    if "Año" in d.columns and d["Año"].nunique() > 1:
        d["Ind"] = d["Ind"] + " " + d["Año"].astype(int).astype(str)
    piv = d.pivot_table(index="Banco", columns="Ind", values="Saldo", aggfunc="mean")
    return piv.round(2)

def df_a_markdown(df_show):
    headers = list(df_show.columns)

    def _cell(v):
        if pd.isna(v):
            return ""
        if isinstance(v, (float, int)) and not isinstance(v, bool):
            return f"{float(v):.2f}"
        return str(v)

    lines = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for _, row in df_show.iterrows():
        lines.append("| " + " | ".join(_cell(row[h]) for h in headers) + " |")
    return "\n".join(lines)

def formatear_tabla_comparacion(df_res, query=None):
    piv = pivot_resultados(df_res)
    if piv is None or piv.empty:
        return None
    prefer = ["ROA", "ROE", "Mora", "Mora directa", "Capital", "Cobertura", "Tarjetas", "Spread"]
    cols = list(piv.columns)
    ordered = [c for c in prefer if c in cols] + [c for c in cols if c not in prefer]
    piv = piv.reindex(columns=ordered)
    tabla = df_a_markdown(piv.reset_index())
    texto = "**Comparación**\n\n" + tabla
    q = normalizar_texto(query or "")
    if any(p in q for p in ("equilibrado", "mejor relacion", "rentabilidad-riesgo", "rentabilidad riesgo", "equilibrio")):
        if "ROE" in piv.columns and "Mora" in piv.columns and "Capital" in piv.columns:
            tmp = piv[(piv["Mora"] > 0) & (piv["Capital"].notna())].copy()
            if not tmp.empty:
                tmp["Score_Triple"] = (tmp["ROE"] / tmp["Mora"]) * (tmp["Capital"] / 100.0)
                best = tmp["Score_Triple"].idxmax()
                sc = float(tmp.loc[best, "Score_Triple"])
                rm = float(tmp.loc[best, "ROE"] / tmp.loc[best, "Mora"])
                texto += (
                    "\n\n**Conclusión:** **" + str(best) + "** presenta el perfil más equilibrado "
                    "(Score = (ROE/Mora)×(Capital/100) = " + f"{sc:.2f}"
                    + f"; ROE/Mora = {rm:.2f})."
                )
        elif "ROE" in piv.columns and "Mora" in piv.columns:
            tmp = piv[piv["Mora"] > 0].copy()
            if not tmp.empty:
                tmp["Ratio"] = tmp["ROE"] / tmp["Mora"]
                best = tmp["Ratio"].idxmax()
                ratio = float(tmp.loc[best, "Ratio"])
                texto += (
                    "\n\n**Conclusión:** **" + str(best) + "** presenta el perfil más equilibrado "
                    "(Ratio ROE/Mora = " + f"{ratio:.2f}" + ")."
                )
    if any(p in q for p in ("riesgo", "preocupante", "morosidad", "calidad de activos")) and "Mora" in piv.columns:
        mora_s = piv["Mora"].dropna().sort_values(ascending=False)
        if not mora_s.empty:
            worst = mora_s.index[0]
            line = f"\n\n**Mayor morosidad:** **{worst}** ({mora_s.iloc[0]:.2f}%)"
            if "Cobertura" in piv.columns and worst in piv.index and pd.notna(piv.loc[worst, "Cobertura"]):
                line += f" · Cobertura {float(piv.loc[worst, 'Cobertura']):.2f}%"
            if "Tarjetas" in piv.columns and worst in piv.index and pd.notna(piv.loc[worst, "Tarjetas"]):
                line += f" · Tarjetas {float(piv.loc[worst, 'Tarjetas']):.2f}%"
            top3 = ", ".join(f"{b} ({v:.2f}%)" for b, v in mora_s.head(3).items())
            line += f"\n**Top 3 morosidad:** {top3}"
            texto += line
    return texto

def formatear_tabla_ranking_triple(df_res):
    if df_res is None or df_res.empty:
        return "No se encontraron datos."
    top = df_res.iloc[0]
    medallas = {1: "🥇", 2: "🥈", 3: "🥉"}
    anio = int(top["Año"]) if "Año" in df_res.columns and pd.notna(top.get("Año")) else ""
    titulo_anio = f" ({anio})" if anio else ""
    sc = float(top["Score_Triple"])
    nivel, emoji = nivel_score(sc)
    lineas = [
        f"### 🏆 Mejor equilibrio rentabilidad–riesgo–solvencia{titulo_anio}",
        "",
        f"**Banco recomendado:** **{top['Banco']}**",
        f"**Score triple:** {emoji} **{sc:.2f}** ({nivel})",
        f"- ROE/Mora: {float(top['Ratio_ROE_Mora']):.2f}",
        f"- ROE: {float(top['ROE']):.2f}%",
        f"- Morosidad: {float(top['Morosidad']):.2f}%",
        f"- Capital: {float(top['Capital']):.2f}%",
        "",
        "---",
        "",
        "#### 📊 Ranking completo",
        "",
        "| | # | Banco | ROE % | Mora % | Capital % | Score | Nivel |",
        "|:--|--:|:------|------:|------:|----------:|------:|:-----|",
    ]
    for _, r in df_res.iterrows():
        rk = int(r["Ranking"])
        medal = medallas.get(rk, "")
        sc_i = float(r["Score_Triple"])
        niv, em = nivel_score(sc_i)
        lineas.append(
            f"| {medal} | {rk} | {r['Banco']} | {float(r['ROE']):.2f} | "
            f"{float(r['Morosidad']):.2f} | {float(r['Capital']):.2f} | {sc_i:.2f} | {em} {niv} |"
        )
    lineas.append("")
    lineas.append(f"_Fórmula: Score = {Config.FORMULA_SCORE_TRIPLE}. Cálculo Pandas determinístico._")
    lineas.append("")
    lineas.append(conclusion_ranking(df_res, "triple"))
    return "\n".join(lineas)

def responder_directamente(df_res, query, meta_info):
    if df_res is None or df_res.empty:
        return "No se encontraron datos."

    if "Ranking" in df_res.columns and "Score_Triple" in df_res.columns:
        return formatear_tabla_ranking_triple(df_res)
    if "Ranking" in df_res.columns and "Ratio_ROE_Mora" in df_res.columns:
        return formatear_tabla_ranking(df_res)

    if consulta_simple_sistema(query, df_res):
        return formatear_sistema_corto(df_res, meta_info.get("anios"))

    if "Indicador" in df_res.columns and (
        df_res["Indicador"].nunique() >= 2
        or ("Banco" in df_res.columns and df_res["Banco"].nunique() >= 2)
        or ("Año" in df_res.columns and df_res["Año"].nunique() > 1)
    ):
        tab = formatear_tabla_comparacion(df_res, query)
        if tab:
            return tab

    if "Indicador" in df_res.columns and df_res["Indicador"].nunique() == 1:
        ind = etiqueta_corta_indicador(df_res["Indicador"].iloc[0])
        orden = df_res.sort_values("Saldo", ascending=False).reset_index(drop=True)
        lineas = [f"**{ind}**\n", "| # | Banco | Valor % |", "|--:|:------|--------:|"]
        for i, (_, row) in enumerate(orden.iterrows(), 1):
            lineas.append(f"| {i} | {row['Banco']} | {row['Saldo']:.2f} |")
        return "\n".join(lineas)

    return df_a_markdown(df_res) if hasattr(df_res, "iterrows") else str(df_res)

# ============================================================
# EXPORTACIONES (PDF)
# ============================================================
try:
    from pdf_renderer import generar_pdf_respuesta, get_pdf_renderer, PDFRenderer
except ImportError:
    def generar_pdf_respuesta(consulta, respuesta="", meta="", titulo="Informe Agente CNBS", **kwargs):
        from datetime import datetime
        text = f"{titulo}\n{datetime.now()}\n\nCONSULTA\n{consulta}\n\nRESPUESTA\n{respuesta}\n\n{meta}"
        content = text.encode("latin-1", errors="replace")
        return b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n" + content

    class PDFRenderer:
        def render(self, **kwargs):
            return generar_pdf_respuesta(
                kwargs.get("consulta", ""),
                kwargs.get("respuesta", ""),
                kwargs.get("metadata", ""),
            )

    def get_pdf_renderer():
        return PDFRenderer()

def fig_to_png_bytes(fig, titulo=None, ylabel="Valor (%)"):
    try:
        return fig.to_image(format="png", width=1100, height=520, scale=2)
    except Exception:
        pass
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=1100, height=520)
    except Exception:
        pass
    try:
        from io import BytesIO
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        MESES = {
            1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
        }
        fig_m, ax = plt.subplots(figsize=(11, 5.2), dpi=140)
        palette = ["#2563EB", "#F97316", "#16A34A", "#DC2626", "#7C3AED", "#0891B2"]
        i_c = 0
        for tr in fig.data:
            name = getattr(tr, "name", None) or ""
            x = list(tr.x) if tr.x is not None else []
            y = list(tr.y) if tr.y is not None else []
            if not (len(x) and len(y)):
                continue
            color = palette[i_c % len(palette)]
            i_c += 1
            if getattr(tr, "type", "") == "bar" or (
                hasattr(tr, "orientation") and tr.orientation == "h"
            ):
                ax.barh(x, y, label=name, color=color)
            else:
                try:
                    xd = pd.to_datetime(x)
                except Exception:
                    xd = x
                ax.plot(xd, y, marker="o", label=name, linewidth=2, markersize=5, color=color)
                if len(y):
                    ax.annotate(
                        f"{float(y[-1]):.2f}%",
                        (xd[-1], y[-1]),
                        textcoords="offset points",
                        xytext=(6, 6),
                        fontsize=8,
                        color=color,
                        fontweight="bold",
                    )
        titulo_final = titulo or (fig.layout.title.text if fig.layout.title and fig.layout.title.text else None)
        if titulo_final:
            ax.set_title(titulo_final, fontsize=12, fontweight="bold", color="#0F3D91", pad=10)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("")
        ax.legend(loc="best", fontsize=8, frameon=True)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
            fig_m.autofmt_xdate(rotation=0, ha="center")
            labels = []
            for t in ax.get_xticks():
                try:
                    dt = mdates.num2date(t)
                    labels.append(f"{MESES.get(dt.month, dt.strftime('%b'))}-{str(dt.year)[2:]}")
                except Exception:
                    labels.append("")
            if len(labels) <= 16:
                ax.set_xticklabels(labels, fontsize=8)
        except Exception:
            pass
        ax.tick_params(axis="y", labelsize=8)
        fig_m.tight_layout()
        buf = BytesIO()
        fig_m.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        plt.close(fig_m)
        return buf.getvalue()
    except Exception:
        return None

def hallazgos_desde_serie(serie, indicador_label, f_ini=None, f_fin=None):
    if serie is None or getattr(serie, "empty", True):
        return ""
    s = serie.copy()
    n_obs = len(s)
    n_bancos = int(s["Banco"].nunique()) if "Banco" in s.columns else 0
    prom = float(s["Saldo"].mean())
    vmax = float(s["Saldo"].max())
    vmin = float(s["Saldo"].min())
    periodo = ""
    if f_ini and f_fin:
        periodo = f" entre {f_ini} y {f_fin}"
    elif "FechaReporte" in s.columns and s["FechaReporte"].notna().any():
        f0 = pd.Timestamp(s["FechaReporte"].min()).strftime("%m/%Y")
        f1 = pd.Timestamp(s["FechaReporte"].max()).strftime("%m/%Y")
        periodo = f" entre {f0} y {f1}"

    lineas = [
        f"Se analizaron {n_obs} observaciones correspondientes a {n_bancos} "
        f"banco(s){periodo} para el indicador {indicador_label}.",
        f"La media del periodo fue {prom:.2f}%.",
        f"El valor máximo observado alcanzó {vmax:.2f}%.",
        f"El valor mínimo observado fue {vmin:.2f}%.",
    ]

    if "Banco" in s.columns and n_bancos >= 1:
        por_banco = s.groupby("Banco")["Saldo"].mean().sort_values(ascending=False)
        b_max, v_max_b = por_banco.index[0], float(por_banco.iloc[0])
        b_min, v_min_b = por_banco.index[-1], float(por_banco.iloc[-1])
        lineas.append(f"Mayor nivel promedio: {b_max} ({v_max_b:.2f}%).")
        if n_bancos >= 2:
            lineas.append(f"Menor nivel promedio: {b_min} ({v_min_b:.2f}%).")
        if "FechaReporte" in s.columns:
            ultimo = s["FechaReporte"].max()
            ult = s[s["FechaReporte"] == ultimo].groupby("Banco")["Saldo"].mean().sort_values(ascending=False)
            if not ult.empty:
                lineas.append(
                    f"En el último periodo ({pd.Timestamp(ultimo).strftime('%m/%Y')}), "
                    f"lidera {ult.index[0]} con {float(ult.iloc[0]):.2f}%."
                )

    if n_bancos >= 2:
        lineas.append(
            "La dispersión entre bancos evidencia diferencias en el nivel del indicador "
            "a lo largo del periodo analizado."
        )
    return "\n".join(f"• {x}" for x in lineas)

def generar_pdf_tendencia(titulo, kpis_texto, png_bytes=None, metadata=None, hallazgos=None):
    from io import BytesIO
    from datetime import datetime

    try:
        from pdf_renderer import get_pdf_renderer
        meta = metadata or {
            "Tipo": "Tendencias",
            "Fuente": "CNBS Honduras",
            "Motor": "Plotly + Pandas",
        }
        summary = hallazgos or None
        respuesta = kpis_texto or ""
        if hallazgos and kpis_texto:
            respuesta = "INDICADORES CLAVE\n" + kpis_texto
        return get_pdf_renderer().render(
            consulta=titulo,
            respuesta=respuesta,
            figure=png_bytes,
            metadata=meta,
            summary=summary,
            titulo="TENDENCIAS CNBS",
        )
    except Exception:
        pass

    lines = [titulo, "", kpis_texto or "", "", "Fuente: CNBS Honduras"]
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable, Table, TableStyle,
        )
        from reportlab.lib.colors import HexColor

        buf = BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=letter,
            leftMargin=1.5 * cm, rightMargin=1.5 * cm,
            topMargin=1.3 * cm, bottomMargin=1.8 * cm,
        )
        styles = getSampleStyleSheet()
        body = ParagraphStyle("bodyt", parent=styles["Normal"], fontSize=9, leading=12)

        def esc(t):
            return (
                str(t or "")
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace("\n", "<br/>")
            )

        story = []
        story.append(
            Paragraph(
                "<font size='18' color='#0F3D91'><b>TENDENCIAS CNBS</b></font>",
                styles["Title"],
            )
        )
        story.append(
            Paragraph(
                "<font size='10' color='#475569'>Agente Financiero CNBS · Análisis temporal</font>",
                styles["BodyText"],
            )
        )
        story.append(
            Paragraph(
                f"<font size='9' color='#64748B'>Generado · {datetime.now():%d/%m/%Y %H:%M}</font>",
                styles["BodyText"],
            )
        )
        story.append(Spacer(1, 8))
        story.append(HRFlowable(width="100%", thickness=1.3, color=HexColor("#1D4ED8")))
        story.append(Spacer(1, 10))

        story.append(
            Paragraph("<font color='#1E3A8A'><b>INDICADOR</b></font>", styles["Heading2"])
        )
        story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#BFDBFE")))
        box = Table([[Paragraph(esc(titulo), body)]], colWidths=[470])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), HexColor("#EFF6FF")),
            ("BOX", (0, 0), (-1, -1), 1, HexColor("#93C5FD")),
            ("LEFTPADDING", (0, 0), (-1, -1), 12),
            ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ]))
        story.append(box)
        story.append(Spacer(1, 10))

        if kpis_texto:
            story.append(
                Paragraph("<font color='#1E3A8A'><b>INDICADORES CLAVE</b></font>", styles["Heading2"])
            )
            story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#BFDBFE")))
            kpi_box = Table([[Paragraph(esc(kpis_texto), body)]], colWidths=[470])
            kpi_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(kpi_box)
            story.append(Spacer(1, 10))

        if hallazgos:
            story.append(
                Paragraph("<font color='#1E3A8A'><b>HALLAZGOS PRINCIPALES</b></font>", styles["Heading2"])
            )
            story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#BFDBFE")))
            hall_box = Table([[Paragraph(esc(hallazgos), body)]], colWidths=[470])
            hall_box.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), HexColor("#ECFDF5")),
                ("BOX", (0, 0), (-1, -1), 1, HexColor("#10B981")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]))
            story.append(hall_box)
            story.append(Spacer(1, 10))

        if png_bytes:
            story.append(
                Paragraph("<font color='#1E3A8A'><b>GRÁFICO</b></font>", styles["Heading2"])
            )
            story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#BFDBFE")))
            story.append(Spacer(1, 6))
            try:
                story.append(Image(BytesIO(png_bytes), width=470, height=250))
            except Exception:
                story.append(Paragraph("<i>Gráfico no disponible.</i>", body))
            story.append(Spacer(1, 10))

        story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#94A3B8")))
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                "<font color='#64748B'>"
                "Fuente: Comisión Nacional de Bancos y Seguros (CNBS)<br/>"
                "Motor: Plotly + Pandas · Agente Financiero CNBS v6.3"
                "</font>",
                styles["Italic"],
            )
        )
        doc.build(story)
        return buf.getvalue()
    except Exception:
        try:
            from pdf_renderer import _pdf_stdlib
            return _pdf_stdlib(lines, "TENDENCIAS CNBS")
        except Exception:
            return b"%PDF-1.4\n%%EOF\n"

def df_to_excel_bytes(df_export):
    from io import BytesIO
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_export.to_excel(writer, index=False, sheet_name="CNBS")
        return buf.getvalue()
    except Exception:
        return df_export.to_csv(index=False).encode("utf-8-sig")

# ============================================================
# STREAMLIT APP
# ============================================================
st.set_page_config(
    page_title="Agente Financiero CNBS",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
section[data-testid="stSidebar"] {
    width: 320px !important;
    min-width: 320px !important;
}
section[data-testid="stSidebar"] .side-card {
    background: rgba(128,128,128,0.12);
    border: 1px solid rgba(128,128,128,0.25);
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
section[data-testid="stSidebar"] .side-card .lbl {
    font-size: 11px; opacity: 0.8; text-transform: uppercase; letter-spacing: 0.3px;
}
section[data-testid="stSidebar"] .side-card .val {
    font-size: 15px; font-weight: 600; margin-top: 3px; word-break: break-word;
}
div[data-testid="stMetric"] {
    background: rgba(128,128,128,0.06);
    border: 1px solid rgba(128,128,128,0.15);
    border-radius: 10px;
    padding: 10px 12px;
}
.block-card {
    border: 1px solid rgba(128,128,128,0.18);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 8px 0 14px 0;
}
.welcome-box {
    border: 1px solid rgba(25,118,210,0.25);
    background: rgba(25,118,210,0.06);
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 12px;
}
.footer-meta {
    font-size: 12.5px;
    color: #666;
    border-top: 1px solid rgba(128,128,128,0.2);
    padding-top: 8px;
    margin-top: 10px;
}
.chip {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
}
.chip-alta { background:#e8f5e9; color:#2e7d32; border:1px solid #a5d6a7; }
.chip-media { background:#fff3e0; color:#ef6c00; border:1px solid #ffcc80; }
.chip-baja { background:#ffebee; color:#c62828; border:1px solid #ef9a9a; }
.header-line {
    font-size: 13.5px;
    opacity: 0.85;
    margin: 2px 0 12px 0;
}
.chat-card {
    border: 1px solid rgba(128,128,128,0.22);
    border-radius: 12px;
    padding: 14px 16px;
    margin: 6px 0 12px 0;
    background: rgba(128,128,128,0.05);
}
.suggest-grid button {
    min-height: 52px;
    font-weight: 600;
}
.winner-card {
    border: 1px solid rgba(46,125,50,0.35);
    background: rgba(46,125,50,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
}
.winner-card .title { font-size: 13px; text-transform: uppercase; letter-spacing: 0.4px; opacity: 0.8; margin-bottom: 6px; }
.winner-card .bank { font-size: 22px; font-weight: 700; color: #2e7d32; margin-bottom: 4px; }
.meta-bar {
    border-top: 1px solid rgba(128,128,128,0.25);
    margin-top: 12px; padding-top: 10px;
    font-size: 12.5px; opacity: 0.9; line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ---- Session ----
for k, v in {
    "messages": [], "query_count": 0, "total_time": 0.0,
    "last_time": 0.0, "last_query": "Ninguna",
    "theme_count": 0, "direct_count": 0, "pending_query": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v
if "contexto" not in st.session_state:
    st.session_state.contexto = ContextoConversacional()

# ============================================================
# CARGA DE DATOS (función definida aquí)
# ============================================================
@st.cache_data
def cargar_datos():
    try:
        df = pd.read_csv("indicadores_financieros_CNBS.csv", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv("indicadores_financieros_CNBS.csv", encoding="latin-1")
    df.rename(columns={df.columns[0]: "Banco"}, inplace=True)
    df["Banco"] = df["Banco"].astype(str).str.strip()
    df["Indicador"] = df["Indicador"].astype(str).str.strip()
    fechas = df["FechaReporte"].astype(str).str.strip()
    df["FechaReporte"] = pd.to_datetime(fechas, errors="coerce")
    m = df["FechaReporte"].isna()
    if m.any():
        df.loc[m, "FechaReporte"] = pd.to_datetime(fechas[m], errors="coerce", format="%m/%Y")
    df["FechaReporte"] = df["FechaReporte"].dt.to_period("M").dt.to_timestamp()
    df["Saldo"] = pd.to_numeric(df["Saldo"].astype(str).str.replace(r"[L\$,\s]", "", regex=True), errors="coerce")
    return df

# ---- Data ----
df = cargar_datos()
ok, msg = validar_dataframe(df)
if not ok:
    st.error(msg)
    st.stop()
catalogo = construir_catalogo(df)
ultima_fecha = df["FechaReporte"].max().strftime("%Y-%m")
min_fecha = df["FechaReporte"].min().strftime("%Y-%m")
total_registros = len(df)
total_meses = int(df["FechaReporte"].nunique())
n_bancos = int(df[~df["Banco"].isin(Config.ENTIDADES_AGREGADAS)]["Banco"].nunique())
n_inds = int(df["Indicador"].nunique())
bancos_lista = sorted(df[~df["Banco"].isin(Config.ENTIDADES_AGREGADAS)]["Banco"].unique().tolist())
inds_lista = sorted(df["Indicador"].dropna().unique().tolist())
promedio_t = (st.session_state.total_time / st.session_state.query_count) if st.session_state.query_count else 0.0

@st.cache_resource
def get_llm(api_key):
    from langchain_groq import ChatGroq
    return ChatGroq(temperature=0, model_name=Config.MODEL_NAME, api_key=api_key, max_retries=2)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🏦 Agente CNBS")
    st.caption("v6.2")
    st.divider()

    st.markdown("""
    <div class="side-card">
      <div class="lbl">🧠 Motor</div>
      <div class="val">Pandas · Groq · Llama 3.3</div>
      <div class="val" style="font-size:13px;font-weight:500;margin-top:6px;">🟢 Operativo</div>
    </div>
    """, unsafe_allow_html=True)

    # ===== GROQ API KEY =====
    st.markdown("**🔐 Groq API Key**")
    if os.environ.get("GROQ_API_KEY"):
        st.success("✅ Configurada desde secrets")
        groq_api_key = os.environ["GROQ_API_KEY"]
    else:
        groq_api_key = st.text_input(
            "Groq key", type="password",
            value="",
            label_visibility="collapsed",
            key="groq_key_input",
            placeholder="Ingresa tu API Key de Groq"
        )

    # ===== LANGSMITH =====
    with st.expander("LangSmith (opcional)", expanded=False):
        st.caption("Trazas de llamadas LLM: prompt, output, latencia y tokens.")
        if os.environ.get("LANGCHAIN_API_KEY"):
            st.success("✅ LangSmith API Key configurada desde secrets")
            ls_key = os.environ["LANGCHAIN_API_KEY"]
        else:
            ls_key = st.text_input(
                "LangSmith API Key", type="password",
                value="",
                key="ls_key_input",
                help="https://smith.langchain.com → Settings → API Keys",
                placeholder="Ingresa tu API Key de LangSmith"
            )
        ls_project = st.text_input(
            "Proyecto",
            value=os.environ.get("LANGCHAIN_PROJECT", "Agente-CNBS"),
            key="ls_project_input",
        )
        ls_on = st.checkbox(
            "Activar tracing",
            value=os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true",
            key="ls_on",
        )
        if ls_on and ls_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_API_KEY"] = ls_key.strip()
            os.environ["LANGCHAIN_PROJECT"] = (ls_project or "Agente-CNBS").strip()
            os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
            st.success(f"🟢 Tracing activo · {os.environ['LANGCHAIN_PROJECT']}")
            gov = st.session_state.get("last_llm_gov")
            if gov:
                st.caption("Última traza gobernada:")
                st.json(gov)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            if ls_on and not ls_key:
                st.warning("Activa el tracing y pega tu API key.")
            else:
                st.caption("⚪ Tracing inactivo")
        st.link_button("↗ Abrir LangSmith", "https://smith.langchain.com", use_container_width=True)

    # ---- Dataset y métricas ----
    st.markdown(f"""
    <div class="side-card"><div class="lbl">📊 Dataset</div><div class="val">{ultima_fecha}</div></div>
    <div class="side-card"><div class="lbl">⚡ Consultas · 🤖 LLM</div>
    <div class="val">{st.session_state.query_count} total · {st.session_state.theme_count} LLM · {st.session_state.direct_count} directo</div></div>
    """, unsafe_allow_html=True)

    if st.button("🗑️ Borrar historial", use_container_width=True):
        st.session_state.messages = []
        st.session_state.contexto.limpiar()
        st.session_state.query_count = 0
        st.session_state.direct_count = 0
        st.session_state.theme_count = 0
        st.session_state.total_time = 0.0
        st.session_state.last_time = 0.0
        st.rerun()

# ============================================================
# HEADER
# ============================================================
st.markdown("## 🏦 Agente Financiero CNBS")
st.markdown(
    f'<div class="header-line">'
    f'Dataset <b>{ultima_fecha}</b> · <b>{n_bancos}</b> bancos · '
    f'<b>{n_inds}</b> indicadores · Última <b>{st.session_state.last_time:.2f}s</b>'
    f'</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["💬 Asistente", "📈 Tendencias", "📋 Datos"])

# ... (el resto del código de tab1, tab2 y tab3 es idéntico al que proporcionaste, no lo repito por longitud)
