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

class Config:
    MODEL_NAME = "llama-3.3-70b-versatile"
    TEMPERATURE = 0.0
    TOP_RESULTS = 14
    ENTIDADES_AGREGADAS = ["BANCOS", "HONDURAS", "Sistema", "SISTEMA", "SISTEMA BANCARIO"]
    PRIORIDAD_TEMAS = ["credito", "riesgo", "salud", "rentabilidad", "solvencia", "liquidez"]
    # Umbrales del score triple (configurables; no hardcode de bancos)
    SCORE_TRIPLE_EXCELENTE = 1.5
    SCORE_TRIPLE_BUENO = 0.8
    FORMULA_SCORE_TRIPLE = "(ROE / Morosidad) × (Capital / 100)"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agente_cnbs")

def _secret(name, default=""):
    """Lee secreto sin exponerlo en widgets. Secrets > env > default."""
    try:
        if hasattr(st, "secrets") and name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.environ.get(name, default) or default




def nivel_score(valor, excelente=None, bueno=None):
    """Clasifica un score numérico en etiqueta/emoji según umbrales de Config."""
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
    """Detecta tipo de ranking por columnas presentes (sin hardcode de bancos)."""
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
    """Conclusión automática desde el DataFrame (determinística)."""
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
    # prefer synonym keys (short) over long official names
    for k in sorted(catalogo.keys(), key=len):
        if k == clave or (len(clave) > 3 and (clave in k or k in clave)):
            # prefer exact synonym match
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

# ===== PARCHE 1: cartera de tarjetas en peso_claves =====
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
        # Si pide ROE explícito y no ROA, no arrastrar ROA por "rentabilidad"
        if c in ("roa", "rentabilidad sobre activos") and ("roe" in q or "patrimonio" in q) and "roa" not in q and "activos" not in q:
            continue
        if c in ("roe", "rentabilidad sobre patrimonio", "rentabilidad sobre el patrimonio") and "roa" in q and "roe" not in q and "patrimonio" not in q:
            # solo ROA pedido
            pass
        claves.append(c)
    # Si dice ROE y no ROA, eliminar claves ROA residuales
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
    """Análisis de riesgo que debe devolver panel por banco, no solo Sistema."""
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
    # informe ejecutivo solo de sistema: no forzar panel bancario
    if "informe ejecutivo" in q and not any(p in q for p in ("banco", "bancos", "preocupante", "mayor riesgo")):
        return False
    return habla_riesgo and pide_detalle_bancos

def detectar_orden_ranking(query):
    """True = orden ascendente (menor/más bajo primero). False = descendente."""
    q = normalizar_texto(query)
    # Frases de "menor valor primero" (mora más baja, menor ROE, etc.)
    ascendente = (
        "mas baja", "mas bajo", "mas bajas", "mas bajos",
        "la menor", "el menor", "menor mora", "menor indice",
        "menor valor", "menor porcentaje", "menor roe", "menor roa",
        "menos riesgoso", "menos riesgosa", "mas seguro", "mas segura",
        "peor",  # peor ranking a veces = menor score; para mora "peor" = mayor → ver abajo
    )
    # "peor mora" / "mayor riesgo" = descendente (valor alto primero)
    if any(p in q for p in ("peor mora", "mayor mora", "mas mora", "mayor riesgo", "mas riesgoso", "mas alta", "mas alto", "la mayor", "el mayor")):
        return False
    if any(p in q for p in ascendente) or "menor" in q:
        return True
    return False

def es_consulta_rentabilidad_riesgo(query):
    """Solo ratio ROE/mora cuando la pregunta se centra en eso.
    Si también pide ROA/capital/varios indicadores explícitos, no interceptar."""
    q = normalizar_texto(query)
    # Si pide un panel amplio (ROA, capital, etc.), dejar motor multi-indicador
    if any(p in q for p in ("roa", "adecuacion", "capital", "spread", "liquidez")):
        return False
    return any(p in q for p in (
        "rentabilidad-riesgo", "rentabilidad riesgo", "relacion rentabilidad",
        "roe y morosidad", "roe y mora", "mejor relacion",
    ))


def es_consulta_conversacional(query):
    """Saludos, fecha/hora u otras frases sin contenido financiero CNBS."""
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
    """Respuesta corta sin invocar Pandas ni LLM."""
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

    # ===== PRIORIDAD 1: indicadores explícitos en la pregunta =====
    claves = extraer_claves_indicador(query)
    if claves:
        inds = resolver_indicadores(claves, catalogo)
        # Solo complementar con panel crédito si el foco es riesgo crediticio genérico
        # y NO pidió ROA/ROE/capital de forma explícita
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

    # ===== PRIORIDAD 2: inversión =====
    if any(p in q for p in ("invertir", "recomendar", "inversion")):
        inds = obtener_indicadores_por_tema("rentabilidad", catalogo)
        if inds:
            return inds, "rentabilidad", detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    # ===== PRIORIDAD 3: panel riesgo crediticio (solo si no hubo claves explícitas) =====
    if es_analisis_riesgo_crediticio(query):
        inds = obtener_indicadores_por_tema("credito", catalogo)
        if inds:
            return inds, "credito", "ranking", bancos, anios, False

    # ===== PRIORIDAD 4: tema general =====
    tema = detectar_tema(query)
    if tema:
        inds = obtener_indicadores_por_tema(tema, catalogo)
        if inds:
            return inds, tema, detectar_tipo_consulta(query), bancos, anios, detectar_orden_ranking(query)

    # ===== PRIORIDAD 5: contexto =====
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

# ===== PARCHE 2: ranking por indicador (no mezclar métricas) =====
def ranking(df, indicadores, anio, top=10, ascending=False):
    """Ranking por Saldo. ascending=True → menor valor primero (ej. mora más baja)."""
    # Multi-indicador: panel completo por banco (no top-N suelto por métrica)
    if indicadores and len(indicadores) > 1:
        d = _base_filtrada(df, None, indicadores, anio)
        if d.empty:
            return d
        res = d.groupby(["Banco", "Indicador"], dropna=False)["Saldo"].mean().reset_index()
        mora_mask = res["Indicador"].str.contains("MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL", case=False, na=False)
        if mora_mask.any():
            orden = (
                res[mora_mask]
                .sort_values("Saldo", ascending=ascending)["Banco"]
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
    res = res.sort_values(by="Saldo", ascending=ascending).reset_index(drop=True)
    if top:
        res = res.head(top).reset_index(drop=True)
    res.insert(0, "Ranking", range(1, len(res) + 1))
    res["Ranking"] = res["Ranking"].astype(int)
    return res

def comparar_bancos(df, bancos, indicadores, anio):
    return promedio(df, aplanar_lista(bancos), indicadores, anio)

def serie_temporal(df, bancos, indicadores):
    d = df[~df["Banco"].isin(Config.ENTIDADES_AGREGADAS)].copy()
    if bancos:
        d = d[d["Banco"].isin(aplanar_lista(bancos))]
    if indicadores:
        d = d[d["Indicador"].isin(indicadores)]
    return d[["FechaReporte", "Banco", "Indicador", "Saldo"]].sort_values("FechaReporte")

# ===== Ratio ROE/morosidad: tabla final Ranking|Banco|ROE|Morosidad|Ratio =====
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
    """Ranking equilibrio ROE + adecuación de capital (por banco, no sistema)."""
    q = normalizar_texto(query)
    habla_roe = "roe" in q or "rentabilidad" in q
    habla_cap = any(p in q for p in (
        "adecuacion", "solvencia", "capital", "solvencia",
    ))
    pide_rank = any(p in q for p in (
        "ranking", "mejores", "mejor equilibrio", "equilibrio",
        "top", "primer lugar", "bancos con mejor",
    ))
    # no confundir con ROE/mora
    if "morosidad" in q or "mora" in q:
        return False
    return habla_roe and habla_cap and (pide_rank or "equilibrio" in q)


def ranking_roe_solvencia(df, anio, top=10, bancos=None):
    """
    Ranking de bancos por equilibrio rentabilidad (ROE) y solvencia (adecuación de capital).
    Score = ROE * Capital / 100 (mayor = mejor equilibrio conjunto).
    Excluye agregados CNBS (Sistema, BANCOS, HONDURAS).
    """
    base = df[(df["FechaReporte"].dt.year == anio)].copy()
    base = base[~base["Banco"].isin(Config.ENTIDADES_AGREGADAS)]
    base = base[~base["Banco"].astype(str).str.upper().str.contains("SISTEMA")]
    if bancos:
        base = base[base["Banco"].isin(aplanar_lista(bancos))]
    roe = base[base["Indicador"].str.contains("ROE", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    cap = base[base["Indicador"].str.contains(
        "ADECUACI", case=False, na=False
    )].groupby("Banco")["Saldo"].mean()
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
    """ROE + morosidad + capital: equilibrio triple."""
    q = normalizar_texto(query)
    habla_roe = "roe" in q or "rentabilidad" in q
    habla_mora = "morosidad" in q or "mora" in q or "riesgo" in q
    habla_cap = any(p in q for p in ("adecuacion", "capital", "solvencia"))
    pide_eq = any(p in q for p in (
        "equilibrio", "equilibrado", "mejor perfil", "perfil mas",
        "ranking", "mejores", "mejor banco",
    ))
    return habla_roe and habla_mora and habla_cap and pide_eq


def ranking_equilibrio_triple(df, anio, top=10, bancos=None):
    """Score = (ROE / Morosidad) * (Capital / 100)."""
    base = df[(df["FechaReporte"].dt.year == anio)].copy()
    base = base[~base["Banco"].isin(Config.ENTIDADES_AGREGADAS)]
    base = base[~base["Banco"].astype(str).str.upper().str.contains("SISTEMA")]
    if bancos:
        base = base[base["Banco"].isin(aplanar_lista(bancos))]
    roe = base[base["Indicador"].str.contains("ROE", case=False, na=False)].groupby("Banco")["Saldo"].mean()
    mora = base[base["Indicador"].str.contains(
        "MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL", case=False, na=False
    )].groupby("Banco")["Saldo"].mean()
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
    """Prosa extra: explicación, por qué, análisis narrativo."""
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
        # Si hay bancos específicos, no tratar como sistema
        if bancos := extraer_bancos(query):
            return False
        return True

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

    # PARCHE 3: ruta especial rentabilidad-riesgo (ROE/Mora)
    if es_consulta_rentabilidad_riesgo(query):
        partes = []
        for anio in anios:
            sub = ranking_rentabilidad_riesgo(df, anio, top=kwargs.get("top", 10), bancos=bancos or None)
            if not sub.empty:
                partes.append(sub)
        if partes:
            return pd.concat(partes, ignore_index=True)

    # PARCHE 3c: equilibrio triple ROE + mora + capital
    if es_consulta_equilibrio_triple(query):
        partes = []
        for anio in anios:
            sub = ranking_equilibrio_triple(
                df, anio, top=kwargs.get("top", 10), bancos=bancos or None
            )
            if not sub.empty:
                partes.append(sub)
        if partes:
            return pd.concat(partes, ignore_index=True)
        return pd.DataFrame()

    # PARCHE 3b: equilibrio ROE + adecuación de capital (por banco)
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

    # Nunca usar promedio agregado ("Sistema") si la pregunta pide ranking/mejores bancos
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
    # Comparaciones multi-banco / multi-año → tabla Markdown desde Pandas
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
    # Panel multi-banco ya formateado: no LLM salvo prosa explícita
    if (
        "Indicador" in df_res.columns
        and "Banco" in df_res.columns
        and df_res["Banco"].nunique() >= 3
        and df_res["Indicador"].nunique() >= 2
        and not usuario_pide_detalle(query)
    ):
        return False
    if any(p in q for p in (
        "recomendar", "explica", "informe narrativo",
    )):
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
    # Prioridad: equilibrio triple (incluye Ratio_ROE_Mora como col auxiliar; no usarla en LLM)
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
    """
    Resultado determinístico de Pandas para gobernar al LLM.
    Devuelve dict con ganador y métricas clave (o {}).
    """
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
    # Rechazar agregados CNBS como "ganador"
    if str(banco).strip().upper() in {"SISTEMA", "BANCOS", "HONDURAS", "SISTEMA BANCARIO"}:
        # intentar siguiente fila bancaria
        for _, alt in df_res.iterrows():
            b2 = alt.get("Banco")
            if pd.notna(b2) and str(b2).strip().upper() not in {
                "SISTEMA", "BANCOS", "HONDURAS", "SISTEMA BANCARIO"
            }:
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
    # lista de bancos del resultado (para validación anti-contradicción)
    try:
        resultado["bancos_en_resultado"] = sorted(
            {str(b) for b in df_res["Banco"].dropna().unique().tolist()}
        )
    except Exception:
        resultado["bancos_en_resultado"] = [resultado["ganador"]]
    return {k: v for k, v in resultado.items() if v is not None or k in ("ganador",)}


def extraer_ganador(df_res):
    """Compatibilidad: devuelve solo el nombre del ganador."""
    r = extraer_resultado(df_res)
    return r.get("ganador")


def extraer_metricas_ganador(df_res, ganador):
    """Compatibilidad con código previo."""
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
    """
    Capa de gobernanza: el LLM solo explica resultados de Pandas.
    No recalcula, no cambia ranking ni ganador.
    """
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
{extra_score}- El banco en Ranking=1 / primer lugar del resultado Pandas es: {ganador}.
- Si la pregunta pide mayor, menor, más baja, más alta, mejor, lidera o equilibrado:
  la única respuesta correcta para el primer lugar es {ganador}.
- NO propongas otro banco como ganador si contradice DATOS.
- Usa el orden de DATOS: la primera fila es el #1 del criterio solicitado.
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
    """Compatibilidad: delega en construir_contexto_llm."""
    return construir_contexto_llm(query, df_res, meta_info, contexto_texto)


def validar_respuesta_llm(respuesta, resultado):
    """
    Valida que la respuesta respete el resultado determinístico.
    - Debe mencionar al ganador (si existe).
    - No debe atribuir victoria/liderazgo a otro banco del resultado.
    """
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

    # Frases de victoria / liderazgo atribuidas a otro banco
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
            # ventana simple: banco cerca de frase de victoria
            for tok in tokens_b:
                if tok in r and frase in r:
                    # si el ganador también está en la misma respuesta con esa frase, ok
                    # rechazar solo si el otro banco aparece junto a victoria y el ganador no está en esa zona
                    # regla estricta: si frase de victoria y otro banco, y ganador no aparece en respuesta → fail (ya cubierto)
                    # si ambos aparecen, exigir que no diga que el otro "es el mejor"
                    idx_f = r.find(frase)
                    idx_b = r.find(tok)
                    if idx_f >= 0 and idx_b >= 0 and abs(idx_f - idx_b) < 80:
                        # ¿el ganador está aún más cerca?
                        idx_g = min((r.find(t) for t in tokens_g if r.find(t) >= 0), default=-1)
                        if idx_g < 0 or abs(idx_f - idx_b) < abs(idx_f - idx_g):
                            return False
    return True


def _langsmith_config(run_name, resultado=None, extra=None):
    """Config de traza para LangSmith (tags + metadata)."""
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
    """
    Invoca LLM, valida contra resultado determinístico; 1 reintento.
    Trazas LangSmith con run_name/tags/metadata si tracing está activo.
    Devuelve (texto, meta_gobernanza).
    """
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
    """Tarjeta ganadora + ranking con medallas (markdown para historial)."""
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
    """UI enriquecida: métricas + dataframe cuando hay ranking (tipo detectado por columnas)."""
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
    """Plantilla ejecutiva para pocas métricas de sistema."""
    titulo = "Sistema bancario"
    if anios:
        titulo += f" {', '.join(map(str, anios))}"
    lineas = [f"**{titulo}**", ""]
    if "Indicador" in df_res.columns:
        for _, row in df_res.iterrows():
            ind = str(row.get("Indicador", ""))
            # nombre corto
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
    """Tabla markdown sin paquete tabulate."""
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
    """Ranking equilibrio triple: fórmula y umbrales desde Config."""
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
        if "Ranking" in df_res.columns:
            orden = df_res.sort_values("Ranking", ascending=True).reset_index(drop=True)
        else:
            asc = detectar_orden_ranking(query)
            orden = df_res.sort_values("Saldo", ascending=asc).reset_index(drop=True)
        lineas = [f"**{ind}**\n", "| # | Banco | Valor % |", "|--:|:------|--------:|"]
        for i, (_, row) in enumerate(orden.iterrows(), 1):
            rk = int(row["Ranking"]) if "Ranking" in orden.columns and pd.notna(row.get("Ranking")) else i
            lineas.append(f"| {rk} | {row['Banco']} | {row['Saldo']:.2f} |")
        top = orden.iloc[0]
        if detectar_orden_ranking(query):
            lineas.append(f"\n**Menor valor:** **{top['Banco']}** ({float(top['Saldo']):.2f}%).")
        else:
            lineas.append(f"\n**Mayor valor:** **{top['Banco']}** ({float(top['Saldo']):.2f}%).")
        return "\n".join(lineas)


    return df_a_markdown(df_res) if hasattr(df_res, "iterrows") else str(df_res)


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


# ============================================================
# EXPORTACIONES
# ============================================================
try:
    from pdf_renderer import generar_pdf_respuesta, get_pdf_renderer, PDFRenderer
except ImportError:
    # Fallback mínimo si no está el módulo
    def generar_pdf_respuesta(consulta, respuesta="", meta="", titulo="Informe Agente CNBS", **kwargs):
        from datetime import datetime
        text = f"{titulo}\n{datetime.now()}\n\nCONSULTA\n{consulta}\n\nRESPUESTA\n{respuesta}\n\n{meta}"
        # PDF muy básico
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
    """PNG del gráfico Plotly: kaleido → matplotlib fallback (con título y ejes)."""
    try:
        return fig.to_image(format="png", width=1100, height=520, scale=2)
    except Exception:
        pass
    try:
        import plotly.io as pio
        return pio.to_image(fig, format="png", width=1100, height=520)
    except Exception:
        pass
    # Fallback matplotlib más profesional
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
                # fechas
                try:
                    xd = pd.to_datetime(x)
                except Exception:
                    xd = x
                ax.plot(xd, y, marker="o", label=name, linewidth=2, markersize=5, color=color)
                # resaltar último punto
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
            ax.xaxis.set_major_formatter(
                mdates.DateFormatter("%b-%y")
            )
            # parche español simple vía tick labels después
            fig_m.autofmt_xdate(rotation=0, ha="center")
            labels = []
            for t in ax.get_xticks():
                try:
                    dt = mdates.num2date(t)
                    labels.append(f"{MESES.get(dt.month, dt.strftime('%b'))}-{str(dt.year)[2:]}")
                except Exception:
                    labels.append("")
            # solo aplicar si hay ticks razonables
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
    """
    Hallazgos automáticos 100% Pandas a partir de la serie temporal filtrada.
    Sin LLM, sin hardcode de bancos.
    """
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

    # Banco con mayor / menor nivel (promedio en el rango)
    if "Banco" in s.columns and n_bancos >= 1:
        por_banco = s.groupby("Banco")["Saldo"].mean().sort_values(ascending=False)
        b_max, v_max_b = por_banco.index[0], float(por_banco.iloc[0])
        b_min, v_min_b = por_banco.index[-1], float(por_banco.iloc[-1])
        lineas.append(f"Mayor nivel promedio: {b_max} ({v_max_b:.2f}%).")
        if n_bancos >= 2:
            lineas.append(f"Menor nivel promedio: {b_min} ({v_min_b:.2f}%).")
        # Último punto disponible por banco
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
    """PDF profesional del gráfico de tendencias (mismo estilo que el informe)."""
    from io import BytesIO
    from datetime import datetime

    # Preferir PDFRenderer si está disponible
    try:
        from pdf_renderer import get_pdf_renderer
        meta = metadata or {
            "Tipo": "Tendencias",
            "Fuente": "CNBS Honduras",
            "Motor": "Plotly + Pandas",
        }
        # kpis como respuesta + summary
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

        # Título del gráfico en tarjeta
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

        # KPIs
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

        # Gráfico
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
        # fallback CSV bytes with xlsx extension avoided — return csv
        return df_export.to_csv(index=False).encode("utf-8-sig")


st.set_page_config(
    page_title="Agente Financiero CNBS",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- CSS ----
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

    # --- API keys: nunca prellenar el input con el secreto real (el 👁 lo revelaría) ---
    groq_from_secrets = bool(_secret("GROQ_API_KEY"))
    st.markdown("**🔐 Groq API Key**")
    if groq_from_secrets:
        groq_api_key = _secret("GROQ_API_KEY")
        st.success("🟢 Clave Groq cargada desde Secrets (oculta)")
        st.caption("No se muestra en pantalla. Configurada en Streamlit Cloud → Settings → Secrets.")
    else:
        groq_api_key = st.text_input(
            "Groq key",
            type="password",
            value="",
            label_visibility="collapsed",
            key="groq_key_input",
            placeholder="gsk_… (solo si no usas Secrets)",
        )
        if groq_api_key:
            st.caption("Clave en sesión · no se guarda en el código")

    with st.expander("LangSmith (opcional)", expanded=False):
        st.caption("Trazas de llamadas LLM: prompt, output, latencia y tokens.")
        ls_from_secrets = bool(_secret("LANGCHAIN_API_KEY"))
        if ls_from_secrets:
            ls_key = _secret("LANGCHAIN_API_KEY")
            st.success("🟢 LangSmith key desde Secrets (oculta)")
        else:
            ls_key = st.text_input(
                "LangSmith API Key",
                type="password",
                value="",
                key="ls_key_input",
                placeholder="lsv2_… (opcional)",
                help="https://smith.langchain.com → Settings → API Keys",
            )
        ls_project = st.text_input(
            "Proyecto",
            value=_secret("LANGCHAIN_PROJECT", "Agente-CNBS") or "Agente-CNBS",
            key="ls_project_input",
        )
        ls_on_default = _secret("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1", "yes")
        ls_on = st.checkbox(
            "Activar tracing",
            value=ls_on_default,
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
                st.warning("Activa el tracing y define LANGCHAIN_API_KEY en Secrets o pégala aquí.")
            else:
                st.caption("⚪ Tracing inactivo")
        st.link_button("↗ Abrir LangSmith", "https://smith.langchain.com", use_container_width=True)

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

# ============================================================
# TAB 1 — ASISTENTE
# ============================================================
with tab1:
    if not groq_api_key:
        st.warning("Introduce tu API Key de Groq en la barra lateral para activar el asistente.")
    else:
        llm = get_llm(groq_api_key)

        # Mensaje de bienvenida (solo visual, no en historial)
        if not st.session_state.messages:
            st.markdown("""
            <div class="welcome-box">
                <div style="font-size:18px;font-weight:600;margin-bottom:6px;">👋 Bienvenido al Asistente CNBS</div>
                <div style="font-size:14px;line-height:1.5;">
                    Puedo responder consultas sobre <b>ROA, ROE, morosidad, capital, spread</b> y más,
                    con cálculos exactos en Pandas. Prueba un botón rápido o escribe tu pregunta.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Historial
        for i, m in enumerate(st.session_state.messages):
            with st.chat_message(m["role"]):
                if m["role"] == "assistant":
                    st.markdown(m["content"])
                    if m.get("meta"):
                        st.markdown(f'<div class="meta-bar">{m["meta"]}</div>', unsafe_allow_html=True)
                    # PDF programático de esta respuesta (Pandas o LLM)
                    pdf_bytes = m.get("pdf")
                    if not pdf_bytes:
                        try:
                            pdf_bytes = generar_pdf_respuesta(
                                m.get("query", ""),
                                m.get("content", ""),
                                m.get("meta", ""),
                            )
                        except Exception as ex:
                            pdf_bytes = None
                            st.caption(f"PDF no disponible: {ex}")
                    if pdf_bytes:
                        st.download_button(
                            "📄 Descargar informe PDF",
                            data=pdf_bytes,
                            file_name=f"informe_cnbs_{i+1}.pdf",
                            mime="application/pdf",
                            key=f"pdf_resp_{i}",
                            use_container_width=False,
                        )
                else:
                    st.markdown(m["content"])

        # Botones rápidos 2x2
        st.markdown("**Consultas sugeridas**")
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            if st.button("📉  Riesgo crediticio", use_container_width=True, key="qb1"):
                st.session_state.pending_query = (
                    "Analiza el riesgo crediticio del sistema bancario hondureño en 2025 "
                    "considerando morosidad, cobertura de mora y cartera de tarjetas de crédito"
                )
        with r1c2:
            if st.button("🏦  Comparar bancos", use_container_width=True, key="qb2"):
                st.session_state.pending_query = (
                    "Compara el desempeño de AZTECA, BAC CREDOMATIC y FICOHSA en 2025 "
                    "considerando ROA, ROE, morosidad y adecuación de capital. "
                    "¿Qué banco presenta el perfil más equilibrado?"
                )
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            if st.button("📊  Ratio ROE / Mora", use_container_width=True, key="qb3"):
                st.session_state.pending_query = (
                    "¿Qué banco tiene mejor relación rentabilidad-riesgo en 2025? Considera ROE y morosidad."
                )
        with r2c2:
            if st.button("📈  Evolución 2024-2025", use_container_width=True, key="qb4"):
                st.session_state.pending_query = (
                    "Compara la evolución del ROA y ROE del sistema bancario hondureño entre 2024 y 2025"
                )

        # Input de pregunta abierta siempre disponible
        chat_val = st.chat_input("Escribe tu consulta financiera...")
        query = st.session_state.pending_query or chat_val
        if st.session_state.pending_query:
            st.session_state.pending_query = None

        if query:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Procesando con motor Pandas..."):
                    t0 = time.time()
                    try:
                        fuente = None
                        if es_consulta_conversacional(query):
                            output = respuesta_conversacional(query)
                            fuente = "Conversacional"
                            usado = False
                            conf, conf_motivo = "Alta", "Consulta no financiera"
                            df_res = None
                            tema = None
                            bancos = []
                            anios = []
                            indicadores = []
                        else:
                            bancos = extraer_bancos(query)
                            anios = extraer_anios(query) or [int(df["FechaReporte"].dt.year.max())]
                            indicadores, tema, tipo, _, _, asc = planificar(
                                query, st.session_state.contexto, catalogo
                            )
                            df_res = ejecutar_consulta(
                                df, indicadores, bancos, anios, tipo,
                                query=query, top=Config.TOP_RESULTS, asc=asc,
                            )
                        if fuente == "Conversacional":
                            pass  # output y conf ya definidos
                        elif not validar_resultado(df_res):
                            output = f"⚠️ Sin datos. Indicadores: {indicadores}. Años: {anios}"
                            fuente = "Motor (sin datos)"
                            usado = False
                            conf, conf_motivo = "Baja", "Sin filas de resultado"
                        else:
                            conf, conf_motivo = calcular_confianza(df_res, indicadores, bancos, anios)
                            meta_info = {
                                "tipo": tipo, "anios": anios,
                                "bancos": bancos or "sistema",
                                "indicadores": indicadores, "filas": len(df_res),
                                "modo_corto": consulta_simple_sistema(query, df_res) and not usuario_pide_detalle(query),
                                "confianza": conf,
                            }
                            if necesita_llm(df_res, query, meta_info):
                                resultado = extraer_resultado(df_res)
                                prompt_llm = construir_contexto_llm(
                                    query, df_res, meta_info,
                                    st.session_state.contexto.obtener_contexto(),
                                    resultado=resultado,
                                )
                                output, meta_gov = redactar_respuesta(
                                    llm, prompt_llm, resultado=resultado
                                )
                                fuente = "Pandas + LLM"
                                usado = True
                                st.session_state.theme_count += 1
                                st.session_state.last_llm_gov = {
                                    "motor": fuente,
                                    "ganador": resultado.get("ganador"),
                                    "validacion": meta_gov.get("validacion"),
                                    "reintentos": meta_gov.get("reintentos", 0),
                                    "tiempo": None,
                                }
                            else:
                                output = responder_directamente(df_res, query, meta_info)
                                fuente = "Pandas directo"
                                usado = False
                                st.session_state.direct_count += 1
                            st.session_state.contexto.actualizar(
                                query, output, tema, bancos, anios[0], indicadores
                            )

                        dt = time.time() - t0
                        st.session_state.query_count += 1
                        st.session_state.total_time += dt
                        st.session_state.last_time = dt
                        st.session_state.last_query = query
                        if st.session_state.get("last_llm_gov"):
                            st.session_state.last_llm_gov["tiempo"] = round(dt, 2)

                        chip = {"Alta": "chip-alta", "Media": "chip-media", "Baja": "chip-baja"}.get(conf, "chip-media")
                        motor = "Pandas" if "directo" in str(fuente).lower() else str(fuente)
                        meta_html = (
                            f'<div class="meta-bar">'
                            f'📌 Motor: <b>{motor}</b><br>'
                            f'🛡️ Confianza: <span class="chip {chip}">{conf}</span>'
                            + (f' — {conf_motivo}' if conf_motivo else '')
                            + f'<br>⏱ {dt:.2f}s · 📅 Datos: CNBS {ultima_fecha}'
                            f'</div>'
                        )
                        meta_plain = (
                            f"📌 Motor: {motor} · 🛡️ Confianza: {conf}"
                            + (f" — {conf_motivo}" if conf_motivo else "")
                            + f" · ⏱ {dt:.2f}s · 📅 CNBS {ultima_fecha}"
                        )
                        df_show = df_res if validar_resultado(df_res) else None
                        render_respuesta_ui(output, df_show, meta_html)
                        try:
                            meta_dict = {
                                "Motor": motor,
                                "Confianza": conf,
                                "Tiempo": f"{dt:.2f}s",
                                "Dataset": f"CNBS {ultima_fecha}",
                            }
                            if conf_motivo:
                                meta_dict["Detalle"] = conf_motivo
                            df_pdf = df_res if validar_resultado(df_res) else None
                            # ranking wide table if present
                            if df_pdf is not None and "Ratio_ROE_Mora" in df_pdf.columns:
                                pass  # already tabular
                            elif df_pdf is not None and "Indicador" in df_pdf.columns:
                                try:
                                    piv = pivot_resultados(df_pdf)
                                    if piv is not None and not piv.empty:
                                        df_pdf = piv.reset_index()
                                except Exception:
                                    pass
                            summary_pdf = None
                            if df_pdf is not None and not df_pdf.empty and "Banco" in df_pdf.columns:
                                if "Ratio_ROE_Mora" in df_pdf.columns:
                                    top = (
                                        df_pdf.sort_values("Ranking").iloc[0]
                                        if "Ranking" in df_pdf.columns else df_pdf.iloc[0]
                                    )
                                    summary_pdf = (
                                        f"El análisis determinístico identifica a {top['Banco']} "
                                        f"como la institución con la mejor relación rentabilidad–riesgo "
                                        f"para el periodo, con un ratio ROE/Morosidad de "
                                        f"{float(top['Ratio_ROE_Mora']):.2f}."
                                    )
                                elif "Score_Triple" in df_pdf.columns:
                                    top = (
                                        df_pdf.sort_values("Ranking").iloc[0]
                                        if "Ranking" in df_pdf.columns else df_pdf.iloc[0]
                                    )
                                    summary_pdf = (
                                        f"El análisis determinístico identifica a {top['Banco']} "
                                        f"con el mejor score de equilibrio triple "
                                        f"({float(top['Score_Triple']):.2f})."
                                    )
                                elif "Saldo" in df_pdf.columns and "Ranking" in df_pdf.columns:
                                    top = df_pdf.sort_values("Ranking").iloc[0]
                                    asc = detectar_orden_ranking(query)
                                    criterio = "menor valor" if asc else "mayor valor"
                                    summary_pdf = (
                                        f"Según el ranking determinístico de Pandas, "
                                        f"{top['Banco']} ocupa el primer lugar ({criterio}: "
                                        f"{float(top['Saldo']):.2f}%)."
                                    )
                            pdf_bytes = generar_pdf_respuesta(
                                consulta=query,
                                respuesta=output,
                                meta=meta_dict,
                                dataframe=df_pdf,
                                summary=summary_pdf,
                                titulo="Informe Financiero CNBS",
                            )
                        except Exception:
                            pdf_bytes = None
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": output,
                            "meta": meta_plain,
                            "query": query,
                            "pdf": pdf_bytes,
                        })
                        st.rerun()
                    except Exception as e:
                        import traceback
                        st.error(f"{e}\n```\n{traceback.format_exc()}\n```")

# ============================================================
# TAB 2 — TENDENCIAS (EDA interactivo)
# ============================================================
with tab2:
    st.markdown("### 📈 Tendencias de indicadores")
    st.caption("Módulo de análisis exploratorio · Evolución temporal por banco · Fuente CNBS")

    # Tarjetas resumen horizontales
    st.markdown(f"""
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin:8px 0 16px 0;">
      <div class="side-card" style="flex:1;min-width:120px;"><div class="lbl">📄 Registros</div><div class="val">{total_registros:,}</div></div>
      <div class="side-card" style="flex:1;min-width:120px;"><div class="lbl">🏦 Bancos</div><div class="val">{n_bancos}</div></div>
      <div class="side-card" style="flex:1;min-width:120px;"><div class="lbl">📊 Indicadores</div><div class="val">{n_inds}</div></div>
      <div class="side-card" style="flex:1;min-width:120px;"><div class="lbl">📅 Meses</div><div class="val">{total_meses}</div></div>
    </div>
    """, unsafe_allow_html=True)

    CAT_ORDER = [
        "1) INDICADORES DE SOLVENCIA",
        "2) INDICADORES DE CALIDAD DE ACTIVOS",
        "3) INDICADORES DE LIQUIDEZ",
        "4) INDICADORES DE RENTABILIDAD",
        "5) INDICADORES DE CUMPLIMIENTO",
        "6) INDICADORES DE GESTIÓN",
    ]
    if "CategoriaIndicador" in df.columns:
        cat_to_inds = {
            c: sorted(df[df["CategoriaIndicador"] == c]["Indicador"].dropna().unique().tolist())
            for c in CAT_ORDER
            if c in set(df["CategoriaIndicador"].dropna().unique())
        }
    else:
        cat_to_inds = {}

    def etiqueta_indicador(nombre):
        u = str(nombre).upper()
        if "ROA" in u and "ROE" not in u:
            return "ROA — Rentabilidad sobre activos"
        if "ROE" in u:
            return "ROE — Rentabilidad sobre patrimonio"
        if "MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL" in u:
            return "Morosidad total"
        if "MOROSIDAD SOBRE CARTERA CREDITICIA DIRECTA" in u:
            return "Morosidad directa"
        if "COBERTURA DE LA MORA" in u:
            return "Cobertura de mora"
        if "TARJETAS" in u:
            return "Cartera de tarjetas"
        if "ADECUACIÓN DE CAPITAL" in u or "ADECUACION DE CAPITAL" in u:
            return "Adecuación de capital"
        if "SPREAD" in u:
            return "Spread de intermediación"
        if "TASA ACTIVA" in u:
            return "Tasa activa"
        if "TASA PASIVA" in u:
            return "Tasa pasiva"
        if "RATIO DE COBERTURA DE LIQUIDEZ" in u:
            return "Ratio cobertura de liquidez"
        if "LIQUIDEZ" in u and "ACTIVOS LÍQUIDOS +" not in u:
            return "Índice de liquidez"
        if "ACTIVOS LÍQUIDOS +" in u:
            return "Liquidez ampliada"
        if "ACTIVOS FIJOS" in u:
            return "Inversión en activos fijos"
        if "GASTOS DE ADMINISTRACIÓN (ANUALIZADOS)" in u:
            return "Gastos adm. / activos"
        if "GASTOS DE ADMINISTRACIÓN/INGRESOS" in u:
            return "Gastos adm. / ingresos"
        if "CALCES" in u:
            return "Calces moneda extranjera"
        return (nombre[:52] + "…") if len(str(nombre)) > 52 else str(nombre)

    st.markdown("#### 🔍 Filtros de análisis")
    f1, f2 = st.columns(2)
    with f1:
        cats_opts = ["Todas las categorías"] + [c for c in CAT_ORDER if c in cat_to_inds]
        cat_sel = st.selectbox("Categoría", cats_opts, key="tend_cat")
        if cat_sel == "Todas las categorías":
            inds_filtrados = list(inds_lista)
        else:
            inds_filtrados = cat_to_inds.get(cat_sel, list(inds_lista))

        etiquetas = {}
        et_list = []
        for i in inds_filtrados:
            et = etiqueta_indicador(i)
            if et in etiquetas:
                et = f"{et} · {str(i)[:18]}"
            etiquetas[et] = i
            et_list.append(et)
        et_sel = st.selectbox("Indicador", et_list, key="tend_ind")
        ind_sel = etiquetas[et_sel]
        with st.expander("ⓘ Ver definición oficial CNBS"):
            st.code(ind_sel, language=None)

    with f2:
        sel_all = st.checkbox("Seleccionar todos los bancos", value=False, key="tend_all")
        if sel_all:
            bancos_sel = bancos_lista
        else:
            bancos_sel = st.multiselect(
                "Bancos", bancos_lista,
                default=bancos_lista[:5] if len(bancos_lista) >= 5 else bancos_lista,
                key="tend_bancos",
            )
        fechas = sorted(df["FechaReporte"].dropna().unique())
        if len(fechas) >= 2:
            f_min, f_max = st.select_slider(
                "Rango de fechas",
                options=fechas,
                value=(fechas[0], fechas[-1]),
                format_func=lambda x: pd.Timestamp(x).strftime("%m/%Y"),
                key="tend_fechas",
            )
        else:
            f_min, f_max = (fechas[0], fechas[0]) if fechas else (None, None)

    serie = serie_temporal(df, bancos_sel if bancos_sel else None, [ind_sel])
    if f_min is not None:
        serie = serie[(serie["FechaReporte"] >= pd.Timestamp(f_min)) & (serie["FechaReporte"] <= pd.Timestamp(f_max))]

    if serie.empty:
        st.info("No hay datos para los filtros seleccionados.")
    else:
        n_sel = len(bancos_sel) if bancos_sel else n_bancos
        # KPIs con acentos visuales
        st.markdown(f"""
        <div style="display:flex;gap:10px;flex-wrap:wrap;margin:4px 0 14px 0;">
          <div class="side-card" style="flex:1;min-width:110px;border-left:4px solid #1976d2;">
            <div class="lbl">🟦 Promedio</div><div class="val">{serie['Saldo'].mean():.2f}%</div>
          </div>
          <div class="side-card" style="flex:1;min-width:110px;border-left:4px solid #2e7d32;">
            <div class="lbl">🟩 Máximo</div><div class="val">{serie['Saldo'].max():.2f}%</div>
          </div>
          <div class="side-card" style="flex:1;min-width:110px;border-left:4px solid #f9a825;">
            <div class="lbl">🟨 Mínimo</div><div class="val">{serie['Saldo'].min():.2f}%</div>
          </div>
          <div class="side-card" style="flex:1;min-width:110px;border-left:4px solid #7b1fa2;">
            <div class="lbl">🟪 Observaciones</div><div class="val">{len(serie):,}</div>
          </div>
          <div class="side-card" style="flex:1;min-width:110px;border-left:4px solid #455a64;">
            <div class="lbl">🏦 Bancos sel.</div><div class="val">{n_sel}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        f_ini = pd.Timestamp(serie["FechaReporte"].min()).strftime("%B %Y")
        f_fin = pd.Timestamp(serie["FechaReporte"].max()).strftime("%B %Y")
        # meses en español aproximado vía strftime locale may be English - use m/Y
        f_ini = pd.Timestamp(serie["FechaReporte"].min()).strftime("%m/%Y")
        f_fin = pd.Timestamp(serie["FechaReporte"].max()).strftime("%m/%Y")

        st.markdown(f"#### 📈 Evolución mensual de **{et_sel}**")
        bancos_lbl = " · ".join(bancos_sel[:4]) if bancos_sel else "Sistema"
        if bancos_sel and len(bancos_sel) > 4:
            bancos_lbl += f" · +{len(bancos_sel)-4}"
        st.caption(f"{bancos_lbl} · {f_ini} — {f_fin} · {serie['FechaReporte'].nunique()} periodos")

        # Misma paleta por banco en línea y barras
        bancos_plot = sorted(serie["Banco"].dropna().unique().tolist())
        palette = (
            px.colors.qualitative.Plotly
            + px.colors.qualitative.Dark24
            + px.colors.qualitative.Set3
        )
        color_map = {b: palette[i % len(palette)] for i, b in enumerate(bancos_plot)}

        titulo_graf = f"Evolución mensual: {et_sel} (%)"
        subtitulo = f"{bancos_lbl} · {f_ini} — {f_fin}"

        fig = px.line(
            serie, x="FechaReporte", y="Saldo", color="Banco",
            markers=True,
            color_discrete_map=color_map,
            category_orders={"Banco": bancos_plot},
            labels={
                "FechaReporte": "",
                "Saldo": f"{et_sel} (%)",
                "Banco": "Institución",
            },
        )
        fig.update_layout(
            template="plotly_white",
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=-0.28, x=0),
            margin=dict(l=10, r=10, t=50, b=10),
            title=dict(text=f"<b>{titulo_graf}</b><br><sup>{subtitulo}</sup>", x=0.01, xanchor="left"),
            yaxis_title=f"{et_sel} (%)",
            xaxis_tickformat="%b-%y",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Resumen automático (Pandas)
        try:
            ult = serie["FechaReporte"].max()
            filas_u = serie[serie["FechaReporte"] == ult].groupby("Banco")["Saldo"].mean()
            bullets = []
            for b, v in filas_u.sort_values(ascending=False).items():
                bullets.append(f"**{b}**: {v:.2f}% (último periodo)")
            if len(filas_u) >= 2:
                top_b, top_v = filas_u.idxmax(), filas_u.max()
                low_b, low_v = filas_u.idxmin(), filas_u.min()
                bullets.append(
                    f"Diferencia {top_b} vs {low_b}: **{top_v - low_v:.2f}** pp en el último mes."
                )
            # variación vs primer mes por banco
            for b in list(filas_u.index)[:3]:
                sub = serie[serie["Banco"] == b].sort_values("FechaReporte")
                if len(sub) >= 2:
                    d = float(sub["Saldo"].iloc[-1] - sub["Saldo"].iloc[0])
                    bullets.append(
                        f"{b}: cambio acumulado en el rango **{d:+.2f}** pp."
                    )
            st.markdown("**Resumen automático**")
            for b in bullets[:5]:
                st.markdown(f"- {b}")
        except Exception:
            pass

        ultimo = serie["FechaReporte"].max()
        rank = (
            serie[serie["FechaReporte"] == ultimo]
            .groupby("Banco", as_index=False)["Saldo"].mean()
            .sort_values("Saldo", ascending=True)
        )
        st.markdown(f"#### 📊 Comparación del último período · {pd.Timestamp(ultimo).strftime('%m/%Y')}")
        fig2 = px.bar(
            rank, x="Saldo", y="Banco", orientation="h",
            color="Banco",
            color_discrete_map=color_map,
            category_orders={"Banco": bancos_plot},
            labels={"Saldo": "Valor (%)", "Banco": ""},
        )
        fig2.update_layout(
            template="plotly_white",
            height=max(280, 28 * len(rank)),
            showlegend=False,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Exportaciones del gráfico (PNG / PDF) — no el dataset
        st.markdown("---")
        st.markdown("**Exportar gráfico**")
        kpis_txt = (
            f"Indicador: {et_sel}\n"
            f"Periodo: {f_ini} — {f_fin}\n"
            f"Promedio: {serie['Saldo'].mean():.2f}% · "
            f"Máximo: {serie['Saldo'].max():.2f}% · "
            f"Mínimo: {serie['Saldo'].min():.2f}%\n"
                        f"Bancos: {n_sel} · Observaciones: {len(serie)}"
        )
        hallazgos_txt = hallazgos_desde_serie(serie, et_sel, f_ini, f_fin)
        with st.expander("📋 Hallazgos principales (Pandas)", expanded=False):
            st.markdown(hallazgos_txt)
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in et_sel)[:40]
        png_bytes = fig_to_png_bytes(fig, titulo=titulo_graf + "\n" + subtitulo, ylabel=f"{et_sel} (%)")
        if png_bytes is None:
            st.caption("PNG no disponible en este entorno.")

        e1, e2 = st.columns(2)
        with e1:
            if png_bytes:
                st.download_button(
                    "📷 Descargar PNG",
                    data=png_bytes,
                    file_name=f"tendencia_{safe_name}.png",
                    mime="image/png",
                    key="dl_tend_png",
                )
        with e2:
            try:
                pdf_t = generar_pdf_tendencia(
                    f"Evolución mensual de {et_sel} ({f_ini} — {f_fin})",
                    kpis_txt,
                    png_bytes,
                    hallazgos=hallazgos_txt,
                )
                st.download_button(
                    "📄 PDF del gráfico",
                    data=pdf_t,
                    file_name=f"tendencia_{safe_name}.pdf",
                    mime="application/pdf",
                    key="dl_tend_pdf",
                )
            except Exception as ex:
                st.caption(f"PDF gráfico: {ex}")

# ============================================================
# TAB 3 — DATOS
# ============================================================
with tab3:
    st.markdown("### 📋 Explorador de datos CNBS")
    st.caption("Filtros por listas desplegables · exporta solo la vista filtrada")

    # Opciones desde el dataset (sin hardcode)
    bancos_opts = sorted(
        df[~df["Banco"].isin(Config.ENTIDADES_AGREGADAS)]["Banco"].dropna().unique().tolist()
    )
    anios_opts = sorted(df["FechaReporte"].dt.year.dropna().unique().astype(int).tolist(), reverse=True)

    CAT_ORDER_DATOS = [
        "1) INDICADORES DE SOLVENCIA",
        "2) INDICADORES DE CALIDAD DE ACTIVOS",
        "3) INDICADORES DE LIQUIDEZ",
        "4) INDICADORES DE RENTABILIDAD",
        "5) INDICADORES DE CUMPLIMIENTO",
        "6) INDICADORES DE GESTIÓN",
    ]
    if "CategoriaIndicador" in df.columns:
        cats_presentes = [c for c in CAT_ORDER_DATOS if c in set(df["CategoriaIndicador"].dropna().unique())]
        # categorías extra no listadas
        extras = sorted(set(df["CategoriaIndicador"].dropna().unique()) - set(cats_presentes))
        cats_opts = ["Todas"] + cats_presentes + extras
    else:
        cats_opts = ["Todas"]

    f1, f2 = st.columns(2)
    with f1:
        banco_sel = st.selectbox(
            "Banco",
            ["Todos"] + bancos_opts,
            key="datos_banco",
        )
        cat_sel = st.selectbox(
            "Categoría",
            cats_opts,
            key="datos_cat",
        )
    with f2:
        # Indicadores filtrados por categoría elegida
        if cat_sel != "Todas" and "CategoriaIndicador" in df.columns:
            inds_base = sorted(
                df[df["CategoriaIndicador"] == cat_sel]["Indicador"].dropna().unique().tolist()
            )
        else:
            inds_base = sorted(df["Indicador"].dropna().unique().tolist())

        def _etiq_ind(nombre):
            u = str(nombre).upper()
            if "ROA" in u and "ROE" not in u:
                return "ROA — Rentabilidad sobre activos"
            if "ROE" in u:
                return "ROE — Rentabilidad sobre patrimonio"
            if "MOROSIDAD SOBRE CARTERA CREDITICIA TOTAL" in u:
                return "Morosidad total"
            if "MOROSIDAD SOBRE CARTERA CREDITICIA DIRECTA" in u:
                return "Morosidad directa"
            if "COBERTURA DE LA MORA" in u:
                return "Cobertura de mora"
            if "TARJETAS" in u:
                return "Cartera de tarjetas"
            if "ADECUACIÓN DE CAPITAL" in u or "ADECUACION DE CAPITAL" in u:
                return "Adecuación de capital"
            if "SPREAD" in u:
                return "Spread de intermediación"
            if "LIQUIDEZ" in u:
                return "Liquidez / cobertura liquidez"
            return (str(nombre)[:48] + "…") if len(str(nombre)) > 48 else str(nombre)

        et_map = {}
        et_list = []
        for i in inds_base:
            et = _etiq_ind(i)
            if et in et_map:
                et = f"{et} · {str(i)[:16]}"
            et_map[et] = i
            et_list.append(et)

        ind_et = st.selectbox(
            "Indicador",
            ["Todos"] + et_list,
            key="datos_ind",
        )
        anio_f = st.selectbox(
            "Año",
            ["Todos"] + anios_opts,
            key="datos_anio",
        )

    vista = df.copy()
    if banco_sel != "Todos":
        vista = vista[vista["Banco"] == banco_sel]
    if cat_sel != "Todas" and "CategoriaIndicador" in vista.columns:
        vista = vista[vista["CategoriaIndicador"] == cat_sel]
    if ind_et != "Todos":
        vista = vista[vista["Indicador"] == et_map[ind_et]]
    if anio_f != "Todos":
        vista = vista[vista["FechaReporte"].dt.year == int(anio_f)]

    vista_show = vista.sort_values(["FechaReporte", "Banco"], ascending=[False, True])
    st.caption(f"Mostrando **{len(vista_show):,}** de {total_registros:,} registros (vista filtrada)")
    st.dataframe(vista_show, use_container_width=True, height=480)

    # Exportar SOLO la vista filtrada
    st.markdown("---")
    st.markdown("**Exportar vista filtrada**")
    partes = []
    if banco_sel != "Todos":
        partes.append(str(banco_sel).replace(" ", "_")[:20])
    if ind_et != "Todos":
        partes.append(str(ind_et).replace(" ", "_")[:24])
    if anio_f != "Todos":
        partes.append(str(anio_f))
    base_name = "_".join(partes) if partes else "vista_cnbs"
    base_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in base_name)

    exp = vista_show.copy()
    if "FechaReporte" in exp.columns:
        exp["FechaReporte"] = exp["FechaReporte"].dt.strftime("%Y-%m-%d")

    d1, d2 = st.columns(2)
    with d1:
        try:
            xls = df_to_excel_bytes(exp)
            st.download_button(
                "⬇ Excel (.xlsx)",
                data=xls,
                file_name=f"{base_name}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_datos_xlsx",
            )
        except Exception as ex:
            st.caption(f"Excel: {ex}")
    with d2:
        csv_bytes = exp.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "⬇ CSV",
            data=csv_bytes,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            key="dl_datos_csv",
        )

st.markdown("---")
st.caption("Stack: Streamlit · Pandas · Groq (Llama 3.3) · Plotly · Datos CNBS · © 2026")
