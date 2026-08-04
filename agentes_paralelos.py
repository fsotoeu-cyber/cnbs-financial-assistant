"""
agentes_paralelos.py – Agentes especializados en paralelo para análisis financiero profundo.
"""

import json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

# ============================================================
# PROMPTS PARA CADA AGENTE
# ============================================================

PROMPT_RIESGO = """
Eres un analista de riesgo crediticio especializado en el sistema bancario de Honduras.

Basado en los datos EXACTOS proporcionados, genera un análisis de riesgo de 100-150 palabras:

**INDICADORES A ANALIZAR:**
- Morosidad (Índice de Morosidad sobre Cartera Crediticia Total)
- Cobertura de mora (Índice de Cobertura de la Mora de Cartera)
- Cartera de tarjetas de crédito (participación en cartera total)

**ESTRUCTURA DE LA RESPUESTA:**
1. Resumen del nivel de riesgo del sistema (morosidad promedio, cobertura promedio).
2. Banco con mayor morosidad (identificar al más riesgoso).
3. Banco con mejor cobertura (menos vulnerable).
4. Observación sobre la exposición a tarjetas de crédito.

DATOS EXACTOS (NO MODIFICAR):
{datos}

REGLAS ESTRICTAS:
- Usa EXCLUSIVAMENTE los números de DATOS.
- No inventes cifras ni causas macroeconómicas.
- No recomiendes inversiones.
- Conclusión breve (2-3 líneas).
"""

PROMPT_RENTABILIDAD = """
Eres un analista de rentabilidad especializado en el sistema bancario de Honduras.

Basado en los datos EXACTOS proporcionados, genera un análisis de rentabilidad de 100-150 palabras:

**INDICADORES A ANALIZAR:**
- ROE (Rentabilidad sobre el Patrimonio)
- ROA (Rentabilidad sobre Activos)
- Spread de intermediación

**ESTRUCTURA DE LA RESPUESTA:**
1. Resumen de la rentabilidad del sistema (ROE/ROA promedio).
2. Banco con mayor rentabilidad (ROE más alto).
3. Banco con menor rentabilidad (si hay negativos, mencionar).
4. Observación sobre el spread (margen de intermediación).

DATOS EXACTOS (NO MODIFICAR):
{datos}

REGLAS ESTRICTAS:
- Usa EXCLUSIVAMENTE los números de DATOS.
- No inventes cifras ni causas macroeconómicas.
- No recomiendes inversiones.
- Conclusión breve (2-3 líneas).
"""

PROMPT_ESTABILIDAD = """
Eres un analista de solvencia y estabilidad especializado en el sistema bancario de Honduras.

Basado en los datos EXACTOS proporcionados, genera un análisis de solvencia de 100-150 palabras:

**INDICADORES A ANALIZAR:**
- Adecuación de capital (Índice de Adecuación de Capital)
- Liquidez (Índice de Liquidez / Cobertura de depósitos)

**ESTRUCTURA DE LA RESPUESTA:**
1. Resumen de la solvencia del sistema (capital promedio, liquidez promedio).
2. Banco con mayor capitalización (más sólido).
3. Banco con menor capitalización (más vulnerable).
4. Observación sobre la liquidez del sistema.

DATOS EXACTOS (NO MODIFICAR):
{datos}

REGLAS ESTRICTAS:
- Usa EXCLUSIVAMENTE los números de DATOS.
- No inventes cifras ni causas regulatorias.
- No recomiendes inversiones.
- Conclusión breve (2-3 líneas).
"""


def ejecutar_agente(llm, prompt, datos_str, nombre_agente):
    """
    Ejecuta un agente con un prompt específico y los datos.
    
    Args:
        llm: Modelo de lenguaje (Groq).
        prompt: Prompt del agente (con {datos}).
        datos_str: Datos formateados para el prompt (string).
        nombre_agente: Nombre del agente (para identificación).
    
    Returns:
        dict: {"agente": nombre, "respuesta": texto}
    """
    prompt_completo = prompt.format(datos=datos_str)
    
    try:
        respuesta = llm.invoke(prompt_completo)
        contenido = respuesta.content if hasattr(respuesta, "content") else str(respuesta)
        return {
            "agente": nombre_agente,
            "respuesta": contenido
        }
    except Exception as e:
        return {
            "agente": nombre_agente,
            "respuesta": f"⚠️ Error en agente {nombre_agente}: {str(e)}"
        }


def preparar_datos_para_agentes(df_pivote, top_n=5):
    """
    Prepara los datos para los prompts de los agentes.
    
    Args:
        df_pivote: DataFrame pivotado (Banco x Indicador).
        top_n: Número de bancos a mostrar en top/bottom.
    
    Returns:
        str: Texto formateado con estadísticas y top bancos.
    """
    if df_pivote is None or df_pivote.empty:
        return "No hay datos disponibles."
    
    # Estadísticas descriptivas (mínimo, máximo, media, mediana)
    stats = df_pivote.describe().round(2)
    stats_str = "📊 ESTADÍSTICAS DESCRIPTIVAS:\n" + stats.to_string()
    
    # Top N y Bottom N por cada columna (para dar contexto)
    top_bottom = []
    for col in df_pivote.columns:
        top = df_pivote.nlargest(top_n, col)
        bottom = df_pivote.nsmallest(top_n, col)
        top_bottom.append(f"\n🔹 {col}:\n  TOP {top_n}: {top[col].to_dict()}\n  BOTTOM {top_n}: {bottom[col].to_dict()}")
    
    top_bottom_str = "\n".join(top_bottom)
    
    return f"{stats_str}\n\n{top_bottom_str}"


def analisis_paralelo_3_agentes(llm, df_resultados, contexto=""):
    """
    Ejecuta 3 agentes en paralelo (riesgo, rentabilidad, estabilidad).
    
    Args:
        llm: Modelo de lenguaje (Groq).
        df_resultados: DataFrame con resultados de Pandas.
        contexto: Contexto de sesión (opcional).
    
    Returns:
        str: Informe integrado con los 3 análisis.
    """
    if df_resultados is None or df_resultados.empty:
        return "⚠️ No hay datos para generar el análisis."
    
    # 1. Pivotear para tener Banco x Indicador
    if "Indicador" in df_resultados.columns and "Saldo" in df_resultados.columns:
        piv = df_resultados.pivot_table(
            index="Banco",
            columns="Indicador",
            values="Saldo",
            aggfunc="mean"
        ).round(2)
    else:
        # Si ya está pivotado, usarlo directamente
        piv = df_resultados.copy()
        # Si tiene columna "Ranking" o similar, quitarla para el análisis
        for col in ["Ranking", "Año"]:
            if col in piv.columns:
                piv = piv.drop(columns=[col])
    
    # 2. Preparar datos para los prompts
    datos_str = preparar_datos_para_agentes(piv)
    
    # 3. Ejecutar los 3 agentes en paralelo
    with ThreadPoolExecutor(max_workers=3) as executor:
        futuro_riesgo = executor.submit(
            ejecutar_agente, llm, PROMPT_RIESGO, datos_str, "Riesgo"
        )
        futuro_renta = executor.submit(
            ejecutar_agente, llm, PROMPT_RENTABILIDAD, datos_str, "Rentabilidad"
        )
        futuro_estab = executor.submit(
            ejecutar_agente, llm, PROMPT_ESTABILIDAD, datos_str, "Estabilidad"
        )
        
        resultado_riesgo = futuro_riesgo.result()
        resultado_renta = futuro_renta.result()
        resultado_estab = futuro_estab.result()
    
    # 4. Supervisor: combinar resultados con formato limpio
    informe = f"""
# 📊 ANÁLISIS INTEGRAL DEL SISTEMA BANCARIO

{contexto if contexto else ""}

## 🔴 1. PERSPECTIVA DE RIESGO CREDITICIO

{resultado_riesgo['respuesta'].strip()}

---

## 🟢 2. PERSPECTIVA DE RENTABILIDAD

{resultado_renta['respuesta'].strip()}

---

## 🔵 3. PERSPECTIVA DE SOLVENCIA

{resultado_estab['respuesta'].strip()}

---

## 📝 CONCLUSIÓN INTEGRADA

Basado en el análisis desde las tres perspectivas (riesgo, rentabilidad y solvencia), 
el sistema bancario hondureño presenta las siguientes características clave:

- **Riesgo crediticio:** [resumen automático del agente de riesgo]
- **Rentabilidad:** [resumen automático del agente de rentabilidad]  
- **Solvencia:** [resumen automático del agente de solvencia]

*Análisis generado con Pandas + 3 Agentes LLM en paralelo.*
"""
    
    return informe


def analisis_paralelo_simple(llm, df_resultados, contexto=""):
    """
    Versión simplificada para integrar con el flujo existente.
    """
    return analisis_paralelo_3_agentes(llm, df_resultados, contexto)