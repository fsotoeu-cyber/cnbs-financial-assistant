# Sistema Analítico Financiero CNBS

**Sistema analítico conversacional híbrido** para el análisis de indicadores financieros del sistema hondureño, con datos oficiales de la [Comisión Nacional de Bancos y Seguros (CNBS)](https://www.cnbs.gob.hn/).

Los **cálculos son determinísticos mediante Pandas**. El LLM (Groq · Llama 3.3) solo genera explicaciones en lenguaje natural, **sujetas a validación** (no inventa cifras ni rankings).

Incluye **capacidades de agente** para interpretar la intención, enrutar la consulta y ejecutar operaciones sobre el dataset; **no** es un agente autónomo que planifique o calcule por sí mismo.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3-00A67E)](https://groq.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![ReportLab](https://img.shields.io/badge/ReportLab-PDF-orange)](https://www.reportlab.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![LangSmith](https://img.shields.io/badge/LangSmith-Observability-green)
![ETL](https://img.shields.io/badge/ETL-CNBS%20Pipeline-blue)

![Dashboard](docs/dashboard.png)

---

## 🚀 Aplicación en Producción

La aplicación está desplegada y operativa en **Streamlit Community Cloud**:

🔗 **Aplicación pública:** [https://cnbs-financial-assistant.streamlit.app/](https://cnbs-financial-assistant.streamlit.app/)

> ⚠️ **Nota:** Tras periodos de inactividad, la primera carga puede tomar unos segundos. Si la aplicación no responde de inmediato, recarga la página.

---

## Resultados del proyecto

| Métrica | Valor |
|---------|--------|
| Tipos de institución | **Bancos comerciales · Bancos estatales · Sociedades financieras** |
| Instituciones (aprox.) | **~25–30** según el cierre CNBS |
| Indicadores | **17** ratios oficiales |
| Registros | **~14,000+** (histórico mensual) |
| Periodo típico | **2024-01 → último cierre publicado** (p. ej. 2026-06) |
| Motor | **Pandas** (cálculo) + **Groq Llama 3.3** (redacción validada) |
| Exportación | PDF · PNG · Excel · CSV |
| Observabilidad | LangSmith *(opcional)* |
| Datos | Pipeline ETL con **quality gate** desde el dump oficial CNBS |

---

## Tecnologías

| Capa | Herramientas |
|------|----------------|
| Lenguaje | Python 3.10+ |
| Interfaz | Streamlit |
| Datos y cálculo | Pandas |
| ETL / fuente | Dump CKAN CNBS · validación de esquema |
| Visualización | Plotly · Kaleido (PNG) |
| LLM | Groq · Llama 3.3 70B · LangChain |
| Informes | ReportLab (PDF) · OpenPyXL (Excel) |
| Observabilidad | LangSmith *(opcional)* |

---

## Qué es (y qué no es)

| Sí | No |
|----|-----|
| Sistema **analítico conversacional híbrido** | Agente **autónomo** (ReAct / multi-agente / planificación libre) |
| **Enrutamiento** de intención y ejecución de consultas | LLM que **elige herramientas** o inventa planes |
| Cálculo **determinista** con Pandas | Cálculo delegado al modelo de lenguaje |
| LLM restringido a **redacción** + validación posterior | “ChatGPT con un CSV” sin gobernanza |

**En una entrevista:** *“La parte analítica no depende del LLM. El sistema interpreta la intención, determina la ruta y ejecuta operaciones sobre los datos; Pandas realiza los cálculos y el LLM queda restringido a la explicación, que después se valida.”*

---

## Características técnicas

- Arquitectura híbrida **Pandas + LLM** (cálculo y redacción desacoplados)
- **Routing de intención** (universo, tipo de consulta, top-N, indicadores)
- Cálculo **determinístico** de rankings, ratios y scores
- **Validación anti-alucinación** (ganador, ranking y cifras)
- Filtro por **tipo de institución** (comerciales / estatales / financieras)
- **Top-N** explícito (`top 3`, `los 3 bancos`, …)
- Score de equilibrio triple: `(ROE / Morosidad) × (Capital / 100)` *(métrica interna, no oficial CNBS)*
- Ranking ROE / morosidad e indicador de **eficiencia** (`Gastos de Administración / Ingresos Totales`)
- Centro de ayuda y glosario **sin consumir tokens**
- Exportación **PDF**, **PNG**, **Excel** y **CSV**
- Pipeline **ETL reproducible** con quality gate y delta check

---

## ¿Cómo evita alucinaciones?

El LLM **nunca calcula**.

1. Todo cálculo (promedios, rankings, ratios, scores) lo hace **Pandas** sobre el CSV CNBS.
2. El modelo **solo redacta** a partir de un contexto ya calculado.
3. Antes de mostrar la respuesta se valida ganador, orden del ranking y cifras del DataFrame.
4. Si hay discrepancia, se **reintenta o se corrige** de forma automática.
5. Si un indicador pedido **no existe** en el dataset, el sistema lo declara (no sustituye por otro).

```text
Pregunta
   │
   ▼
Detectar universo (comerciales / estatales / financieras)
   │
   ▼
Pandas  ── calcula (rankings, ratios, scores)
   │
   ├── respuesta directa (tablas / rankings)
   │
   └── si hace falta explicar
           │
           ▼
        LLM (solo redacta)
           │
           ▼
        Validador (ganador · cifras)
           │
           ▼
        UI / PDF
```

---

## Módulos de la aplicación

| Módulo | Descripción |
|--------|-------------|
| **💬 Asistente** | Consultas en lenguaje natural: ROA, ROE, mora, capital, liquidez, eficiencia, rankings y comparaciones |
| **📈 Tendencias** | Series temporales por tipo de institución, KPIs y hallazgos con Pandas |
| **📋 Datos** | Explorador del dataset con filtros y export Excel/CSV de la vista filtrada |

### Universo de datos

Por defecto el asistente analiza **bancos comerciales**.

| Tipo | Ejemplo de pregunta |
|------|---------------------|
| Bancos comerciales | `Top 3 bancos comerciales por adecuación de capital en 2026` |
| Bancos estatales | `Ranking de mora en bancos estatales 2025` |
| Sociedades financieras | `Financiera con menor mora en 2026` |
| Cruzar / todos | `Compara liquidez de BAC y COFISA` · `todos los tipos` |

---

## Ejemplo de conversación

**Usuario**

> ¿Qué banco tiene mejor relación rentabilidad-riesgo en 2025? Considera ROE y morosidad.

**Asistente**

> **FICENSA** — Ratio ROE/Mora **12.95** (ROE 9.30% · Mora 0.72%).  
> Ranking completo calculado por Pandas · ~0.02 s

**Usuario**

> Top 3 bancos comerciales por adecuación de capital en 2026

**Asistente**

> | # | Institución | Capital % |
> |---|-------------|-----------|
> | 1 | BANCOFINTER | 55.53 |
> | 2 | BANHCAFE | 30.27 |
> | 3 | AZTECA | 27.38 |

**Usuario**

> Compara la eficiencia (Gastos de Administración / Ingresos Totales) de BAC y Ficohsa en 2025

**Asistente**

> | # | Banco | Valor % |
> |---|--------|---------|
> | 1 | FICOHSA | 31.87 |
> | 2 | BAC CREDOMATIC | 27.29 |
>
> *(Ratio más alto = mayor gasto sobre ingresos.)*

---

## Pipeline ETL (extracción CNBS → producción)

El dataset de la app **no se edita a mano**. Se genera con un pipeline reproducible desde el dump oficial de datos abiertos de la CNBS (`etl_cnbs.py`).

### Diagrama de flujo ETL

```text
┌─────────────────────────────────────────────────────────────────┐
│                    FUENTE OFICIAL CNBS                          │
│  https://datos.cnbs.gob.hn/datastore/dump/<resource-id>         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. EXTRACCIÓN                                                  │
│     pd.read_csv(URL_CNBS)                                       │
│     · log de filas y columnas detectadas                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. VALIDACIÓN DE ESTRUCTURA (fuente)                           │
│     Columnas requeridas:                                        │
│     TipoInstitucion · Logo · FechaReporte ·                     │
│     Indicador · TipoIndicador · Saldo                           │
│     ✗ Si faltan → ABORT (no se toca el CSV de producción)       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. TRANSFORMACIÓN                                              │
│     · strip de textos                                           │
│     · filtro TIPOS_OK (comerciales · estatales · financieras) │
│     · FechaReporte → fin de mes (YYYY-MM-DD)                    │
│     · Saldo → numérico                                          │
│     · Logo → Banco · TipoIndicador → CategoriaIndicador         │
│     · dropna en llaves · drop_duplicates                        │
│     · sort determinista                                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. EDA                                                         │
│     Conteos por TipoInstitucion · n únicos · periodo · nulos    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. QUALITY GATE (validación de datos)                          │
│     Ver tabla de reglas abajo                                   │
│     ✗ Si falla → ABORT (CSV de producción intacto)              │
│     ✓ Si pasa → continúa                                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  6. DELTA CHECK                                                 │
│     Compara con indicadores_financieros_CNBS.csv previo         │
│     · ¿Nueva fecha máxima? · ¿Más filas?                        │
│     · Log: NUEVOS DATOS / SISTEMA AL DÍA                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  7. EXPORTACIÓN                                                 │
│     indicadores_financieros_CNBS.csv                            │
│     encoding utf-8-sig · sin índice · FechaReporte YYYY-MM-DD   │
│     (+ descarga automática en Google Colab)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  8. RESUMEN DE PRODUCCIÓN                                       │
│     Registros · instituciones · tipos · indicadores             │
└─────────────────────────────────────────────────────────────────┘
```

### Validación de datos (Quality Gate)

El gate se ejecuta **después** de transformar y **antes** de exportar. Si alguna regla falla, el pipeline lanza `ValueError` y **no sobrescribe** el CSV en producción.

| # | Regla | Criterio de fallo |
|---|--------|-------------------|
| 1 | Dataset no vacío | `df.empty` |
| 2 | Completitud de `Saldo` | Nulos en `Saldo` **> 5%** |
| 3 | Bancos clave presentes | Falta alguno de: BANCATLAN, BAC CREDOMATIC, FICOHSA, BANPAIS, BANCOCCI, LAFISE, AZTECA, FICENSA, BANHCAFE |
| 4 | Diversidad de tipos | Menos de **2** valores en `TipoInstitucion` |
| 5 | Fechas válidas | Cualquier `FechaReporte` nula tras el parseo |
| 6 | Cobertura de indicadores | Menos de **5** indicadores únicos |

**Validación de estructura (etapa 2, sobre el crudo CNBS)**

| Columna origen | Obligatoria |
|----------------|-------------|
| `TipoInstitucion` | Sí |
| `Logo` | Sí |
| `FechaReporte` | Sí |
| `Indicador` | Sí |
| `TipoIndicador` | Sí |
| `Saldo` | Sí |

### Esquema de salida (compatible 100% con `app.py`)

| Columna | Origen CNBS | Tipo | Descripción |
|---------|-------------|------|-------------|
| `Banco` | `Logo` | string | Nombre de la institución |
| `FechaReporte` | `FechaReporte` | date `YYYY-MM-DD` | Fin de mes del reporte |
| `Indicador` | `Indicador` | string | Nombre oficial del ratio |
| `CategoriaIndicador` | `TipoIndicador` | string | Categoría CNBS |
| `Saldo` | `Saldo` | float | Valor del indicador |
| `TipoInstitucion` | `TipoInstitucion` | string | Comerciales / Estatales / Financieras |

### Tipos de institución admitidos (`TIPOS_OK`)

```text
BANCOS COMERCIALES
BANCOS ESTATALES
SOCIEDADES FINANCIERAS
```

Cualquier otro tipo en el dump (p. ej. cooperativas) se **excluye** en la transformación.

### Fuente oficial

```text
https://datos.cnbs.gob.hn/datastore/dump/509e19c4-09d1-4f3d-9ec4-f7a6e874bb78
```

### Cómo ejecutar el pipeline

```bash
python etl_cnbs.py
# o, en Colab: pegar el script y ejecutar run_pipeline()
```

Salida esperada (resumen):

```text
✅ QUALITY GATE PASSED
Instituciones : …
Tipos         : 3
Indicadores   : 17
Periodo       : 2024-01-31 → 2026-06-30
Archivo generado : indicadores_financieros_CNBS.csv
```

Tras cada cierre mensual CNBS: **ETL → commit del CSV → redeploy Streamlit**.

---

## Documentación técnica (API interna)

No es una API HTTP pública: es la **capa de funciones** que usa `app.py` sobre el DataFrame. Útil para extender el motor o escribir tests.

### Carga y contexto

| Función | Entrada | Salida | Rol |
|---------|---------|--------|-----|
| `construir_catalogo(df)` | DataFrame | `dict` sinónimo → nombre oficial | Mapea aliases (ROE, mora, eficiencia, …) |
| `detectar_universo(query, bancos=None)` | texto | `"comerciales"` \| `"estatales"` \| `"financieras"` \| `"todos"` | Universo de trabajo |
| `filtrar_instituciones(df, universo, …)` | DataFrame | DataFrame filtrado | Aplica `TipoInstitucion` + excluye agregados |
| `extraer_anios(query)` | texto | `list[int]` | Años `20xx` en la pregunta |
| `extraer_top_n(query)` | texto | `int` \| `None` | `top 3`, `los 5`, etc. |
| `extraer_bancos(query)` | texto | `list[str]` | Alias → nombres canónicos |

### Planificación y routing

| Función | Entrada | Salida | Rol |
|---------|---------|--------|-----|
| `planificar(query, contexto, catalogo)` | consulta + estado | `(indicadores, tema, tipo, …, asc)` | Intención + indicadores |
| `detectar_tipo_consulta(query)` | texto | `"ranking"` \| `"comparar"` \| `"promedio"` \| … | Tipo de operación |
| `necesita_llm(df_res, query, meta)` | resultado | `bool` | ¿Redacción narrativa? |
| `indicadores_pedidos_no_disponibles(query, catalogo)` | texto | `list[str]` | Indicadores pedidos ausentes en el dataset |

### Cálculo (Pandas)

| Función | Parámetros principales | Salida | Rol |
|---------|------------------------|--------|-----|
| `ranking(df, indicadores, anio, top, ascending)` | indicador(es), año, top-N | DataFrame con `Ranking` | Ranking por `Saldo` |
| `promedio(df, bancos, indicadores, anio)` | filtros | DataFrame | Medias por banco/indicador |
| `comparar_bancos(df, bancos, indicadores, anio)` | lista bancos | DataFrame | Panel comparativo |
| `serie_temporal(df, bancos, indicadores)` | filtros | DataFrame ordenado por fecha | Series para Tendencias |
| `ranking_rentabilidad_riesgo(df, anio, top)` | año | DataFrame | Ratio ROE / morosidad |
| `ranking_equilibrio_triple(df, anio, top, bancos)` | año, opcional bancos | DataFrame | Score `(ROE/Mora)×(Capital/100)` |
| `ejecutar_consulta(df, indicadores, bancos, anios, tipo, **kwargs)` | plan | DataFrame | Orquestador de estrategias |

### Gobernanza LLM

| Función | Rol |
|---------|-----|
| `extraer_resultado(df_res)` | Extrae ganador, ranking, ratio, año (inmutable para el prompt) |
| `construir_contexto_llm(...)` | Arma el prompt con DATOS + REGLAS (sin recalcular) |
| `redactar_respuesta(llm, prompt, resultado)` | Invoca Groq + validación / reintento |
| `validar_respuesta` / checks de cifras | Rechaza si el texto contradice el ganador Pandas |

### Contrato de datos hacia el LLM

```text
RESULTADO DETERMINÍSTICO DE PANDAS (INMUTABLE)
  ganador · ranking · ratio · anio · bancos_en_resultado

DATOS EXACTOS (filas JSON)
  banco · indicador · valor_pct · …

REGLAS
  NO recalcular · NO cambiar ranking · NO inventar cifras
```

### Exportación

| Función / módulo | Salida |
|------------------|--------|
| `pdf_renderer.PDFRenderer.render(...)` | PDF (consulta, resumen, tabla, metodología) |
| Export Tendencias | PNG (Plotly/Kaleido) · PDF del gráfico |
| Export Datos | Excel / CSV de la **vista filtrada** |

---

## Arquitectura del motor

```text
                 ┌──────────────────────┐
                 │   Streamlit (UI)     │
                 │ Asistente·Tendencias │
                 │ ·Datos               │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     Consulta conversacional      Consulta financiera
     (sin Pandas / sin LLM)               │
                                          ▼
                               Clasificador de intención
                               + detectar_universo()
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    ▼                     ▼                     ▼
              Ranking               Comparar              Serie / promedio
           Top-N / Score         multi-indicador           temporal
                    │                     │                     │
                    └─────────────────────┴─────────────────────┘
                                          │
                                          ▼
                                   ┌─────────────┐
                                   │   Pandas    │
                                   │  (cálculo)  │
                                   └──────┬──────┘
                                          │
                           ┌──────────────┴──────────────┐
                           ▼                             ▼
                    Respuesta directa              ¿Necesita LLM?
                    (tablas / rankings)              Sí → redacta
                                                     + validador
```

**Stack:** Streamlit · Pandas · Plotly · Groq (Llama 3.3 70B) · ReportLab · LangSmith (opcional)

---

## Estructura del repositorio

```text
cnbs-financial-assistant/
├── app.py                              # Aplicación Streamlit (producción)
├── pdf_renderer.py                     # Informes PDF
├── etl_cnbs.py                         # Pipeline ETL CNBS → CSV
├── indicadores_financieros_CNBS.csv    # Dataset de producción
├── requirements.txt
├── LICENSE
├── README.md
└── docs/
    ├── dashboard.png
    ├── tendencias.png
    ├── datos.png
    ├── informe_pdf.png
    └── langsmith_trace.png             # opcional
```

---

## Instalación local

```bash
git clone https://github.com/<tu-usuario>/cnbs-financial-assistant.git
cd cnbs-financial-assistant
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Variables de entorno

| Variable | Obligatoria | Descripción |
|----------|-------------|-------------|
| `GROQ_API_KEY` | Sí (para redacción LLM) | Clave en [console.groq.com](https://console.groq.com) |
| `LANGCHAIN_API_KEY` | No | Trazas LangSmith |
| `LANGCHAIN_TRACING_V2` | No | `true` para activar tracing |
| `LANGCHAIN_PROJECT` | No | p. ej. `Agente-CNBS` |

```bash
export GROQ_API_KEY="gsk_..."
streamlit run app.py
```

Sin `GROQ_API_KEY`, el motor **Pandas directo** sigue respondiendo rankings y tablas; solo se desactiva la redacción narrativa.

---

## Deploy en Streamlit Community Cloud

1. Sube el repo a GitHub (`app.py`, `pdf_renderer.py`, `etl_cnbs.py`, CSV, `requirements.txt`).
2. En [share.streamlit.io](https://share.streamlit.io) → **New app** → selecciona el repo.
3. **Main file path:** `app.py`
4. **Secrets** (TOML):

```toml
GROQ_API_KEY = "gsk_..."
# Opcional:
# LANGCHAIN_API_KEY = "..."
# LANGCHAIN_TRACING_V2 = "true"
# LANGCHAIN_PROJECT = "Agente-CNBS"
```

5. Deploy. Tras actualizar el CSV con el ETL, haz push y Streamlit recarga los datos.

---

## Cómo preguntar

| Objetivo | Ejemplo |
|----------|---------|
| Ranking por mora | `¿Qué banco tiene mayor morosidad en 2025?` |
| Top-N | `Top 3 bancos comerciales por adecuación de capital en 2026` |
| ROE / mora | `¿Qué banco tiene mejor relación rentabilidad-riesgo en 2025?` |
| Score triple | `Score de equilibrio ROE + mora + capital en 2025` |
| Eficiencia | `Compara la eficiencia de BAC y Ficohsa en 2025` |
| Financieras | `Financiera con menor mora en 2026` |
| Estatales | `Ranking de mora en bancos estatales 2025` |
| Sistema | `Analiza el riesgo crediticio del sistema en 2025 (mora, cobertura y tarjetas)` |
| Ayuda | `¿Qué puedo preguntar?` |

**Consejos:** nombra banco, indicador y año; por defecto = comerciales; un objetivo por pregunta funciona mejor que frases multi-paso.

---

## Rendimiento (orientativo)

| Tipo de consulta | Tiempo típico |
|------------------|---------------|
| Ranking / comparación Pandas | ~0.02–0.08 s |
| Con redacción LLM | ~1–2 s |
| Ayuda / conversacional | ~0 s (sin tokens) |

---

## Limitaciones

- Solo indicadores presentes en el dataset oficial CNBS cargado.
- No realiza predicciones ni sustituye análisis regulatorios oficiales.
- El **score triple** es una métrica **interna** de comparación, no una metodología publicada por la CNBS.
- Agregados del sistema no son instituciones individuales.
- Universos muy pequeños se marcan como descriptivos, no como ranking competitivo.
- Consultas muy compuestas pueden requerir reformulación en pasos.
- **No** es un agente autónomo: no hay planificación libre multi-paso, ni tool-calling dinámico por el LLM, ni memoria de trabajo tipo agentic frameworks.

---

## Aprendizajes

- Arquitecturas híbridas **Pandas + LLM** (cálculo desacoplado de la redacción)  
- **Routing de intención** y ejecución determinística (capacidades de agente sin autonomía de cálculo)  
- Ingeniería de prompts y gobernanza del redactor  
- Validación anti-alucinación  
- ETL reproducible con quality gate desde datos abiertos  
- Streamlit · Plotly · ReportLab · LangSmith  
- Clasificación de intención y filtrado por tipo de institución  

---

## Licencia

MIT License — ver [LICENSE](LICENSE).

Proyecto desarrollado con fines **educativos**, demostración técnica y **portafolio profesional**. Los datos pertenecen a la Comisión Nacional de Bancos y Seguros (CNBS); este software no es un producto oficial de la CNBS.

---

## Créditos

- **Datos:** [Comisión Nacional de Bancos y Seguros (CNBS)](https://www.cnbs.gob.hn/) · [datos.cnbs.gob.hn](https://datos.cnbs.gob.hn/)  
- **Stack:** Streamlit · Pandas · Groq · Plotly · ReportLab · LangSmith  
- **© 2026** — Sistema Analítico Financiero CNBS  
