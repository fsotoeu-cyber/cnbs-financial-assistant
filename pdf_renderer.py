"""
pdf_renderer.py — Renderer universal de informes CNBS (Sistema Analítico v6.3).
reportlab si está disponible; si no, PDF mínimo con stdlib.
"""
from __future__ import annotations

import re
from datetime import datetime
from io import BytesIO
from typing import Any, Mapping, Optional

import pandas as pd

# Umbrales configurables (sin hardcode de bancos)
SCORE_EXCELENTE = 1.5
SCORE_BUENO = 0.8
FORMULA_SCORE_TRIPLE = "(ROE / Morosidad) × (Capital / 100)"


def _nivel_score(valor, excelente=SCORE_EXCELENTE, bueno=SCORE_BUENO):
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return "N/D", "#94A3B8"
    if v >= excelente:
        return "Excelente", "#16A34A"  # verde
    if v >= bueno:
        return "Bueno", "#CA8A04"  # ámbar
    return "Bajo", "#DC2626"  # rojo


def _detectar_tipo_ranking(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None or getattr(df, "empty", True):
        return None
    cols = set(df.columns)
    if "Score_Triple" in cols:
        return "triple"
    if "Ratio_ROE_Mora" in cols and "Ranking" in cols:
        return "roe_mora"
    if "Score_ROE_Capital" in cols and "Ranking" in cols:
        return "roe_capital"
    return None


def _conclusion_desde_df(df: Optional[pd.DataFrame]) -> Optional[str]:
    """Conclusión automática desde Pandas (ranking o comparación)."""
    if df is None or getattr(df, "empty", True) or "Banco" not in df.columns:
        return None
    top = df.iloc[0]
    banco = str(top["Banco"])
    n = len(df)
    anio = ""
    if "Año" in df.columns and pd.notna(top.get("Año")):
        anio = f" en {int(top['Año'])}"
    tipo = _detectar_tipo_ranking(df)
    if tipo == "triple":
        sc = float(top["Score_Triple"])
        nivel, _ = _nivel_score(sc)
        return (
            f"{banco} presenta el mejor equilibrio entre rentabilidad, riesgo crediticio "
            f"y solvencia entre los {n} bancos analizados{anio} "
            f"(Score = {sc:.2f}, nivel {nivel}; fórmula: {FORMULA_SCORE_TRIPLE})."
        )
    if tipo == "roe_mora":
        ratio = float(top["Ratio_ROE_Mora"])
        return (
            f"{banco} presenta la mejor relación rentabilidad–riesgo entre los {n} "
            f"bancos analizados{anio} (Ratio ROE/Mora = {ratio:.2f})."
        )
    if tipo == "roe_capital":
        sc = float(top["Score_ROE_Capital"])
        return (
            f"{banco} presenta el mejor equilibrio rentabilidad–solvencia entre los {n} "
            f"bancos analizados{anio} (Score = {sc:.2f})."
        )
    # Ranking explícito (una sola métrica)
    if any(c in df.columns for c in ("Ranking", "#")):
        try:
            rc = "Ranking" if "Ranking" in df.columns else "#"
            top = df.sort_values(rc).iloc[0]
            banco = str(top["Banco"])
            for vc in ("Valor %", "Saldo", "Valor_pct", "Liquidez"):
                if vc in df.columns and pd.notna(top.get(vc)):
                    return (
                        f"{banco} ocupa el primer lugar del ranking calculado por Pandas{anio} "
                        f"({float(top[vc]):.2f}%)."
                    )
        except Exception:
            pass
        return f"{banco} ocupa el primer lugar del ranking calculado por Pandas{anio}."

    # Comparación multi-métrica: solo si hay 2+ indicadores numéricos
    num_cols = [
        c for c in df.columns
        if c not in ("Banco", "Año", "Indicador", "Ranking", "#", "Valor %", "Valor_pct")
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(df["Banco"].unique()) >= 2 and num_cols and len(num_cols) >= 2:
        partes = []
        for _, row in df.iterrows():
            b = str(row["Banco"])
            vals = []
            for c in num_cols[:4]:
                try:
                    if pd.notna(row[c]):
                        vals.append(f"{c} {float(row[c]):.2f}%")
                except Exception:
                    pass
            if vals:
                partes.append(f"{b}: " + ", ".join(vals))
        if partes:
            return (
                "Comparación determinística (Pandas)" + anio + ": " + "; ".join(partes) + "."
            )

    if "Saldo" in df.columns and n == 1:
        return f"{banco}: {float(top['Saldo']):.2f}%{anio}."
    return None


def _esc_pdf_text(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _limpiar_md(texto: Any) -> str:
    t = str(texto or "")
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.MULTILINE)
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"\*(.+?)\*", r"\1", t)
    t = re.sub(r"`+", "", t)
    t = re.sub(r"\|[-:| ]+\|", "", t)
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _extraer_resumen_desde_texto(
    respuesta: str,
    dataframe: Optional[pd.DataFrame],
    consulta: Optional[str] = None,
) -> Optional[str]:
    """Resumen ejecutivo alineado al ranking Pandas (respeta menor/mayor)."""
    q = (consulta or "").lower()
    pide_menor = any(
        p in q
        for p in (
            "más baja", "mas baja", "más bajo", "mas bajo",
            "menor", "la menor", "el menor", "mínima", "minima",
        )
    )
    pide_mayor = any(
        p in q
        for p in (
            "más alta", "mas alta", "más alto", "mas alto",
            "mayor mora", "peor mora", "máxima", "maxima",
        )
    )

    if dataframe is not None and not dataframe.empty:
        cols = {c.lower(): c for c in dataframe.columns}
        tipo = _detectar_tipo_ranking(dataframe)
        df = dataframe.copy()
        if "Ranking" in df.columns:
            try:
                df = df.sort_values("Ranking", ascending=True)
            except Exception:
                pass
        top = df.iloc[0]
        banco = top[cols["banco"]] if "banco" in cols else None

        if tipo == "triple" and banco is not None:
            sc = float(top["Score_Triple"])
            nivel, _ = _nivel_score(sc)
            return (
                f"El análisis determinístico identifica a {banco} como la institución "
                f"con el mejor equilibrio rentabilidad–riesgo–solvencia "
                f"(Score = {sc:.2f}, nivel {nivel})."
            )
        if tipo == "roe_capital" and banco is not None:
            sc = float(top["Score_ROE_Capital"])
            return (
                f"El análisis determinístico identifica a {banco} como la institución "
                f"con el mejor equilibrio rentabilidad–solvencia (Score = {sc:.2f})."
            )
        if "banco" in cols and any("ratio" in c for c in cols):
            ratio_col = next(cols[c] for c in cols if "ratio" in c)
            ratio = top[ratio_col]
            try:
                ratio_s = f"{float(ratio):.2f}"
            except Exception:
                ratio_s = str(ratio)
            return (
                f"El análisis determinístico identifica a {banco} como la institución "
                f"con la mejor relación rentabilidad–riesgo, con un ratio ROE/Morosidad de {ratio_s}."
            )

        if "banco" in cols and "saldo" in cols and "Ranking" in df.columns:
            try:
                v = f"{float(top[cols['saldo']]):.2f}%"
            except Exception:
                v = str(top[cols["saldo"]])
            if pide_menor:
                return (
                    f"La menor morosidad (menor valor del indicador) corresponde a "
                    f"{banco} ({v}), según el ranking determinístico de Pandas."
                )
            if pide_mayor:
                return (
                    f"La mayor morosidad (mayor valor del indicador) corresponde a "
                    f"{banco} ({v}), según el ranking determinístico de Pandas."
                )
            return (
                f"{banco} ocupa el primer lugar del ranking calculado por Pandas "
                f"con un valor de {v}."
            )

        if "banco" in cols and any("mora" in c for c in cols):
            mora_col = next(cols[c] for c in cols if "mora" in c)
            asc = True if pide_menor else False
            orden = df.sort_values(mora_col, ascending=asc)
            top = orden.iloc[0]
            try:
                v = f"{float(top[mora_col]):.2f}%"
            except Exception:
                v = str(top[mora_col])
            banco_m = top[cols["banco"]]
            if asc:
                return (
                    f"La menor morosidad corresponde a {banco_m} ({v}), "
                    f"según los datos CNBS del periodo analizado."
                )
            return (
                f"La mayor morosidad corresponde a {banco_m} ({v}), "
                f"según los datos CNBS del periodo analizado."
            )

    limpio = _limpiar_md(respuesta or "")
    candidatos = []
    for line in limpio.split("\n"):
        line = line.strip()
        if len(line) > 40 and not line.startswith("|"):
            candidatos.append(line[:280])
    if pide_menor:
        for line in candidatos:
            if any(x in line.lower() for x in ("más baja", "mas baja", "menor", "0.38", "ficensa")):
                return line
    for line in candidatos:
        return line
    return None


def _pdf_stdlib(lines: list[str], title: str = "Informe CNBS") -> bytes:
    content_lines = [title, f"Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]
    for ln in lines:
        for chunk in str(ln).split("\n"):
            chunk = chunk.strip()
            if not chunk:
                content_lines.append("")
                continue
            while len(chunk) > 100:
                content_lines.append(chunk[:100])
                chunk = chunk[100:]
            content_lines.append(chunk)

    lines_per_page = 55
    pages = [
        content_lines[i : i + lines_per_page]
        for i in range(0, len(content_lines), lines_per_page)
    ]
    if not pages:
        pages = [[title]]

    page_streams = []
    for page_lines in pages:
        cmds = ["BT", "/F1 9 Tf", "50 800 Td", "11 TL"]
        for i, line in enumerate(page_lines):
            esc = _esc_pdf_text(line)
            if i == 0:
                cmds.append(f"({esc}) Tj")
            else:
                cmds.append(f"T* ({esc}) Tj")
        cmds.append("ET")
        page_streams.append("\n".join(cmds).encode("latin-1", errors="replace"))

    objs: list[bytes] = []
    objs.append(b"1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj\n")
    kids = " ".join(f"{3 + i * 2} 0 R" for i in range(len(page_streams)))
    objs.append(
        f"2 0 obj<< /Type /Pages /Kids [{kids}] /Count {len(page_streams)} >>endobj\n".encode(
            "latin-1"
        )
    )
    font_num = 3 + len(page_streams) * 2
    for i, stream in enumerate(page_streams):
        page_num = 3 + i * 2
        content_num = page_num + 1
        objs.append(
            (
                f"{page_num} 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_num} 0 R /Resources<< /Font<< /F1 {font_num} 0 R >> >> >>endobj\n"
            ).encode("latin-1")
        )
        objs.append(
            f"{content_num} 0 obj<< /Length {len(stream)} >>stream\n".encode("latin-1")
            + stream
            + b"\nendstream\nendobj\n"
        )
    objs.append(
        f"{font_num} 0 obj<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>endobj\n".encode(
            "latin-1"
        )
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode("latin-1"))
    out.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin-1"))
    out.extend(
        f"trailer<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    return bytes(out)


class PDFRenderer:
    """Renderer universal con portada, tarjetas y tabla moderna."""

    VERSION = "v6.3"

    def __init__(self) -> None:
        self._reportlab = False
        try:
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import cm
            from reportlab.platypus import (
                SimpleDocTemplate,
                Paragraph,
                Spacer,
                Table,
                TableStyle,
                HRFlowable,
                Image,
            )

            self.colors = colors
            self.cm = cm
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.Table = Table
            self.TableStyle = TableStyle
            self.HRFlowable = HRFlowable
            self.Image = Image

            styles = getSampleStyleSheet()
            self.styles = styles
            self.title = ParagraphStyle(
                "TitleCNBS",
                parent=styles["Heading1"],
                fontSize=18,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#0F3D91"),
                spaceAfter=4,
            )
            self.subtitle = ParagraphStyle(
                "SubCNBS",
                parent=styles["Normal"],
                fontSize=10,
                alignment=TA_CENTER,
                textColor=colors.HexColor("#475569"),
                spaceAfter=4,
            )
            self.section = ParagraphStyle(
                "SectionCNBS",
                parent=styles["Heading2"],
                fontSize=11,
                textColor=colors.HexColor("#1E3A8A"),
                spaceBefore=10,
                spaceAfter=4,
            )
            self.body = ParagraphStyle(
                "BodyCNBS",
                parent=styles["BodyText"],
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#1E293B"),
            )
            self._reportlab = True
        except Exception:
            self._reportlab = False

    def render(
        self,
        consulta: str,
        respuesta: str = "",
        dataframe: Optional[pd.DataFrame] = None,
        figure: Any = None,
        metadata: Optional[Mapping[str, Any]] = None,
        summary: Optional[str] = None,
        titulo: str = "INFORME FINANCIERO CNBS",
    ) -> bytes:
        if not summary:
            summary = _extraer_resumen_desde_texto(respuesta, dataframe, consulta=consulta)

        respuesta_limpia = _limpiar_md(respuesta)
        # Conservar prosa; quitar tablas markdown y filas de ranking si ya hay dataframe
        if dataframe is not None and not getattr(dataframe, "empty", True):
            lineas = []
            in_tbl = False
            for ln in respuesta_limpia.split("\n"):
                s = ln.strip()
                if not s:
                    lineas.append("")
                    in_tbl = False
                    continue
                if s.startswith("|") or re.match(r"^\|?[-:| ]+\|?$", s):
                    in_tbl = True
                    continue
                if in_tbl:
                    continue
                if re.match(r"^\d+\s+[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 .\-]{1,40}\s+[\d.\-]+", s):
                    continue
                if re.match(r"^(Ranking\s+Banco|Banco\s+ROE|Banco\s+Indicador|Banco\s+Liquidez)", s, re.I):
                    continue
                if re.match(r"^[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 .,]{1,40}\s+.+\s+\d+[.,]\d+", s) and "lidera" not in s.lower():
                    # posible fila tabular pegada
                    if any(x in s for x in ("%","Ranking","2024","2025","2026")):
                        continue
                lineas.append(s)
            tmp, prev = [], False
            for ln in lineas:
                emp = ln == ""
                if emp and prev:
                    continue
                tmp.append(ln)
                prev = emp
            respuesta_limpia = "\n".join(tmp).strip()
            if len(respuesta_limpia) < 80 and len(_limpiar_md(respuesta)) > 120:
                respuesta_limpia = _limpiar_md(respuesta)

        if self._reportlab:
            try:
                return self._render_reportlab(
                    consulta,
                    respuesta_limpia,
                    dataframe,
                    figure,
                    metadata,
                    summary,
                    titulo,
                )
            except Exception:
                pass
        return self._render_stdlib(
            consulta, respuesta_limpia, dataframe, metadata, summary, titulo
        )

    def _esc(self, t: Any) -> str:
        return (
            str(t or "")
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br/>")
        )

    def _add_page_number(self, canvas, doc) -> None:
        canvas.saveState()
        page = canvas.getPageNumber()
        text = (
            f"Sistema Analítico Financiero CNBS {self.VERSION}  ·  "
            f"Fuente: CNBS Honduras  ·  Página {page}"
        )
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(self.colors.HexColor("#64748B"))
        canvas.drawString(doc.leftMargin, 12, text)
        canvas.restoreState()

    def _section_title(self, story, label: str) -> None:
        story.append(
            self.Paragraph(
                f"<font color='#1E3A8A'><b>{label}</b></font>",
                self.section,
            )
        )
        story.append(
            self.HRFlowable(
                width="100%",
                thickness=0.6,
                color=self.colors.HexColor("#BFDBFE"),
            )
        )
        story.append(self.Spacer(1, 6))

    def _header(
        self,
        story,
        titulo: str,
        consulta: str = "",
        metadata: Optional[Mapping] = None,
    ) -> None:
        story.append(
            self.Paragraph(
                "<font size='20' color='#0F3D91'><b>INFORME FINANCIERO CNBS</b></font>",
                self.styles["Title"],
            )
        )
        story.append(
            self.Paragraph(
                "<font size='11' color='#475569'>Sistema Analítico Financiero CNBS</font>",
                self.styles["BodyText"],
            )
        )
        story.append(
            self.Paragraph(
                f"<font size='9' color='#64748B'>Generado automáticamente · "
                f"{datetime.now():%d/%m/%Y %H:%M}</font>",
                self.styles["BodyText"],
            )
        )
        story.append(self.Spacer(1, 8))
        story.append(
            self.HRFlowable(
                width="100%",
                thickness=1.3,
                color=self.colors.HexColor("#1D4ED8"),
            )
        )
        story.append(self.Spacer(1, 10))

        if consulta:
            story.append(
                self.Paragraph(
                    "<font color='#1E3A8A'><b>CONSULTA DEL INFORME</b></font>",
                    self.section,
                )
            )
            box = self.Table(
                [[self.Paragraph(self._esc(consulta), self.body)]],
                colWidths=[470],
            )
            box.setStyle(
                self.TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#EFF6FF")),
                        ("BOX", (0, 0), (-1, -1), 1, self.colors.HexColor("#93C5FD")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 12),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                        ("TOPPADDING", (0, 0), (-1, -1), 10),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                    ]
                )
            )
            story.append(box)
            story.append(self.Spacer(1, 12))

    def _metadata(self, story, meta: Optional[Mapping[str, Any]]) -> None:
        if not meta:
            return
        rows = []
        for k, v in meta.items():
            rows.append(
                [
                    self.Paragraph(f"<b>{self._esc(k)}</b>", self.body),
                    self.Paragraph(self._esc(v), self.body),
                ]
            )
        t = self.Table(rows, colWidths=[140, 330])
        t.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.6, self.colors.HexColor("#CBD5E1")),
                    ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#E2E8F0")),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(t)
        story.append(self.Spacer(1, 12))

    def _consulta(self, story, texto: str) -> None:
        pass

    def _winner_card(self, story, df: Optional[pd.DataFrame]) -> None:
        """Caja visual del banco recomendado (sin hardcode de nombres)."""
        tipo = _detectar_tipo_ranking(df)
        if not tipo or df is None or df.empty or "Banco" not in df.columns:
            return
        top = df.iloc[0]
        banco = str(top["Banco"])
        anio = ""
        if "Año" in df.columns and pd.notna(top.get("Año")):
            anio = f" · {int(top['Año'])}"

        if tipo == "triple":
            sc = float(top["Score_Triple"])
            nivel, color = _nivel_score(sc)
            titulo = f"🏆 BANCO RECOMENDADO{anio}"
            lineas_html = [
                f"<b><font size='12' color='#0F3D91'>{self._esc(titulo)}</font></b><br/><br/>",
                f"<b><font size='14'>{self._esc(banco)}</font></b><br/><br/>",
                f"<font color='{color}'><b>Score triple: {sc:.2f}</b> · {nivel}</font><br/>",
                f"ROE: {float(top['ROE']):.2f}% &nbsp;|&nbsp; "
                f"Morosidad: {float(top['Morosidad']):.2f}% &nbsp;|&nbsp; "
                f"Capital: {float(top['Capital']):.2f}%",
            ]
        elif tipo == "roe_mora":
            ratio = float(top["Ratio_ROE_Mora"])
            titulo = f"🏆 BANCO RECOMENDADO{anio}"
            lineas_html = [
                f"<b><font size='12' color='#0F3D91'>{self._esc(titulo)}</font></b><br/><br/>",
                f"<b><font size='14'>{self._esc(banco)}</font></b><br/><br/>",
                f"<b>Ratio ROE/Mora: {ratio:.2f}</b><br/>",
                f"ROE: {float(top['ROE']):.2f}% &nbsp;|&nbsp; "
                f"Morosidad: {float(top['Morosidad']):.2f}%",
            ]
        elif tipo == "roe_capital":
            sc = float(top["Score_ROE_Capital"])
            titulo = f"🏆 BANCO RECOMENDADO{anio}"
            lineas_html = [
                f"<b><font size='12' color='#0F3D91'>{self._esc(titulo)}</font></b><br/><br/>",
                f"<b><font size='14'>{self._esc(banco)}</font></b><br/><br/>",
                f"<b>Score ROE×Capital: {sc:.2f}</b><br/>",
                f"ROE: {float(top['ROE']):.2f}% &nbsp;|&nbsp; "
                f"Capital: {float(top['Capital']):.2f}%",
            ]
        else:
            return

        inner = self.Paragraph("".join(lineas_html), self.body)
        box = self.Table([[inner]], colWidths=[470])
        box.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#ECFDF5")),
                    ("BOX", (0, 0), (-1, -1), 1.5, self.colors.HexColor("#10B981")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 14),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(box)
        story.append(self.Spacer(1, 12))

    def _conclusion_box(self, story, df: Optional[pd.DataFrame]) -> None:
        texto = _conclusion_desde_df(df)
        if not texto:
            return
        self._section_title(story, "CONCLUSIÓN")
        inner = self.Paragraph(self._esc(texto), self.body)
        box = self.Table([[inner]], colWidths=[470])
        box.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#FFFBEB")),
                    ("BOX", (0, 0), (-1, -1), 1, self.colors.HexColor("#F59E0B")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(box)
        story.append(self.Spacer(1, 10))

    def _summary(self, story, texto: Optional[str]) -> None:
        if not texto:
            return
        titulo = "HALLAZGOS PRINCIPALES" if str(texto).lstrip().startswith("•") else "RESUMEN EJECUTIVO"
        self._section_title(story, titulo)
        inner = self.Paragraph(self._esc(texto), self.body)
        box = self.Table([[inner]], colWidths=[470])
        box.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#EFF6FF")),
                    ("BOX", (0, 0), (-1, -1), 1, self.colors.HexColor("#93C5FD")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 10),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ]
            )
        )
        story.append(box)
        story.append(self.Spacer(1, 10))

    def _respuesta(self, story, texto: str) -> None:
        if not texto:
            return
        label = "ANÁLISIS" if len(texto) > 180 else "RESULTADO"
        self._section_title(story, label)
        box = self.Table(
            [[self.Paragraph(self._esc(texto), self.body)]],
            colWidths=[470],
        )
        box.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 1, self.colors.HexColor("#CBD5E1")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 12),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )
        story.append(box)
        story.append(self.Spacer(1, 10))

    def _table(self, story, df: Optional[pd.DataFrame]) -> None:
        if df is None or getattr(df, "empty", True):
            return
        view = df.head(40).copy()
        for rc in ("Ranking", "#"):
            if rc in view.columns:
                try:
                    view = view.sort_values(rc, ascending=True)
                except Exception:
                    pass
                break
        for c in view.columns:
            cname = str(c).lower()
            # Ranking / posición: enteros sin decimales
            if cname in ("ranking", "#", "rank", "pos", "posicion", "posición"):
                view[c] = view[c].map(
                    lambda x: str(int(float(x))) if pd.notna(x) else ""
                )
            elif pd.api.types.is_integer_dtype(view[c]) and cname in ("año", "anio", "year"):
                view[c] = view[c].map(
                    lambda x: str(int(x)) if pd.notna(x) else ""
                )
            elif pd.api.types.is_float_dtype(view[c]) or pd.api.types.is_integer_dtype(view[c]):
                view[c] = view[c].map(
                    lambda x: f"{float(x):.2f}" if pd.notna(x) else ""
                )
            else:
                view[c] = view[c].astype(str).str[:36]

        headers = [str(c) for c in view.columns]
        datos = [headers] + view.values.tolist()
        tabla = self.Table(datos, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), self.colors.HexColor("#1E40AF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), self.colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
            ("TOPPADDING", (0, 0), (-1, 0), 10),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
            ("TOPPADDING", (0, 1), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("GRID", (0, 0), (-1, -1), 0.25, self.colors.HexColor("#CBD5E1")),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [self.colors.white, self.colors.HexColor("#F8FAFC")],
            ),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        if len(headers) > 1:
            style_cmds.append(("ALIGN", (1, 1), (-1, -1), "RIGHT"))
            style_cmds.append(("ALIGN", (1, 0), (-1, 0), "CENTER"))
        if len(datos) > 1:
            style_cmds.append(
                ("BACKGROUND", (0, 1), (-1, 1), self.colors.HexColor("#DBEAFE"))
            )
            style_cmds.append(("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"))

        tabla.setStyle(self.TableStyle(style_cmds))
        self._section_title(story, "TABLA DE RESULTADOS")
        story.append(tabla)
        if len(df) > 40:
            story.append(
                self.Paragraph(
                    f"<i>Mostrando 40 de {len(df)} filas.</i>",
                    self.body,
                )
            )
        story.append(self.Spacer(1, 10))

    def _figure(self, story, figure: Any) -> None:
        if figure is None:
            return
        img_src = None
        if isinstance(figure, (bytes, bytearray)):
            img_src = BytesIO(figure)
        elif isinstance(figure, str):
            img_src = figure
        elif hasattr(figure, "read"):
            img_src = figure
        if img_src is None:
            return
        self._section_title(story, "GRÁFICO")
        try:
            story.append(self.Image(img_src, width=460, height=250))
        except Exception:
            story.append(
                self.Paragraph(
                    "<i>Gráfico no disponible en este PDF.</i>",
                    self.body,
                )
            )
        story.append(self.Spacer(1, 8))

    def _methodology(self, story, metadata=None) -> None:
        motor_raw = "Pandas"
        if metadata and isinstance(metadata, dict):
            motor_raw = str(metadata.get("Motor", metadata.get("motor", "Pandas")))
        mlow = motor_raw.lower()
        if "plotly" in mlow or "tendencia" in mlow:
            motor_label = "Pandas + Plotly"
            texto = (
                "Las estadísticas descriptivas (promedio, máximo, mínimo y observaciones) "
                "fueron calculadas de forma determinística mediante Pandas sobre el dataset "
                "oficial de la Comisión Nacional de Bancos y Seguros (CNBS). "
                "La visualización fue generada automáticamente con Plotly, sin modelos de lenguaje."
            )
        elif "llm" in mlow or "groq" in mlow or "llama" in mlow or "gpt-oss" in mlow:
            motor_label = "Pandas + Groq (gpt-oss-120b)"
            texto = (
                "Los cálculos (rankings, ratios, promedios y comparaciones) fueron realizados "
                "de forma determinística mediante Pandas sobre el dataset oficial de la CNBS. "
                "Groq (gpt-oss-120b) se utilizó únicamente para redactar la interpretación, "
                "sin modificar cifras, rankings ni resultados."
            )
        else:
            motor_label = "Pandas (Determinístico)"
            texto = (
                "Los cálculos (rankings, ratios, promedios y comparaciones) fueron realizados "
                "de forma determinística mediante Pandas sobre el dataset oficial de la CNBS. "
                "No intervino ningún modelo de lenguaje en esta respuesta."
            )
        self._last_motor_label = motor_label
        self._section_title(story, "METODOLOGÍA")
        box = self.Table(
            [[self.Paragraph(self._esc(texto), self.body)]],
            colWidths=[470],
        )
        box.setStyle(
            self.TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), self.colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.6, self.colors.HexColor("#CBD5E1")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 10),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(box)
        story.append(self.Spacer(1, 10))

    def _footer_block(self, story, metadata=None) -> None:
        motor_label = getattr(self, "_last_motor_label", None)
        if not motor_label and metadata and isinstance(metadata, dict):
            mlow = str(metadata.get("Motor", "")).lower()
            if "plotly" in mlow:
                motor_label = "Pandas + Plotly"
            elif "llm" in mlow or "groq" in mlow or "gpt-oss" in mlow or "llama" in mlow:
                motor_label = "Pandas + Groq (gpt-oss-120b)"
            else:
                motor_label = "Pandas (Determinístico)"
        if not motor_label:
            motor_label = "Pandas (Determinístico)"
        story.append(
            self.HRFlowable(
                width="100%",
                thickness=0.6,
                color=self.colors.HexColor("#94A3B8"),
            )
        )
        story.append(self.Spacer(1, 6))
        story.append(
            self.Paragraph(
                "<font color='#64748B'>"
                "Fuente: Comisión Nacional de Bancos y Seguros (CNBS)<br/>"
                f"Motor de análisis: {self._esc(motor_label)}<br/>"
                f"Generado automáticamente por Sistema Analítico Financiero CNBS {self.VERSION}"
                "</font>",
                self.styles["Italic"],
            )
        )

    def _render_reportlab(
        self,
        consulta,
        respuesta,
        dataframe,
        figure,
        metadata,
        summary,
        titulo,
    ) -> bytes:
        buffer = BytesIO()
        doc = self.SimpleDocTemplate(
            buffer,
            leftMargin=1.5 * self.cm,
            rightMargin=1.5 * self.cm,
            topMargin=1.3 * self.cm,
            bottomMargin=1.8 * self.cm,
        )
        story = []
        self._header(story, titulo, consulta=consulta, metadata=metadata)
        self._metadata(story, metadata)
        self._summary(story, summary)
        self._winner_card(story, dataframe)
        self._figure(story, figure)
        self._respuesta(story, respuesta)
        self._table(story, dataframe)
        self._conclusion_box(story, dataframe)
        self._methodology(story, metadata)
        self._footer_block(story, metadata)
        doc.build(
            story,
            onFirstPage=self._add_page_number,
            onLaterPages=self._add_page_number,
        )
        return buffer.getvalue()

    def _render_stdlib(
        self,
        consulta,
        respuesta,
        dataframe,
        metadata,
        summary,
        titulo,
    ) -> bytes:
        lines = ["CONSULTA", _limpiar_md(consulta), ""]
        if summary:
            lines += ["RESUMEN EJECUTIVO", _limpiar_md(summary), ""]
        if respuesta:
            lines += ["RESULTADO", _limpiar_md(respuesta), ""]
        if metadata:
            lines.append("METADATOS")
            for k, v in metadata.items():
                lines.append(f"{k}: {v}")
            lines.append("")
        if dataframe is not None and not getattr(dataframe, "empty", True):
            lines.append("TABLA")
            cols = list(dataframe.columns)
            lines.append(" | ".join(str(c) for c in cols))
            for _, row in dataframe.head(25).iterrows():
                lines.append(" | ".join(str(row[c])[:18] for c in cols))
            lines.append("")
        lines.append(
            f"Sistema Analítico Financiero CNBS {self.VERSION} | Fuente: CNBS Honduras"
        )
        return _pdf_stdlib(lines, titulo)


_renderer: Optional[PDFRenderer] = None


def get_pdf_renderer() -> PDFRenderer:
    global _renderer
    if _renderer is None:
        _renderer = PDFRenderer()
    return _renderer


def generar_pdf_respuesta(
    consulta: str,
    respuesta: str = "",
    meta: Any = "",
    titulo: str = "Informe Financiero CNBS",
    dataframe: Optional[pd.DataFrame] = None,
    figure: Any = None,
    summary: Optional[str] = None,
) -> bytes:
    """API compatible con app.py."""
    metadata = None
    if isinstance(meta, dict):
        metadata = meta
    elif meta:
        metadata = {"Info": str(meta)}
    titulo_final = titulo.upper() if titulo else "INFORME FINANCIERO CNBS"
    return get_pdf_renderer().render(
        consulta=consulta,
        respuesta=respuesta,
        dataframe=dataframe,
        figure=figure,
        metadata=metadata,
        summary=summary,
        titulo=titulo_final,
    )
