import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
import io
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Azure Cost Forecast",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME / GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0f1e;
    border-right: 1px solid #1e2d4d;
}
section[data-testid="stSidebar"] * {
    color: #c8d8f0 !important;
}
section[data-testid="stSidebar"] label {
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5b8fc9 !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput input {
    background: #131929 !important;
    border: 1px solid #1e3a5f !important;
    color: #e0ecff !important;
    border-radius: 6px !important;
}
section[data-testid="stSidebar"] .stFileUploader {
    background: #0d1628 !important;
    border: 1px dashed #2a4a7f !important;
    border-radius: 8px !important;
    padding: 8px !important;
}

/* ── Main background ── */
.stApp {
    background: #f4f7fb;
}

/* ── Top header bar ── */
.top-header {
    background: linear-gradient(135deg, #0a2342 0%, #1a4a7a 100%);
    padding: 28px 36px 22px;
    border-radius: 12px;
    margin-bottom: 28px;
    border-left: 5px solid #2196F3;
}
.top-header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}
.top-header p {
    font-size: 0.85rem;
    color: #7bafd4;
    margin: 0;
    letter-spacing: 0.04em;
}

/* ── KPI cards ── */
.kpi-row { display: flex; gap: 16px; margin-bottom: 28px; }
.kpi-card {
    flex: 1;
    background: #ffffff;
    border: 1px solid #e0e8f5;
    border-top: 3px solid #2196F3;
    border-radius: 10px;
    padding: 20px 22px 16px;
}
.kpi-card.accent { border-top-color: #00bcd4; }
.kpi-card.warn   { border-top-color: #ff7043; }
.kpi-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7a93b8;
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.65rem;
    font-weight: 600;
    color: #0a2342;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #9aafcc;
    margin-top: 5px;
}

/* ── Section titles ── */
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5b8fc9;
    border-bottom: 1px solid #dbe8f5;
    padding-bottom: 8px;
    margin-bottom: 18px;
    margin-top: 28px;
}

/* ── Table style ── */
.styled-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
    background: #fff;
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #dbeafe;
}
.styled-table th {
    background: #0a2342;
    color: #c8d8f0;
    padding: 10px 14px;
    text-align: left;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
}
.styled-table td {
    padding: 9px 14px;
    border-bottom: 1px solid #eef3fb;
    color: #1a2e4a;
}
.styled-table tr:last-child td { border-bottom: none; }
.styled-table tr:hover td { background: #f0f6ff; }
.mono { font-family: 'IBM Plex Mono', monospace; }
.proj-row td { background: #fffbf0 !important; color: #7a4800; }
.proj-row td.mono { color: #b55a00; font-weight: 600; }

/* ── Badges ── */
.badge-hist {
    background: #e3edf7; color: #1a4a7a;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.68rem; font-weight: 600;
}
.badge-proj {
    background: #fff3e0; color: #bf5000;
    border-radius: 4px; padding: 2px 8px;
    font-size: 0.68rem; font-weight: 600;
}

/* ── Divider ── */
hr { border: none; border-top: 1px solid #dbe8f5; margin: 24px 0; }

/* ── Info box ── */
.info-box {
    background: #e8f4fd;
    border-left: 4px solid #2196F3;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #0d3a63;
    margin-bottom: 20px;
}
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #ffa000;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: #5d3a00;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="top-header">
    <h1>☁️ Azure Cost Forecast</h1>
    <p>Carga tu reporte de costos, elige cómo agrupar y proyecta los próximos meses con regresión lineal</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuración")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Archivo Excel (.xlsx)",
        type=["xlsx"],
        help="Exporta el reporte de costos de Azure Portal"
    )

    if uploaded_file:
        try:
            df_raw = pd.read_excel(uploaded_file)

            # Detectar columna de fecha
            date_cols = [c for c in df_raw.columns if df_raw[c].dtype in ['datetime64[ns]', '<M8[ns]', 'datetime64[us]']]
            if not date_cols:
                date_cols = [c for c in df_raw.columns if 'date' in c.lower() or 'fecha' in c.lower()]

            st.markdown("**Columna de Fecha**")
            date_col = st.selectbox("", date_cols if date_cols else df_raw.columns.tolist(), label_visibility="collapsed")

            # Detectar columna de costo
            cost_cols = [c for c in df_raw.columns if df_raw[c].dtype in [np.float64, np.int64] and c != date_col]
            cost_guess = [c for c in cost_cols if 'cost' in c.lower() or 'costo' in c.lower() or 'amount' in c.lower()]

            st.markdown("**Columna de Costo**")
            cost_col = st.selectbox("", cost_cols, index=cost_cols.index(cost_guess[0]) if cost_guess else 0, label_visibility="collapsed")

            # Columna de agrupación
            group_candidates = [c for c in df_raw.columns if c not in [date_col, cost_col]
                                and df_raw[c].dtype == object]

            st.markdown("**Columna de Agrupación**")
            group_col = st.selectbox("", group_candidates, label_visibility="collapsed")

            st.markdown("**Meses a proyectar**")
            n_months = st.number_input("", min_value=1, max_value=12, value=6, step=1, label_visibility="collapsed")

            st.markdown("---")
            run = st.button("▶  Generar Proyección", use_container_width=True, type="primary")

        except Exception as e:
            st.error(f"Error al leer el archivo: {e}")
            run = False
    else:
        st.markdown("""
        <div style="background:#0d1628;border-radius:8px;padding:16px;font-size:0.8rem;color:#5b8fc9;line-height:1.7;">
        <strong style="color:#7bafd4;">Pasos:</strong><br>
        1. Sube tu archivo <code>.xlsx</code><br>
        2. Selecciona las columnas<br>
        3. Define cuántos meses proyectar<br>
        4. Haz clic en <strong>Generar</strong>
        </div>
        """, unsafe_allow_html=True)
        run = False

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def linear_forecast(series: np.ndarray, n: int):
    """Regresión lineal simple → devuelve n predicciones y R²."""
    X = np.arange(len(series)).reshape(-1, 1)
    model = LinearRegression().fit(X, series)
    r2 = model.score(X, series)
    Xf = np.arange(len(series), len(series) + n).reshape(-1, 1)
    preds = np.maximum(model.predict(Xf), 0)
    return preds, r2, model.coef_[0]

def fmt_usd(val):
    return f"${val:,.2f}"

BLUE_DARK  = "#0a2342"
BLUE_MID   = "#1a6fc4"
BLUE_LIGHT = "#7bafd4"
ORANGE     = "#ff7043"
TEAL       = "#00bcd4"
BG         = "#f4f7fb"
GRID_C     = "#dbe8f5"

# ─────────────────────────────────────────────
# MAIN — NO FILE
# ─────────────────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div class="info-box">
        📂 <strong>Sube un archivo Excel</strong> en el panel izquierdo para comenzar el análisis.
        Puedes usar el reporte de costos exportado directamente desde <em>Azure Cost Management</em>.
    </div>
    """, unsafe_allow_html=True)

    # Demo placeholder
    col1, col2, col3 = st.columns(3)
    for col, label, val, sub in zip(
        [col1, col2, col3],
        ["COSTO HISTÓRICO TOTAL", "PROMEDIO MENSUAL", "MESES A PROYECTAR"],
        ["—", "—", "—"],
        ["Esperando datos…", "Esperando datos…", "Configura en sidebar"]
    ):
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{val}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────
# MAIN — PROCESS
# ─────────────────────────────────────────────
if run:
    try:
        df = df_raw.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df["_month"] = df[date_col].dt.to_period("M")

        # Agrupar por mes + grupo
        monthly = (df.groupby(["_month", group_col])[cost_col]
                     .sum()
                     .reset_index()
                     .rename(columns={cost_col: "cost"}))

        groups   = monthly[group_col].unique()
        months   = sorted(monthly["_month"].unique())
        last_m   = months[-1]
        fut_months = [last_m + i for i in range(1, n_months + 1)]

        # ── FORECAST ───────────────────────────────
        projections = []
        model_info  = []

        for grp in groups:
            sub = monthly[monthly[group_col] == grp].sort_values("_month")
            arr = sub["cost"].values

            if len(arr) < 2:
                preds = np.full(n_months, arr[-1] if len(arr) else 0.0)
                r2, slope = 0.0, 0.0
            else:
                preds, r2, slope = linear_forecast(arr, n_months)

            model_info.append({"group": grp, "r2": r2, "slope": slope, "total_hist": arr.sum()})
            for i, fm in enumerate(fut_months):
                projections.append({"_month": fm, group_col: grp, "cost": preds[i], "type": "Proyección"})

        df_proj = pd.DataFrame(projections)
        df_hist = monthly.copy(); df_hist["type"] = "Histórico"
        df_all  = pd.concat([df_hist, df_proj], ignore_index=True)
        df_mi   = pd.DataFrame(model_info)

        # ── KPI TOTALS ──────────────────────────────
        total_hist   = monthly["cost"].sum()
        monthly_avg  = monthly.groupby("_month")["cost"].sum().mean()
        total_proj   = df_proj["cost"].sum()
        proj_monthly = df_proj.groupby("_month")["cost"].sum()
        growth       = ((proj_monthly.mean() - monthly_avg) / monthly_avg * 100) if monthly_avg else 0

        # ── KPI ROW ─────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        for col, cls, label, val, sub in zip(
            [c1, c2, c3, c4],
            ["", "accent", "warn", ""],
            ["COSTO HISTÓRICO TOTAL", "TOTAL PROYECTADO", "TENDENCIA", "RECURSOS ANALIZADOS"],
            [fmt_usd(total_hist), fmt_usd(total_proj),
             f"{growth:+.1f}%", str(len(groups))],
            [f"{len(months)} meses de historial",
             f"Próximos {n_months} mes{'es' if n_months>1 else ''}",
             "Crecimiento mensual est.",
             f"en columna '{group_col}'"]
        ):
            col.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        # ── CHART 1: Total mensual histórico + proyectado ──
        st.markdown('<div class="section-title">Evolución Total de Costos</div>', unsafe_allow_html=True)

        hist_totals = monthly.groupby("_month")["cost"].sum()
        proj_totals = df_proj.groupby("_month")["cost"].sum()

        fig, ax = plt.subplots(figsize=(12, 3.8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        hx = [str(m) for m in hist_totals.index]
        px = [str(m) for m in proj_totals.index]
        hy = hist_totals.values
        py = proj_totals.values

        ax.plot(hx, hy, marker="o", linewidth=2.2, markersize=7,
                color=BLUE_MID, label="Histórico", zorder=3)
        ax.fill_between(range(len(hx)), hy, alpha=0.10, color=BLUE_MID)

        # Línea de proyección: parte desde el último punto histórico
        # x: [offset, offset+1, offset+2, ...] → len(px)+1 puntos
        # y: [hy[-1], py[0], py[1], ...] → len(px)+1 puntos  ✓
        offset = len(hx) - 1
        proj_x = [offset + i for i in range(len(px) + 1)]
        proj_y = [hy[-1]] + list(py)

        ax.plot(proj_x, proj_y,
                marker="s", linewidth=2.2, markersize=7,
                color=ORANGE, linestyle="--", label="Proyección", zorder=3)

        ax.set_xticks(range(len(hx) + len(px)))
        ax.set_xticklabels(hx + px, rotation=45, ha="right", fontsize=8.5,
                           fontfamily="monospace")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.tick_params(colors="#4a6a8a", labelsize=8.5)
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color(GRID_C)
        ax.yaxis.grid(True, color=GRID_C, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9, framealpha=0.5, facecolor="white")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── CHART 2: Top 8 recursos proyectados ────
        st.markdown('<div class="section-title">Top Recursos por Costo Proyectado</div>',
                    unsafe_allow_html=True)

        top_n = min(8, len(groups))
        top_proj_grp = (df_proj.groupby(group_col)["cost"].sum()
                               .sort_values(ascending=False)
                               .head(top_n))

        fig2, ax2 = plt.subplots(figsize=(12, 3.6))
        fig2.patch.set_facecolor(BG)
        ax2.set_facecolor(BG)

        colors = [BLUE_MID if i > 0 else ORANGE for i in range(top_n)][::-1]
        bars = ax2.barh(range(top_n), top_proj_grp.values[::-1],
                        color=colors, height=0.6, edgecolor="none")

        for bar, val in zip(bars, top_proj_grp.values[::-1]):
            ax2.text(bar.get_width() + max(top_proj_grp.values)*0.01,
                     bar.get_y() + bar.get_height()/2,
                     fmt_usd(val), va="center", fontsize=8.2,
                     fontfamily="monospace", color="#0a2342")

        labels = [g.split("/")[-1] if "/" in g else g for g in top_proj_grp.index[::-1]]
        ax2.set_yticks(range(top_n))
        ax2.set_yticklabels(labels, fontsize=8.5)
        ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax2.tick_params(colors="#4a6a8a", labelsize=8.5)
        ax2.spines[["top","right","bottom"]].set_visible(False)
        ax2.spines["left"].set_color(GRID_C)
        ax2.xaxis.grid(True, color=GRID_C, linewidth=0.8)
        ax2.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        # ── TABLE: Proyección detallada ─────────────
        st.markdown('<div class="section-title">Proyección Detallada por Recurso</div>',
                    unsafe_allow_html=True)

        # Pivot: recursos × meses (hist + proj)
        all_months_str = [str(m) for m in months] + [str(m) for m in fut_months]
        pivot = (df_all.groupby([group_col, "_month"])["cost"]
                       .sum()
                       .reset_index())
        pivot["_month_str"] = pivot["_month"].astype(str)
        table = (pivot.pivot(index=group_col, columns="_month_str", values="cost")
                      .fillna(0)
                      .reindex(columns=all_months_str, fill_value=0))
        table["Total Proyectado"] = table[[str(m) for m in fut_months]].sum(axis=1)
        table = table.sort_values("Total Proyectado", ascending=False)

        hist_cols = [str(m) for m in months]
        proj_cols = [str(m) for m in fut_months]

        header_html = "".join(
            f"<th>{c}</th>" for c in
            [group_col] + hist_cols + proj_cols + ["Total Proyectado"]
        )
        rows_html = ""
        for grp_name, row in table.head(20).iterrows():
            cells = f"<td>{grp_name}</td>"
            for c in hist_cols:
                cells += f'<td class="mono">{fmt_usd(row[c])}</td>'
            for c in proj_cols:
                cells += f'<td class="mono" style="color:#bf5000;font-weight:600">{fmt_usd(row[c])}</td>'
            cells += f'<td class="mono" style="font-weight:700;color:#0a2342">{fmt_usd(row["Total Proyectado"])}</td>'
            rows_html += f"<tr>{cells}</tr>"

        st.markdown(f"""
        <div style="overflow-x:auto;">
        <table class="styled-table">
            <thead><tr>{header_html}</tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
        </div>
        <p style="font-size:0.71rem;color:#9aafcc;margin-top:8px;">
            Columnas en <strong style="color:#1a6fc4">azul</strong> = histórico ·
            Columnas en <strong style="color:#bf5000">naranja</strong> = proyección (regresión lineal)
        </p>
        """, unsafe_allow_html=True)

        # ── TABLE: Resumen mensual ───────────────────
        st.markdown('<div class="section-title">Resumen Mensual</div>', unsafe_allow_html=True)

        rows_sum = ""
        for m in months:
            v = hist_totals.get(m, 0)
            rows_sum += f'<tr><td class="mono">{m}</td><td><span class="badge-hist">Histórico</span></td><td class="mono">{fmt_usd(v)}</td></tr>'
        for m in fut_months:
            v = proj_totals.get(m, 0)
            rows_sum += f'<tr><td class="mono">{m}</td><td><span class="badge-proj">Proyección</span></td><td class="mono" style="color:#bf5000;font-weight:600">{fmt_usd(v)}</td></tr>'

        st.markdown(f"""
        <div style="max-width:480px;">
        <table class="styled-table">
            <thead><tr><th>Mes</th><th>Tipo</th><th>Costo Total</th></tr></thead>
            <tbody>{rows_sum}</tbody>
        </table>
        </div>
        """, unsafe_allow_html=True)

        # ── DOWNLOAD ─────────────────────────────────
        st.markdown('<div class="section-title">Exportar Resultados</div>', unsafe_allow_html=True)

        df_export = df_all.copy()
        df_export["_month"] = df_export["_month"].astype(str)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df_export.to_excel(writer, sheet_name="Datos Completos", index=False)
            table.reset_index().to_excel(writer, sheet_name="Tabla Resumen", index=False)
            df_mi.to_excel(writer, sheet_name="Métricas Modelos", index=False)
        buf.seek(0)

        st.download_button(
            label="⬇  Descargar Excel con proyecciones",
            data=buf,
            file_name="azure_forecast_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=False,
        )

    except Exception as e:
        st.markdown(f"""
        <div class="warn-box">
            ⚠️ <strong>Error al procesar el archivo:</strong> {e}<br><br>
            Verifica que las columnas seleccionadas sean correctas e inténtalo de nuevo.
        </div>
        """, unsafe_allow_html=True)
        st.exception(e)

elif uploaded_file and not run:
    st.markdown("""
    <div class="info-box">
        ✅ Archivo cargado correctamente. Configura las columnas en el panel izquierdo
        y haz clic en <strong>▶ Generar Proyección</strong> para comenzar.
    </div>
    """, unsafe_allow_html=True)