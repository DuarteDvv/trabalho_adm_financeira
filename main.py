import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy import stats


# =================================================================
# HELPERS
# =================================================================

def extrair_precos(dados_brutos, tickers=None):
    """
    Extrai coluna de preço de fechamento do DataFrame retornado pelo yfinance.
    Lida com MultiIndex (múltiplos ativos), colunas planas (único ativo) e
    o novo formato do yfinance >= 0.2 onde Price/Ticker são os dois níveis.
    """
    if dados_brutos is None or dados_brutos.empty:
        return pd.DataFrame()

    cols = dados_brutos.columns

    # ── MultiIndex ──────────────────────────────────────────────────
    if isinstance(cols, pd.MultiIndex):
        nivel0 = [str(v) for v in cols.get_level_values(0)]
        nivel1 = [str(v) for v in cols.get_level_values(1)]

        # Formato novo yfinance: (Price, Ticker) — primeiro nível é o campo
        if any(p in nivel0 for p in ("Close", "Adj Close", "Open")):
            campo = "Adj Close" if "Adj Close" in nivel0 else "Close"
            precos = dados_brutos[campo]
        # Formato antigo: (Ticker, Price) — segundo nível é o campo
        elif any(p in nivel1 for p in ("Close", "Adj Close")):
            campo = "Adj Close" if "Adj Close" in nivel1 else "Close"
            precos = dados_brutos.xs(campo, axis=1, level=1)
        else:
            return pd.DataFrame()

        if isinstance(precos, pd.Series):
            precos = precos.to_frame()

        # Renomear colunas para os tickers originais quando possível
        if tickers and len(precos.columns) == len(tickers):
            precos.columns = tickers

        return precos

    # ── Colunas planas (único ativo) ────────────────────────────────
    col_names = [str(c) for c in cols]
    if "Adj Close" in col_names:
        precos = dados_brutos[["Adj Close"]]
    elif "Close" in col_names:
        precos = dados_brutos[["Close"]]
    else:
        return pd.DataFrame()

    # Renomear para o ticker se fornecido
    if tickers and len(tickers) == 1:
        precos.columns = tickers

    return precos


def calcular_metricas(retornos, rf_anual):
    retorno_anual = retornos.mean() * 252
    risco_anual = retornos.std() * np.sqrt(252)
    downside = retornos.clip(upper=0)
    downside_risco = downside.std() * np.sqrt(252)
    risco_anual = risco_anual.replace(0, np.nan)
    downside_risco = downside_risco.replace(0, np.nan)
    sharpe = (retorno_anual - rf_anual) / risco_anual
    sortino = (retorno_anual - rf_anual) / downside_risco
    acumulado = (1 + retornos).cumprod()
    max_drawdown = (acumulado / acumulado.cummax() - 1).min()
    cagr = acumulado.iloc[-1] ** (252 / len(retornos)) - 1
    retorno_total = acumulado.iloc[-1] - 1
    calmar = cagr / abs(max_drawdown.replace(0, np.nan))
    var_95 = retornos.quantile(0.05)
    cvar_95 = retornos.apply(lambda col: col[col <= col.quantile(0.05)].mean())
    skew = retornos.apply(lambda x: stats.skew(x.dropna()))
    kurt = retornos.apply(lambda x: stats.kurtosis(x.dropna()))
    return {
        "retorno_anual": retorno_anual,
        "risco_anual": risco_anual,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "cagr": cagr,
        "retorno_total": retorno_total,
        "acumulado": acumulado,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": skew,
        "kurt": kurt,
    }


def calcular_vpl(investimento_inicial, fluxos_caixa, taxa_desconto_percentual):
    """
    Calcula o Valor Presente Líquido (VPL) de um projeto.
    O investimento inicial é tratado como saída de caixa no período zero.
    """
    taxa = taxa_desconto_percentual / 100
    vpl = -investimento_inicial

    for periodo, fluxo in enumerate(fluxos_caixa, start=1):
        vpl += fluxo / ((1 + taxa) ** periodo)

    return vpl


def calcular_tir(investimento_inicial, fluxos_caixa, tolerancia=1e-6, max_iteracoes=200):
    """
    Calcula a Taxa Interna de Retorno (TIR) pelo método da bisseção.
    Retorna NaN quando não há troca de sinal suficiente para encontrar raiz.
    """
    fluxos_completos = [-investimento_inicial] + list(fluxos_caixa)

    def funcao_vpl(taxa):
        return sum(fluxo / ((1 + taxa) ** periodo) for periodo, fluxo in enumerate(fluxos_completos))

    taxa_minima = -0.9999
    taxa_maxima = 10.0
    valor_minimo = funcao_vpl(taxa_minima)
    valor_maximo = funcao_vpl(taxa_maxima)

    if np.sign(valor_minimo) == np.sign(valor_maximo):
        return np.nan

    taxa_media = np.nan
    for _ in range(max_iteracoes):
        taxa_media = (taxa_minima + taxa_maxima) / 2
        valor_medio = funcao_vpl(taxa_media)

        if abs(valor_medio) < tolerancia:
            return taxa_media

        if np.sign(valor_medio) == np.sign(valor_minimo):
            taxa_minima = taxa_media
            valor_minimo = valor_medio
        else:
            taxa_maxima = taxa_media

    return taxa_media


def calcular_payback_simples(investimento_inicial, fluxos_caixa):
    """
    Calcula o payback simples, permitindo retorno fracionado dentro do ano.
    """
    saldo_acumulado = -investimento_inicial

    for periodo, fluxo in enumerate(fluxos_caixa, start=1):
        saldo_anterior = saldo_acumulado
        saldo_acumulado += fluxo

        if saldo_acumulado >= 0 and fluxo != 0:
            fracao = abs(saldo_anterior) / fluxo
            return (periodo - 1) + fracao

    return np.nan


def calcular_payback_descontado(investimento_inicial, fluxos_caixa, taxa_desconto_percentual):
    """
    Calcula o payback descontado com base na taxa mínima de atratividade.
    """
    taxa = taxa_desconto_percentual / 100
    saldo_acumulado = -investimento_inicial

    for periodo, fluxo in enumerate(fluxos_caixa, start=1):
        fluxo_descontado = fluxo / ((1 + taxa) ** periodo)
        saldo_anterior = saldo_acumulado
        saldo_acumulado += fluxo_descontado

        if saldo_acumulado >= 0 and fluxo_descontado != 0:
            fracao = abs(saldo_anterior) / fluxo_descontado
            return (periodo - 1) + fracao

    return np.nan


def montar_tabela_projeto(investimento_inicial, fluxos_caixa, taxa_desconto_percentual):
    """
    Gera a tabela consolidada de fluxos do projeto com valores nominais,
    descontados e acumulados para exibição e exportação.
    """
    taxa = taxa_desconto_percentual / 100
    periodos = list(range(0, len(fluxos_caixa) + 1))
    fluxos = [-investimento_inicial] + list(fluxos_caixa)
    fluxos_descontados = [fluxo / ((1 + taxa) ** periodo) for periodo, fluxo in enumerate(fluxos)]

    df_projeto = pd.DataFrame({
        "Período": periodos,
        "Fluxo de Caixa (R$)": fluxos,
        "Fluxo Descontado (R$)": fluxos_descontados,
    })
    df_projeto["Acumulado (R$)"] = df_projeto["Fluxo de Caixa (R$)"].cumsum()
    df_projeto["Acumulado Descontado (R$)"] = df_projeto["Fluxo Descontado (R$)"].cumsum()
    return df_projeto


# =================================================================
# PALETA E CONFIGURAÇÃO DE PÁGINA
# =================================================================
PALETTE = {
    "bg": "#0B0F19",
    "panel": "#111827",
    "panel2": "#1a2235",
    "border": "#1E2D45",
    "text": "#E8EDF5",
    "muted": "#6B7FA3",
    "accent": "#22D3EE",       # cyan
    "accent2": "#F59E0B",      # amber
    "accent3": "#A78BFA",      # violet
    "positive": "#34D399",
    "negative": "#F87171",
}

st.set_page_config(
    page_title="FinRisk — UFMG · CAD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {{
    background: {PALETTE['bg']} !important;
    color: {PALETTE['text']};
    font-family: 'Space Grotesk', sans-serif;
}}
[data-testid="stAppViewContainer"] > .main {{
    background: {PALETTE['bg']};
}}
.block-container {{
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
}}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: #0d1422 !important;
    border-right: 1px solid {PALETTE['border']};
}}
section[data-testid="stSidebar"] .block-container {{
    padding: 1.5rem 1.2rem;
}}
section[data-testid="stSidebar"] * {{
    color: {PALETTE['text']} !important;
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stNumberInput label,
section[data-testid="stSidebar"] .stCheckbox label {{
    font-size: 0.78rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {PALETTE['muted']} !important;
    font-weight: 500;
}}

/* ── Inputs ── */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea,
[data-baseweb="select"] > div,
[data-baseweb="input"] > div {{
    background: #151f30 !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 8px !important;
    color: {PALETTE['text']} !important;
}}
[data-baseweb="tag"] {{
    background: {PALETTE['accent']}22 !important;
    border: 1px solid {PALETTE['accent']}55 !important;
    color: {PALETTE['accent']} !important;
    border-radius: 6px !important;
}}

/* ── Metrics cards ── */
div[data-testid="stMetric"] {{
    background: {PALETTE['panel']} !important;
    border: 1px solid {PALETTE['border']};
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    position: relative;
    overflow: hidden;
}}
div[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {PALETTE['accent']}, {PALETTE['accent3']});
    border-radius: 14px 14px 0 0;
}}
div[data-testid="stMetric"] label {{
    color: {PALETTE['muted']} !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {PALETTE['text']} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 500;
}}
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {{
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    background: transparent;
    border-bottom: 1px solid {PALETTE['border']};
    padding-bottom: 0;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    border: none;
    border-radius: 0;
    color: {PALETTE['muted']};
    font-size: 0.82rem;
    letter-spacing: 0.05em;
    padding: 10px 18px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500;
    border-bottom: 2px solid transparent;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    color: {PALETTE['accent']} !important;
    border-bottom: 2px solid {PALETTE['accent']} !important;
    background: transparent !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: {PALETTE['text']} !important;
}}

/* ── Tables ── */
.stTable, table {{
    background: {PALETTE['panel']} !important;
    border-radius: 12px;
    overflow: hidden;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
}}
thead tr th {{
    background: {PALETTE['panel2']} !important;
    color: {PALETTE['muted']} !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-bottom: 1px solid {PALETTE['border']} !important;
}}
tbody tr td {{
    color: {PALETTE['text']} !important;
    border-bottom: 1px solid {PALETTE['border']}44 !important;
}}
tbody tr:hover td {{
    background: {PALETTE['panel2']} !important;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {PALETTE['accent']}, {PALETTE['accent3']});
    color: #000 !important;
    border: none;
    border-radius: 10px;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.05em;
    padding: 0.65rem 1.4rem;
    width: 100%;
    transition: all 0.2s;
}}
.stButton > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 8px 25px {PALETTE['accent']}55;
}}

/* ── Download button ── */
.stDownloadButton > button {{
    background: transparent !important;
    border: 1px solid {PALETTE['border']} !important;
    color: {PALETTE['muted']} !important;
    border-radius: 8px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
}}
.stDownloadButton > button:hover {{
    border-color: {PALETTE['accent']} !important;
    color: {PALETTE['accent']} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {PALETTE['panel']} !important;
    border: 1px solid {PALETTE['border']} !important;
    border-radius: 12px !important;
}}
[data-testid="stExpander"] summary {{
    color: {PALETTE['muted']} !important;
    font-size: 0.82rem;
}}

/* ── Spinner / info / error ── */
.stSpinner > div > div {{
    border-top-color: {PALETTE['accent']} !important;
}}
.stAlert {{
    background: {PALETTE['panel']} !important;
    border-radius: 10px !important;
    border: 1px solid {PALETTE['border']} !important;
    color: {PALETTE['text']} !important;
}}

/* ── Divider ── */
hr {{
    border-color: {PALETTE['border']} !important;
    margin: 1.5rem 0;
}}

/* ── Caption ── */
.stCaption, small {{
    color: {PALETTE['muted']} !important;
    font-size: 0.75rem;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: {PALETTE['bg']}; }}
::-webkit-scrollbar-thumb {{ background: {PALETTE['border']}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)

# =================================================================
# PLOTLY THEME
# =================================================================
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color=PALETTE["text"], size=12),
    xaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"])),
    yaxis=dict(gridcolor=PALETTE["border"], linecolor=PALETTE["border"], tickfont=dict(color=PALETTE["muted"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["border"]),
    margin=dict(l=40, r=20, t=40, b=40),
)
COLORS_SEQ = [PALETTE["accent"], PALETTE["accent2"], PALETTE["accent3"],
              PALETTE["positive"], PALETTE["negative"], "#60A5FA", "#FB923C", "#E879F9"]

tickers_selecionados = []
fluxos_caixa_projeto = []

# =================================================================
# SIDEBAR
# =================================================================
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:1.5rem;'>
        <svg width='44' height='44' viewBox='0 0 44 44' fill='none' xmlns='http://www.w3.org/2000/svg'>
            <rect width='44' height='44' rx='10' fill='#0d1e33'/>
            <rect x='8' y='28' width='5' height='9' rx='2' fill='#22D3EE'/>
            <rect x='16' y='20' width='5' height='17' rx='2' fill='#A78BFA'/>
            <rect x='24' y='14' width='5' height='23' rx='2' fill='#22D3EE' opacity='0.7'/>
            <rect x='32' y='8' width='5' height='29' rx='2' fill='#F59E0B'/>
            <polyline points='10.5,27 18.5,19 26.5,13 34.5,7' stroke='#34D399' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>
        </svg>
        <div>
            <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:800;color:#E8EDF5;'>FinRisk</div>
            <div style='font-size:0.68rem;color:#6B7FA3;letter-spacing:0.1em;text-transform:uppercase;'>UFMG · CAD</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    modulo_escolhido = st.selectbox(
        "Módulo do trabalho:",
        ["Trabalho 1 — Risco e Retorno", "Trabalho 2 — Projeto de Investimento"],
    )

    st.markdown("---")

    if modulo_escolhido == "Trabalho 1 — Risco e Retorno":
        st.markdown("#### 🇧🇷 Ações B3")
        ACOES_B3 = {
            "PETR4.SA — Petrobras PN": "PETR4.SA",
            "VALE3.SA — Vale ON": "VALE3.SA",
            "ITUB4.SA — Itaú PN": "ITUB4.SA",
            "WEGE3.SA — WEG ON": "WEGE3.SA",
            "ABEV3.SA — Ambev ON": "ABEV3.SA",
            "MGLU3.SA — Magazine Luiza": "MGLU3.SA",
            "BBDC4.SA — Bradesco PN": "BBDC4.SA",
            "BBAS3.SA — Banco do Brasil": "BBAS3.SA",
            "RENT3.SA — Localiza ON": "RENT3.SA",
            "LREN3.SA — Lojas Renner ON": "LREN3.SA",
            "GGBR4.SA — Gerdau PN": "GGBR4.SA",
            "SUZB3.SA — Suzano ON": "SUZB3.SA",
            "RDOR3.SA — Rede D'Or ON": "RDOR3.SA",
            "B3SA3.SA — B3 ON": "B3SA3.SA",
            "CPLE6.SA — Copel": "CPLE6.SA",
        }
        sel_acoes = st.multiselect(
            "Selecione ações:",
            options=list(ACOES_B3.keys()),
            default=["PETR4.SA — Petrobras PN", "VALE3.SA — Vale ON", "ITUB4.SA — Itaú PN", "WEGE3.SA — WEG ON"],
            key="acoes",
        )

        st.markdown("#### 🌎 ETFs & BDRs")
        ETFS = {
            "BOVA11.SA — ETF IBOV": "BOVA11.SA",
            "IVVB11.SA — ETF S&P 500": "IVVB11.SA",
            "HASH11.SA — ETF Crypto": "HASH11.SA",
            "GOLD11.SA — ETF Ouro": "GOLD11.SA",
            "FIXA11.SA — ETF Renda Fixa": "FIXA11.SA",
            "XINA11.SA — ETF China": "XINA11.SA",
            "EURP11.SA — ETF Europa": "EURP11.SA",
        }
        sel_etfs = st.multiselect("Selecione ETFs/BDRs:", options=list(ETFS.keys()), default=[], key="etfs")

        st.markdown("#### 🌐 Ativos Globais (USD)")
        GLOBAIS = {
            "AAPL — Apple": "AAPL",
            "MSFT — Microsoft": "MSFT",
            "GOOGL — Alphabet": "GOOGL",
            "AMZN — Amazon": "AMZN",
            "NVDA — NVIDIA": "NVDA",
            "TSLA — Tesla": "TSLA",
            "META — Meta": "META",
            "SPY — S&P 500 ETF": "SPY",
            "QQQ — Nasdaq ETF": "QQQ",
            "GLD — Gold ETF": "GLD",
            "BTC-USD — Bitcoin": "BTC-USD",
            "ETH-USD — Ethereum": "ETH-USD",
        }
        sel_globais = st.multiselect("Selecione ativos globais:", options=list(GLOBAIS.keys()), default=[], key="globais")

        st.markdown("---")
        st.markdown("#### ⚙️ Período")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            data_inicio = st.date_input("Início", datetime.now() - timedelta(days=365 * 3), label_visibility="visible")
        with col_d2:
            data_fim = st.date_input("Fim", datetime.now(), label_visibility="visible")

        st.markdown("#### 📐 Parâmetros")
        taxa_livre_risco = st.number_input("Taxa livre de risco (% a.a.):", min_value=0.0, max_value=25.0, value=10.75, step=0.25)
        benchmark_label = st.selectbox("Benchmark:", ["Nenhum", "IBOV (^BVSP)", "S&P 500 (^GSPC)", "Nasdaq (^IXIC)"])
        adicionar_carteira = st.checkbox("Incluir carteira igual peso", value=True)

        tickers_selecionados = (
            [ACOES_B3[a] for a in sel_acoes]
            + [ETFS[e] for e in sel_etfs]
            + [GLOBAIS[g] for g in sel_globais]
        )

    else:
        st.markdown("#### 🧾 Dados do Projeto")
        nome_projeto = st.text_input("Nome do projeto:", value="Expansão da unidade")
        investimento_inicial = st.number_input("Investimento inicial (R$):", min_value=0.0, value=80000.0, step=5000.0)
        taxa_desconto = st.number_input("TMA / taxa de desconto (% a.a.):", min_value=0.0, max_value=100.0, value=12.0, step=0.5)
        quantidade_periodos = st.slider("Quantidade de períodos anuais:", min_value=3, max_value=10, value=5)

        st.markdown("#### 💰 Fluxos de Caixa")
        for periodo in range(1, quantidade_periodos + 1):
            fluxo = st.number_input(
                f"Fluxo de caixa do ano {periodo} (R$):",
                value=float(20000 + (periodo - 1) * 4000),
                step=1000.0,
                key=f"fluxo_projeto_{periodo}",
            )
            fluxos_caixa_projeto.append(fluxo)

    st.markdown("---")
    executar = st.button("▶  Gerar Análise", use_container_width=True)

if modulo_escolhido == "Trabalho 1 — Risco e Retorno":
    subtitulo_modulo = "Análise de Risco e Retorno"
    rotulo_trabalho = "Trabalho 1 · CAD/UFMG"
else:
    subtitulo_modulo = "Análise de Projeto de Investimento"
    rotulo_trabalho = "Trabalho 2 · CAD/UFMG"

# =================================================================
# HEADER
# =================================================================
st.markdown(f"""
<div style='margin-bottom:2rem;'>
    <div style='display:flex;align-items:baseline;gap:1rem;'>
        <span style='font-family:Syne,sans-serif;font-size:2.2rem;font-weight:800;
            background:linear-gradient(135deg,{PALETTE["accent"]},{PALETTE["accent3"]});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            FinRisk
        </span>
        <span style='color:{PALETTE["muted"]};font-size:1rem;'>{subtitulo_modulo}</span>
    </div>
    <div style='display:flex;gap:2rem;margin-top:0.4rem;flex-wrap:wrap;'>
        <span style='font-size:0.75rem;color:{PALETTE["muted"]};'>
            <span style='color:{PALETTE["accent"]};font-weight:600;'>■</span> UFMG · Ciências Administrativas
        </span>
        <span style='font-size:0.75rem;color:{PALETTE["muted"]};'>
            <span style='color:{PALETTE["accent3"]};font-weight:600;'>■</span> Administração Financeira · Prof. Bruno Pérez Ferreira
        </span>
        <span style='font-size:0.75rem;color:{PALETTE["muted"]};'>
            <span style='color:{PALETTE["accent2"]};font-weight:600;'>■</span> {rotulo_trabalho}
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# =================================================================
# COLETA E PROCESSAMENTO
# =================================================================
if modulo_escolhido == "Trabalho 1 — Risco e Retorno":
    bench_map = {
        "IBOV (^BVSP)": "^BVSP",
        "S&P 500 (^GSPC)": "^GSPC",
        "Nasdaq (^IXIC)": "^IXIC",
    }

    if executar and tickers_selecionados:
        with st.spinner("Conectando ao Yahoo Finance..."):
            try:
                dados_brutos = yf.download(
                    tickers_selecionados,
                    start=data_inicio,
                    end=data_fim,
                    auto_adjust=True,
                    group_by="column",
                    progress=False,
                )
                dados = extrair_precos(dados_brutos, tickers=tickers_selecionados)

                if dados.empty:
                    st.error("Nenhum dado retornado. Verifique os ativos e o período.")
                    st.stop()

                dados = dados.dropna(how="all")
                if dados.shape[0] < 2:
                    st.error("Dados insuficientes no período escolhido.")
                    st.stop()

                retornos_diarios = dados.pct_change().dropna(how="any")

                benchmark_ticker = bench_map.get(benchmark_label)
                retornos_benchmark = None

                if benchmark_ticker:
                    dados_bm = yf.download(benchmark_ticker, start=data_inicio, end=data_fim, auto_adjust=True, progress=False)
                    precos_bm = extrair_precos(dados_bm, tickers=[benchmark_ticker])
                    if not precos_bm.empty:
                        rb = precos_bm.iloc[:, 0].pct_change().dropna()
                        alinhado = retornos_diarios.join(rb.rename("_BM"), how="inner").dropna()
                        if not alinhado.empty:
                            retornos_benchmark = alinhado["_BM"]
                            retornos_diarios = alinhado.drop(columns=["_BM"])

                if adicionar_carteira and retornos_diarios.shape[1] > 1:
                    retornos_diarios["⚖ Carteira Eq. Peso"] = retornos_diarios.mean(axis=1)

                rf_anual = taxa_livre_risco / 100
                metricas = calcular_metricas(retornos_diarios, rf_anual)

                df_resumo = pd.DataFrame({
                    "Retorno Total (%)": metricas["retorno_total"] * 100,
                    "CAGR (%)": metricas["cagr"] * 100,
                    "Retorno Anual (%)": metricas["retorno_anual"] * 100,
                    "Volatilidade (%)": metricas["risco_anual"] * 100,
                    "Sharpe": metricas["sharpe"],
                    "Sortino": metricas["sortino"],
                    "Calmar": metricas["calmar"],
                    "Max Drawdown (%)": metricas["max_drawdown"] * 100,
                    "VaR 95% (diário %)": metricas["var_95"] * 100,
                    "CVaR 95% (diário %)": metricas["cvar_95"] * 100,
                    "Assimetria": metricas["skew"],
                    "Curtose": metricas["kurt"],
                })

                if retornos_benchmark is not None and retornos_benchmark.var() != 0:
                    beta = retornos_diarios.apply(
                        lambda s: s.cov(retornos_benchmark) / retornos_benchmark.var()
                    )
                    alpha_jensen = (
                        metricas["retorno_anual"]
                        - rf_anual
                        - beta * (retornos_benchmark.mean() * 252 - rf_anual)
                    )
                    df_resumo["Beta"] = beta
                    df_resumo["Alpha de Jensen (%)"] = alpha_jensen * 100

                melhor_retorno = df_resumo["Retorno Anual (%)"].idxmax()
                melhor_sharpe = df_resumo["Sharpe"].idxmax()
                menor_drawdown = df_resumo["Max Drawdown (%)"].idxmax()
                menor_volatilidade = df_resumo["Volatilidade (%)"].idxmin()

                st.markdown(f"""
                <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};
                    border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.5rem;
                    display:flex;gap:2rem;flex-wrap:wrap;align-items:center;'>
                    <div>
                        <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Período</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["text"]};'>
                            {data_inicio.strftime("%d/%m/%Y")} → {data_fim.strftime("%d/%m/%Y")}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Dias úteis</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["text"]};'>
                            {len(retornos_diarios)}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Séries</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["text"]};'>
                            {retornos_diarios.shape[1]}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Taxa livre de risco</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["accent2"]};'>
                            {taxa_livre_risco:.2f}% a.a.
                        </div>
                    </div>
                    <div>
                        <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Benchmark</div>
                        <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["accent3"]};'>
                            {benchmark_label}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🏆 Maior Retorno Anual", melhor_retorno,
                          f"{df_resumo.loc[melhor_retorno, 'Retorno Anual (%)']:.2f}%")
                c2.metric("⚡ Maior Sharpe", melhor_sharpe,
                          f"{df_resumo.loc[melhor_sharpe, 'Sharpe']:.2f}")
                c3.metric("🛡 Menor Drawdown", menor_drawdown,
                          f"{df_resumo.loc[menor_drawdown, 'Max Drawdown (%)']:.2f}%")
                c4.metric("🎯 Menor Volatilidade", menor_volatilidade,
                          f"{df_resumo.loc[menor_volatilidade, 'Volatilidade (%)']:.2f}%")

                st.markdown("")

                tabs = st.tabs([
                    "📈 Retorno Acumulado",
                    "🎯 Risco × Retorno",
                    "📉 Drawdown",
                    "🔗 Correlação",
                    "📊 Distribuições",
                    "🧮 Estatísticas",
                    "📋 Relatório",
                ])

                with tabs[0]:
                    base100 = metricas["acumulado"] * 100
                    fig = go.Figure()
                    for i, col in enumerate(base100.columns):
                        cor = COLORS_SEQ[i % len(COLORS_SEQ)]
                        fig.add_trace(go.Scatter(
                            x=base100.index, y=base100[col],
                            name=col, line=dict(color=cor, width=2),
                            hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.1f}</b><extra>" + col + "</extra>",
                        ))
                    fig.add_hline(y=100, line_dash="dot", line_color=PALETTE["muted"], opacity=0.5)
                    fig.update_layout(**PLOTLY_LAYOUT, title="Retorno Acumulado (Base 100)", height=430)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("Base 100 no início do período. Retornos calculados com preços ajustados.")

                with tabs[1]:
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        fig2 = px.scatter(
                            df_resumo.reset_index().rename(columns={"index": "Ativo"}),
                            x="Volatilidade (%)", y="Retorno Anual (%)",
                            text="Ativo", size=df_resumo["Sharpe"].clip(lower=0.01).values,
                            color="Sharpe", color_continuous_scale=["#F87171", "#F59E0B", "#34D399"],
                            title="Risco × Retorno (tamanho = Sharpe)",
                            hover_data=["Sharpe", "Max Drawdown (%)"],
                        )
                        fig2.update_traces(textposition="top center", marker=dict(opacity=0.85, line=dict(width=1, color="#000")))
                        fig2.update_layout(**PLOTLY_LAYOUT, height=430,
                                           coloraxis_colorbar=dict(title="Sharpe", tickfont=dict(color=PALETTE["muted"])))
                        st.plotly_chart(fig2, use_container_width=True)
                    with col_b:
                        st.markdown(f"""
                        <div style='background:{PALETTE["panel2"]};border:1px solid {PALETTE["border"]};
                            border-radius:12px;padding:1.2rem;height:430px;overflow-y:auto;'>
                            <div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;
                                color:{PALETTE["muted"]};margin-bottom:1rem;'>Ranking por Sharpe</div>
                        """ + "".join([
                            f"""<div style='display:flex;justify-content:space-between;align-items:center;
                                padding:0.5rem 0;border-bottom:1px solid {PALETTE["border"]}44;'>
                                <span style='font-size:0.8rem;color:{PALETTE["text"]};'>{idx}</span>
                                <span style='font-family:JetBrains Mono,monospace;font-size:0.8rem;
                                    color:{PALETTE["accent"] if row["Sharpe"] > 0 else PALETTE["negative"]};'>
                                    {row["Sharpe"]:.2f}
                                </span></div>"""
                            for idx, row in df_resumo.sort_values("Sharpe", ascending=False).iterrows()
                        ]) + "</div>", unsafe_allow_html=True)

                with tabs[2]:
                    drawdown = metricas["acumulado"] / metricas["acumulado"].cummax() - 1
                    fig3 = go.Figure()
                    for i, col in enumerate(drawdown.columns):
                        cor = COLORS_SEQ[i % len(COLORS_SEQ)]
                        fig3.add_trace(go.Scatter(
                            x=drawdown.index, y=drawdown[col] * 100,
                            name=col, fill="tozeroy",
                            line=dict(color=cor, width=1.5),
                            fillcolor=f"rgba({int(cor[1:3],16)},{int(cor[3:5],16)},{int(cor[5:7],16)},0.13)" if cor.startswith("#") and len(cor)==7 else cor,
                            hovertemplate="%{x|%d/%m/%Y}<br><b>%{y:.2f}%</b><extra>" + col + "</extra>",
                        ))
                    fig3.update_layout(**PLOTLY_LAYOUT, title="Drawdown (%)", height=430,
                                       yaxis_ticksuffix="%")
                    st.plotly_chart(fig3, use_container_width=True)

                with tabs[3]:
                    corr = retornos_diarios.corr()
                    fig4 = px.imshow(
                        corr, text_auto=".2f",
                        color_continuous_scale=["#F87171", PALETTE["panel"], "#22D3EE"],
                        zmin=-1, zmax=1, aspect="auto",
                    )
                    fig4.update_layout(**PLOTLY_LAYOUT, title="Matriz de Correlação (Retornos Diários)", height=420,
                                       coloraxis_colorbar=dict(title="ρ", tickfont=dict(color=PALETTE["muted"])))
                    fig4.update_traces(textfont=dict(color="white", size=10))
                    st.plotly_chart(fig4, use_container_width=True)
                    st.caption("Correlação de Pearson entre os retornos diários dos ativos.")

                with tabs[4]:
                    ret_long = retornos_diarios.copy()
                    ret_long.index.name = "Data"
                    ret_long = ret_long.reset_index().melt(id_vars="Data", var_name="Ativo", value_name="Retorno")

                    col_h, col_b2 = st.columns(2)
                    with col_h:
                        fig5 = px.histogram(
                            ret_long, x="Retorno", color="Ativo",
                            nbins=80, opacity=0.7, barmode="overlay",
                            title="Histograma de Retornos Diários",
                            color_discrete_sequence=COLORS_SEQ,
                        )
                        fig5.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_tickformat=".1%")
                        st.plotly_chart(fig5, use_container_width=True)
                    with col_b2:
                        fig6 = px.box(
                            ret_long, x="Ativo", y="Retorno",
                            points="suspectedoutliers", color="Ativo",
                            title="Box Plot — Retornos Diários",
                            color_discrete_sequence=COLORS_SEQ,
                        )
                        fig6.update_layout(**PLOTLY_LAYOUT, height=380, yaxis_tickformat=".1%",
                                           showlegend=False, xaxis_tickangle=-30)
                        st.plotly_chart(fig6, use_container_width=True)

                with tabs[5]:
                    st.markdown(f"<div style='color:{PALETTE['muted']};font-size:0.8rem;margin-bottom:1rem;'>Indicadores de risco avançados e estatísticas de distribuição dos retornos.</div>", unsafe_allow_html=True)
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        fig_var = go.Figure()
                        fig_var.add_trace(go.Bar(
                            x=df_resumo.index, y=df_resumo["VaR 95% (diário %)"],
                            name="VaR 95%", marker_color=PALETTE["negative"], opacity=0.8,
                        ))
                        fig_var.add_trace(go.Bar(
                            x=df_resumo.index, y=df_resumo["CVaR 95% (diário %)"],
                            name="CVaR 95%", marker_color="#7C3AED", opacity=0.8,
                        ))
                        fig_var.update_layout(**PLOTLY_LAYOUT, title="VaR e CVaR (95%, diário)", height=320,
                                              barmode="group", yaxis_ticksuffix="%")
                        st.plotly_chart(fig_var, use_container_width=True)
                    with col_v2:
                        fig_sk = go.Figure()
                        fig_sk.add_trace(go.Bar(
                            x=df_resumo.index, y=df_resumo["Assimetria"],
                            marker_color=[PALETTE["positive"] if v >= 0 else PALETTE["negative"]
                                          for v in df_resumo["Assimetria"]],
                            name="Assimetria",
                        ))
                        fig_sk.update_layout(**PLOTLY_LAYOUT, title="Assimetria (Skewness)", height=320)
                        st.plotly_chart(fig_sk, use_container_width=True)

                    melhor = melhor_retorno
                    ret_mensal = retornos_diarios[melhor].resample("ME").apply(lambda x: (1 + x).prod() - 1)
                    ret_mensal_df = pd.DataFrame({
                        "Ano": ret_mensal.index.year,
                        "Mês": ret_mensal.index.month,
                        "Retorno": ret_mensal.values * 100,
                    })
                    pivot = ret_mensal_df.pivot(index="Ano", columns="Mês", values="Retorno")
                    pivot.columns = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
                    fig_heat = px.imshow(
                        pivot, text_auto=".1f",
                        color_continuous_scale=["#F87171", PALETTE["panel"], "#34D399"],
                        zmin=-20, zmax=20, aspect="auto",
                        title=f"Retorno Mensal (%) — {melhor}",
                    )
                    fig_heat.update_layout(**PLOTLY_LAYOUT, height=280,
                                            coloraxis_colorbar=dict(title="%", tickfont=dict(color=PALETTE["muted"])))
                    fig_heat.update_traces(textfont=dict(color="white", size=9))
                    st.plotly_chart(fig_heat, use_container_width=True)

                with tabs[6]:
                    fmt = {c: "{:.2f}" for c in df_resumo.columns}
                    st.dataframe(
                        df_resumo.style
                        .format(fmt)
                        .background_gradient(subset=["Sharpe"], cmap="YlGn")
                        .background_gradient(subset=["Max Drawdown (%)"], cmap="RdYlGn_r"),
                        use_container_width=True,
                    )
                    csv = (
                        df_resumo.reset_index()
                        .rename(columns={"index": "Ativo"})
                        .to_csv(index=False)
                        .encode("utf-8")
                    )
                    st.download_button("⬇ Exportar CSV", data=csv, file_name="finrisk_relatorio.csv", mime="text/csv")

                with st.expander("🎓 Fundamentos Teóricos"):
                    st.markdown(f"""
                    <div style='font-size:0.85rem;line-height:1.8;color:{PALETTE["muted"]};'>

                    | Métrica | Definição |
                    |---|---|
                    | **Retorno Total / CAGR** | Crescimento acumulado e taxa de crescimento anual composta |
                    | **Volatilidade** | Desvio padrão dos retornos diários × √252 |
                    | **Índice de Sharpe** | (Retorno – Rf) / Volatilidade total |
                    | **Índice de Sortino** | (Retorno – Rf) / Volatilidade negativa |
                    | **Índice de Calmar** | CAGR / |Max Drawdown| |
                    | **Max Drawdown** | Maior queda do pico ao vale no período |
                    | **VaR 95%** | Perda máxima esperada em 95% dos dias |
                    | **CVaR 95%** | Perda média esperada nos piores 5% dos dias |
                    | **Beta** | Sensibilidade do ativo em relação ao benchmark |
                    | **Alpha de Jensen** | Retorno em excesso ajustado ao risco sistemático |
                    | **Assimetria / Curtose** | Forma da distribuição dos retornos |

                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Erro no processamento: {e}")

    elif executar and not tickers_selecionados:
        st.warning("Selecione ao menos um ativo na barra lateral.")

    else:
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;'>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>📈</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Retorno & Risco</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Sharpe, Sortino, Calmar, VaR, CVaR e muito mais</div>
            </div>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>🌎</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Ativos Globais</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Ações B3, ETFs, BDRs, ações americanas e criptomoedas</div>
            </div>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>🧮</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Análise Avançada</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Alpha de Jensen, heatmap mensal, matriz de correlação</div>
            </div>
        </div>
        <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.5rem;'>
            <div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:{PALETTE["muted"]};margin-bottom:1rem;'>Como usar</div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;'>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>01</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Selecione os ativos na barra lateral</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>02</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Defina o período e os parâmetros</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>03</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Clique em "Gerar Análise"</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>04</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Explore os gráficos e exporte o relatório</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    if executar:
        tabela_projeto = montar_tabela_projeto(investimento_inicial, fluxos_caixa_projeto, taxa_desconto)
        vpl = calcular_vpl(investimento_inicial, fluxos_caixa_projeto, taxa_desconto)
        tir = calcular_tir(investimento_inicial, fluxos_caixa_projeto)
        payback_simples = calcular_payback_simples(investimento_inicial, fluxos_caixa_projeto)
        payback_descontado = calcular_payback_descontado(investimento_inicial, fluxos_caixa_projeto, taxa_desconto)

        taxas_sensibilidade = np.arange(0, 31, 2)
        df_sensibilidade = pd.DataFrame({
            "Taxa (%)": taxas_sensibilidade,
            "VPL (R$)": [calcular_vpl(investimento_inicial, fluxos_caixa_projeto, taxa) for taxa in taxas_sensibilidade],
        })

        st.markdown(f"""
        <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};
            border-radius:14px;padding:1rem 1.5rem;margin-bottom:1.5rem;
            display:flex;gap:2rem;flex-wrap:wrap;align-items:center;'>
            <div>
                <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Projeto</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["text"]};'>
                    {nome_projeto}
                </div>
            </div>
            <div>
                <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Investimento Inicial</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["accent2"]};'>
                    R$ {investimento_inicial:,.2f}
                </div>
            </div>
            <div>
                <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>TMA</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["accent3"]};'>
                    {taxa_desconto:.2f}% a.a.
                </div>
            </div>
            <div>
                <div style='font-size:0.68rem;color:{PALETTE["muted"]};text-transform:uppercase;letter-spacing:0.1em;'>Horizonte</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:0.9rem;color:{PALETTE["text"]};'>
                    {quantidade_periodos} anos
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💵 VPL", f"R$ {vpl:,.2f}")
        c2.metric("📈 TIR", "N/D" if pd.isna(tir) else f"{tir * 100:.2f}%")
        c3.metric("⏱ Payback Simples", "N/D" if pd.isna(payback_simples) else f"{payback_simples:.2f} anos")
        c4.metric("🧮 Payback Descontado", "N/D" if pd.isna(payback_descontado) else f"{payback_descontado:.2f} anos")

        tabs = st.tabs(["💸 Fluxo de Caixa", "📊 Sensibilidade", "📋 Relatório"])

        with tabs[0]:
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                fig_fluxo = go.Figure()
                fig_fluxo.add_trace(go.Bar(
                    x=tabela_projeto["Período"],
                    y=tabela_projeto["Fluxo de Caixa (R$)"],
                    marker_color=[PALETTE["negative"]] + [PALETTE["positive"]] * (len(tabela_projeto) - 1),
                    name="Fluxo de Caixa",
                ))
                fig_fluxo.update_layout(**PLOTLY_LAYOUT, title="Fluxo de Caixa do Projeto", height=360)
                st.plotly_chart(fig_fluxo, use_container_width=True)

            with col_g2:
                fig_acumulado = go.Figure()
                fig_acumulado.add_trace(go.Scatter(
                    x=tabela_projeto["Período"],
                    y=tabela_projeto["Acumulado Descontado (R$)"],
                    mode="lines+markers",
                    line=dict(color=PALETTE["accent"], width=3),
                    name="Acumulado Descontado",
                ))
                fig_acumulado.add_hline(y=0, line_dash="dot", line_color=PALETTE["muted"])
                fig_acumulado.update_layout(**PLOTLY_LAYOUT, title="Evolução do Fluxo Acumulado Descontado", height=360)
                st.plotly_chart(fig_acumulado, use_container_width=True)

            st.dataframe(
                tabela_projeto.style.format({
                    "Fluxo de Caixa (R$)": "R$ {:,.2f}",
                    "Fluxo Descontado (R$)": "R$ {:,.2f}",
                    "Acumulado (R$)": "R$ {:,.2f}",
                    "Acumulado Descontado (R$)": "R$ {:,.2f}",
                }),
                use_container_width=True,
            )

        with tabs[1]:
            fig_sensibilidade = px.line(
                df_sensibilidade,
                x="Taxa (%)",
                y="VPL (R$)",
                markers=True,
                title="Sensibilidade do VPL em relação à TMA",
            )
            fig_sensibilidade.update_traces(line=dict(color=PALETTE["accent3"], width=3))
            fig_sensibilidade.update_layout(**PLOTLY_LAYOUT, height=360)
            st.plotly_chart(fig_sensibilidade, use_container_width=True)

            st.caption("Se o VPL permanecer positivo em taxas maiores, o projeto tende a ser mais robusto.")

        with tabs[2]:
            resumo_projeto = pd.DataFrame({
                "Indicador": ["Projeto", "Investimento Inicial", "TMA (%)", "VPL (R$)", "TIR (%)", "Payback Simples (anos)", "Payback Descontado (anos)"],
                "Valor": [
                    nome_projeto,
                    investimento_inicial,
                    taxa_desconto,
                    vpl,
                    np.nan if pd.isna(tir) else tir * 100,
                    payback_simples,
                    payback_descontado,
                ],
            })
            st.dataframe(resumo_projeto, use_container_width=True)

            csv_projeto = tabela_projeto.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Exportar Relatório do Projeto",
                data=csv_projeto,
                file_name="projeto_investimento_relatorio.csv",
                mime="text/csv",
            )

        with st.expander("🎓 Fundamentos Teóricos"):
            st.markdown(f"""
            <div style='font-size:0.85rem;line-height:1.8;color:{PALETTE["muted"]};'>

            | Indicador | Definição |
            |---|---|
            | **VPL** | Soma dos fluxos de caixa descontados menos o investimento inicial |
            | **TIR** | Taxa que zera o VPL do projeto |
            | **Payback Simples** | Tempo necessário para recuperar o investimento sem descontar a taxa |
            | **Payback Descontado** | Tempo necessário para recuperar o investimento considerando a TMA |
            | **Sensibilidade do VPL** | Mostra como o projeto reage a mudanças na taxa de desconto |

            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-bottom:2rem;'>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>💵</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Análise de Projeto</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Avalie investimentos com VPL, TIR e payback.</div>
            </div>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>🧾</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Captura de Dados</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Informe investimento inicial, TMA e fluxos de caixa anuais.</div>
            </div>
            <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.4rem;'>
                <div style='font-size:1.5rem;margin-bottom:0.5rem;'>📋</div>
                <div style='font-weight:600;color:{PALETTE["text"]};margin-bottom:0.3rem;'>Relatório</div>
                <div style='font-size:0.8rem;color:{PALETTE["muted"]};'>Gere tabela, gráficos e exporte o resultado em CSV.</div>
            </div>
        </div>
        <div style='background:{PALETTE["panel"]};border:1px solid {PALETTE["border"]};border-radius:14px;padding:1.5rem;'>
            <div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:{PALETTE["muted"]};margin-bottom:1rem;'>Como usar</div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;'>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>01</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Escolha o módulo Trabalho 2</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>02</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Preencha investimento inicial e TMA</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>03</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Informe os fluxos anuais do projeto</div>
                </div>
                <div style='text-align:center;'>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1.2rem;color:{PALETTE["accent"]};font-weight:700;'>04</div>
                    <div style='font-size:0.8rem;color:{PALETTE["muted"]};margin-top:0.3rem;'>Clique em "Gerar Análise" para emitir o relatório</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div style='margin-top:3rem;border-top:1px solid {PALETTE["border"]};padding-top:1rem;
    display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;'>
    <span style='font-size:0.7rem;color:{PALETTE["muted"]};'>
        FinRisk · UFMG · CAD · Administração Financeira · Prof. Bruno Pérez Ferreira
    </span>
    <span style='font-size:0.7rem;color:{PALETTE["muted"]};font-family:JetBrains Mono,monospace;'>
        Trabalho 1 + Trabalho 2 · Dados de mercado e análise de projetos
    </span>
</div>
""", unsafe_allow_html=True)