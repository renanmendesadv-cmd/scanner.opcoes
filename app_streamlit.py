#!/usr/bin/env python3
"""
app_streamlit.py — Scanner de Opções SuperTrend
Dashboard web acessível de qualquer dispositivo.

Rodar:
    streamlit run app_streamlit.py
"""

import sys
import re
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from scanner_opcoes_supertrend import (
    ATIVOS_PADRAO, ATR_PERIODO, ATR_MULT_PADRAO,
    VOL_THRESHOLD, DU_MIN, DU_MAX, REGRA_1PCT,
    analisar, _aplicar_iv_real, _enriquecer_legs, _montar_checklist,
)

try:
    import iv_b3
    IV_DISPONIVEL = True
except ImportError:
    IV_DISPONIVEL = False

import carteira_sinais

PASTA = Path(__file__).parent

# ══════════════════════════════════════════════════════════════════════
# CONFIG DA PÁGINA
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Scanner Opções — SuperTrend",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# CSS GLOBAL
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
  /* Cards */
  .card {
    background: #1e293b;
    border-radius: 12px;
    padding: 18px 20px;
    margin-bottom: 16px;
    border-left: 5px solid #334155;
  }
  .card-alta   { border-left-color: #22c55e; }
  .card-baixa  { border-left-color: #ef4444; }
  .card-lateral{ border-left-color: #f59e0b; }
  .card-erro   { border-left-color: #64748b; opacity: 0.6; }

  /* Badges */
  .badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-right: 4px;
  }
  .b-alta    { background:#14532d; color:#4ade80; }
  .b-baixa   { background:#450a0a; color:#f87171; }
  .b-lateral { background:#451a03; color:#fbbf24; }
  .b-vol-alta { background:#450a0a; color:#fca5a5; }
  .b-vol-baixa{ background:#0c2340; color:#93c5fd; }

  .nivel {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 12px;
    font-size: 0.70rem;
    font-weight: 600;
  }
  .n-ini { background:#14532d; color:#4ade80; }
  .n-int { background:#1e3a5f; color:#60a5fa; }
  .n-ava { background:#4a044e; color:#e879f9; }

  /* Estratégia bloco */
  .strat-bloco {
    background: #0f172a;
    border-radius: 8px;
    padding: 12px 14px;
    margin: 8px 0;
  }
  .strat-nome {
    font-size: 0.90rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 8px;
  }

  /* Preços da operação */
  .po-grid {
    display: flex;
    gap: 8px;
    margin: 8px 0;
    flex-wrap: wrap;
  }
  .po-box {
    flex: 1;
    min-width: 70px;
    background: #1e293b;
    border-radius: 6px;
    padding: 6px 10px;
    text-align: center;
    font-size: 0.75rem;
    color: #94a3b8;
  }
  .po-box b { display: block; font-size: 0.95rem; margin-top: 2px; }
  .po-entrada b { color: #60a5fa; }
  .po-stop    b { color: #f87171; }
  .po-alvo    b { color: #4ade80; }
  .po-rr      b { color: #fbbf24; }

  .disclaimer {
    background: #1c1510;
    color: #fb923c;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 0.78rem;
    margin-bottom: 16px;
  }
  .iv-real { color: #34d399; font-weight: 600; }
  .iv-hist { color: #f59e0b; font-weight: 600; }

  /* Carteira — status badges */
  .st-aberto  { color: #fbbf24; font-weight: 700; }
  .st-alvo    { color: #4ade80; font-weight: 700; }
  .st-stop    { color: #f87171; font-weight: 700; }
  .st-exp     { color: #94a3b8; font-weight: 700; }

  /* Carteira — célula P&L */
  .pnl-pos { color: #4ade80; font-weight: 700; }
  .pnl-neg { color: #f87171; font-weight: 700; }
  .pnl-zer { color: #94a3b8; }

  /* Ocultar menu e footer do Streamlit */
  #MainMenu { visibility: hidden; }
  footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def badge_dir(d):
    if d == "ALTA":
        return '<span class="badge b-alta">▲ ALTA</span>'
    if d == "BAIXA":
        return '<span class="badge b-baixa">▼ BAIXA</span>'
    return '<span class="badge b-lateral">↔ LATERAL</span>'


def badge_vol(v):
    if v == "alta":
        return '<span class="badge b-vol-alta">Vol ALTA</span>'
    return '<span class="badge b-vol-baixa">Vol BAIXA</span>'


def badge_nivel(n):
    if "Iniciante" in n:
        return f'<span class="nivel n-ini">Iniciante</span>'
    if "Intermediário" in n:
        return f'<span class="nivel n-int">Intermediário</span>'
    return f'<span class="nivel n-ava">Avançado</span>'


def po_html(po):
    if not po or not po.get("entrada"):
        return ""
    entrada = po.get("entrada")
    stop    = po.get("stop")
    alvo    = po.get("alvo")
    rr      = po.get("rr", "—")
    desc    = po.get("descricao", "")

    stop_txt = f"R$ {stop:.2f}" if stop is not None else "—"
    alvo_txt = f"R$ {alvo:.2f}" if alvo is not None else "—"

    return f"""
    <div class="po-grid">
      <div class="po-box po-entrada">Entrada<br><b>R$ {entrada:.2f}</b></div>
      <div class="po-box po-stop">Stop<br><b>{stop_txt}</b></div>
      <div class="po-box po-alvo">Alvo<br><b>{alvo_txt}</b></div>
      <div class="po-box po-rr">R/R<br><b>{rr}</b></div>
    </div>
    <div style="font-size:0.75rem;color:#64748b;margin-top:2px;">{desc}</div>
    """


def renderizar_legs_df(legs):
    if not legs:
        return
    rows = []
    for lg in legs:
        rows.append({
            "Ação":       lg["acao"],
            "Ticker":     lg["codneg"],
            "Tipo":       lg["tipo"],
            "Strike":     f"R$ {lg['strike']:.2f}",
            "Prêmio":     f"R$ {lg['preult']:.2f}",
            "Vencimento": lg["vencimento"].strftime("%d/%m/%Y"),
            "du":         lg["du"],
        })
    df = pd.DataFrame(rows)

    def colorir_acao(val):
        if val == "COMPRAR":
            return "background-color:#14532d; color:#4ade80; font-weight:700"
        return "background-color:#450a0a; color:#f87171; font-weight:700"

    st.dataframe(
        df.style.map(colorir_acao, subset=["Ação"]),
        use_container_width=True,
        hide_index=True,
        height=min(70 + len(rows) * 38, 200),
    )


def renderizar_card(r):
    if not r.get("ok"):
        st.markdown(
            f'<div class="card card-erro">❌ <b>{r["ticker"]}</b> — {r.get("erro","sem dados")}</div>',
            unsafe_allow_html=True,
        )
        return

    checks  = _montar_checklist(r)
    n_ok    = sum(checks.values())
    dir_cls = r["direcao_str"].lower()
    venc_str = r["vencimento"].strftime("%d/%m/%Y") if r["vencimento"] else "N/A"

    if r["iv_real"]:
        iv_html = f'<span class="iv-real">IV real B3: {r["iv_real"]*100:.1f}%</span>'
    else:
        iv_html = f'<span class="iv-hist">Vol hist: {r["vol_hist"]*100:.1f}% ⚠</span>'

    st.markdown(f"""
    <div class="card card-{dir_cls}">
      <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px;">
        <span style="font-size:1.3rem;font-weight:700;color:#f1f5f9;">{r['ticker']}</span>
        {badge_dir(r['direcao_str'])}
        {badge_vol(r['vol_str'])}
        <span style="margin-left:auto;background:#334155;padding:2px 10px;border-radius:20px;font-size:0.78rem;color:#94a3b8;">{n_ok}/{len(checks)} ✓</span>
      </div>
      <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:4px;">
        Preço: <b style="color:#e2e8f0;">R$ {r['preco']:.2f}</b> &nbsp;|&nbsp;
        SuperTrend: <b style="color:#e2e8f0;">R$ {r['st_valor']:.2f}</b> &nbsp;|&nbsp;
        Dist: <b style="color:#64748b;">{r['dist_pct']:+.1f}%</b>
      </div>
      <div style="font-size:0.80rem;color:#64748b;">
        {iv_html} &nbsp;|&nbsp; Vol hist: {r['vol_hist']*100:.1f}% &nbsp;|&nbsp;
        Venc: <b style="color:#e2e8f0;">{venc_str}</b> ({r['du']} du{'  ⚠' if not r['du_ok'] else ''})
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Estratégias com legs
    for det in r.get("estrategias_detalhadas", []):
        nivel = det["nivel"]
        if "Iniciante" in nivel:
            icone = "🟢"
        elif "Intermediário" in nivel:
            icone = "🔵"
        else:
            icone = "🟣"
        with st.expander(f"{det['nome']}  {icone} {nivel}", expanded=True):
            if det["legs"]:
                renderizar_legs_df(det["legs"])
                st.markdown(po_html(det.get("preco_op", {})), unsafe_allow_html=True)
            else:
                po = det.get("preco_op", {})
                msg = po.get("descricao", "Opções não encontradas no COTAHIST para este vencimento.")
                st.caption(f"⚠ {msg}")

    # Checklist
    with st.expander("Checklist"):
        for item, ok in checks.items():
            st.markdown(
                f"{'✅' if ok else '⚠️'} {item}",
            )


# ══════════════════════════════════════════════════════════════════════
# CACHE — COTAHIST (pesado, cachear por 1h)
# ══════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def carregar_cotahist(tickers_str, precos_str):
    """Baixa e parseia o COTAHIST. Cache de 1 hora."""
    tickers = tickers_str.split(",")
    precos  = {k: float(v) for k, v in (p.split(":") for p in precos_str.split(","))}
    return iv_b3.obter_iv_todos(
        tickers, precos, verbose=False, retornar_opcoes=True,
    )


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📊 Scanner de Opções")
    st.markdown("**SuperTrend v1** — IV real B3")
    st.divider()

    patrimonio = st.number_input(
        "Patrimônio (R$)", min_value=1000.0, max_value=10_000_000.0,
        value=10_000.0, step=1000.0, format="%.0f",
    )

    mult = st.slider(
        "Multiplicador ATR", min_value=1.5, max_value=5.0,
        value=float(ATR_MULT_PADRAO), step=0.5,
    )

    usar_iv = st.checkbox("Usar IV real (COTAHIST B3)", value=IV_DISPONIVEL and True, disabled=not IV_DISPONIVEL)

    todos_ativos = sorted(set(
        ATIVOS_PADRAO + [
            "ITUB4.SA","BBDC4.SA","BBAS3.SA","ABEV3.SA",
            "WEGE3.SA","RENT3.SA","LREN3.SA","HAPV3.SA",
            "SUZB3.SA","JBSS3.SA","BEEF3.SA",
        ]
    ))

    ativos_sel = st.multiselect(
        "Ativos",
        options=todos_ativos,
        default=ATIVOS_PADRAO,
        placeholder="Selecione os ativos...",
    )

    if not ativos_sel:
        ativos_sel = ATIVOS_PADRAO

    st.divider()
    rodar = st.button("▶ Rodar Scanner", type="primary", use_container_width=True)

    st.divider()
    st.caption("⚠️ Material educacional.\nNão é recomendação de investimento.")

    # Histórico de relatórios
    relatorios = sorted(PASTA.glob("scanner_*.xlsx"), reverse=True)[:5]
    if relatorios:
        st.markdown("### 📁 Últimos relatórios")
        for f in relatorios:
            ts = f.stem.replace("scanner_", "")
            try:
                dt = datetime.strptime(ts, "%Y%m%d_%H%M")
                label = dt.strftime("%d/%m %H:%M")
            except Exception:
                label = ts
            with open(f, "rb") as fh:
                st.download_button(
                    label=f"⬇ {label}",
                    data=fh,
                    file_name=f.name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL
# ══════════════════════════════════════════════════════════════════════

st.markdown("# 📊 Scanner de Opções — SuperTrend v1")
st.markdown(
    '<div class="disclaimer">⚠️ Material exclusivamente educacional. '
    'Não constitui recomendação de investimento. '
    'Operações com opções envolvem risco de perda total do capital investido.</div>',
    unsafe_allow_html=True,
)

# ── Execução ─────────────────────────────────────────────────────────
if rodar:
    resultados_base = {}

    prog = st.progress(0, text="Baixando cotações históricas...")
    n = len(ativos_sel)

    for i, ticker in enumerate(ativos_sel):
        prog.progress((i + 1) / (n * 2), text=f"Analisando {ticker.replace('.SA','')}...")
        resultados_base[ticker] = analisar(ticker, mult=mult)

    iv_reais  = {}
    opcoes_df = None
    cdi       = iv_b3.CDI_PADRAO if IV_DISPONIVEL else 0.1325

    if usar_iv and IV_DISPONIVEL:
        prog.progress(0.6, text="Baixando COTAHIST B3 (pode levar ~30s na 1ª vez)...")
        precos_ok = {
            t.replace(".SA", ""): r["preco"]
            for t, r in resultados_base.items() if r.get("ok")
        }
        if precos_ok:
            try:
                tickers_str = ",".join(precos_ok.keys())
                precos_str  = ",".join(f"{k}:{v}" for k, v in precos_ok.items())
                iv_reais, opcoes_df, cdi = carregar_cotahist(tickers_str, precos_str)
            except Exception as e:
                st.warning(f"COTAHIST indisponível: {e}. Usando vol. histórica.")

    prog.progress(0.9, text="Consolidando resultados...")
    resultados = []
    for ticker in ativos_sel:
        base = ticker.replace(".SA", "")
        r = resultados_base[ticker]
        r = _aplicar_iv_real(r, iv_reais.get(base))
        r = _enriquecer_legs(r, opcoes_df, cdi)
        resultados.append(r)

    prog.progress(1.0, text="Concluído!")
    prog.empty()

    st.session_state["resultados"] = resultados
    st.session_state["patrimonio"] = patrimonio
    st.session_state["timestamp"]  = datetime.now().strftime("%d/%m/%Y %H:%M")
    st.session_state["opcoes_df"]  = opcoes_df

    # Registrar novos sinais e atualizar preços na carteira
    novos = carteira_sinais.registrar_sinais(resultados)
    _, atualizados = carteira_sinais.atualizar_precos(opcoes_df)
    if novos > 0:
        st.toast(f"📋 {novos} novo(s) sinal(is) registrado(s) na Carteira!", icon="✅")
    if atualizados > 0:
        st.toast(f"🔄 {atualizados} posição(ões) atualizada(s) na Carteira.", icon="📊")

# ── Tabs principais ───────────────────────────────────────────────────
tab_scanner, tab_carteira = st.tabs(["🔍 Scanner", "📈 Carteira de Sinais"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — SCANNER
# ══════════════════════════════════════════════════════════════════════
with tab_scanner:
    if not rodar and "resultados" not in st.session_state:
        st.info("Configure os parâmetros na barra lateral e clique em **▶ Rodar Scanner**.")
        st.stop()

    resultados  = st.session_state.get("resultados", [])
    pat_usado   = st.session_state.get("patrimonio", patrimonio)
    ts_execucao = st.session_state.get("timestamp", "—")

    sinais = [r for r in resultados if r.get("ok")]
    erros  = [r for r in resultados if not r.get("ok")]

    # Métricas resumo
    risco  = pat_usado * REGRA_1PCT
    ex_min = pat_usado * 0.05
    ex_max = pat_usado * 0.20

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ativos analisados", len(sinais))
    col2.metric("Com IV real", sum(1 for r in sinais if r.get("iv_real")))
    col3.metric("Risco máx/op (1%)", f"R$ {risco:,.0f}")
    col4.metric("Exposição mín (5%)", f"R$ {ex_min:,.0f}")
    col5.metric("Exposição máx (20%)", f"R$ {ex_max:,.0f}")

    st.caption(f"Última execução: {ts_execucao}  |  ATR {ATR_PERIODO} | Mult. {mult} | Vol threshold: {VOL_THRESHOLD*100:.0f}%")
    st.divider()

    # Filtros rápidos
    fcol1, fcol2 = st.columns([3, 1])
    with fcol1:
        filtro_dir = st.multiselect(
            "Filtrar por direção",
            ["ALTA", "BAIXA", "LATERAL"],
            default=["ALTA", "BAIXA", "LATERAL"],
        )
    with fcol2:
        apenas_iv = st.checkbox("Só com IV real", value=False)

    sinais_filtrados = [
        r for r in sinais
        if r["direcao_str"] in filtro_dir
        and (not apenas_iv or r.get("iv_real"))
    ]

    st.markdown(f"**{len(sinais_filtrados)} ativo(s) exibido(s)**")

    # Cards em 2 colunas
    if sinais_filtrados:
        col_a, col_b = st.columns(2)
        for i, r in enumerate(sinais_filtrados):
            with (col_a if i % 2 == 0 else col_b):
                renderizar_card(r)

    if erros:
        with st.expander(f"❌ Sem dados — {len(erros)} ativo(s)"):
            for r in erros:
                st.write(f"**{r['ticker']}** — {r.get('erro','erro desconhecido')}")

    # Mapa de decisão
    with st.expander("📋 Mapa de decisão rápido"):
        st.markdown("""
        | Cenário | Estratégias |
        |---|---|
        | Acima do ST + Vol BAIXA | 1 – Compra Call, 3 – Trava Alta Call, 16 – Lanç. Coberto |
        | Acima do ST + Vol ALTA  | 2 – Venda Put, 4 – Trava Alta Put |
        | Abaixo do ST + Vol BAIXA| 5 – Compra Put, 7 – Trava Baixa Put |
        | Abaixo do ST + Vol ALTA | 6 – Venda Call, 8 – Trava Baixa Call |
        | Lateral (ST serrando)   | 9 – Trava de Linha, 10 – Straddle Vend., 11 – Straddle Sint. |
        | Evento / Explosão       | 12 – Straddle Comp., 13 – Straddle Sint. |
        """)
        st.markdown("""
        **Regras gerais:**
        - Opere sempre no **fechamento do candle diário**
        - **1% do patrimônio** por operação
        - **5–20% de exposição** total em opções
        - Vencimento ideal: **20–40 dias úteis**
        """)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — CARTEIRA DE SINAIS
# ══════════════════════════════════════════════════════════════════════
with tab_carteira:

    df_cart = carteira_sinais.carregar()

    # Botão de atualização manual
    ccol1, ccol2 = st.columns([4, 1])
    with ccol2:
        if st.button("🔄 Atualizar preços", use_container_width=True):
            opcoes_df_cached = st.session_state.get("opcoes_df")
            if opcoes_df_cached is not None:
                df_cart, n_upd = carteira_sinais.atualizar_precos(opcoes_df_cached)
                st.toast(f"✅ {n_upd} posição(ões) atualizada(s).")
            else:
                st.warning("Rode o Scanner primeiro para carregar os preços do COTAHIST.")

    if df_cart.empty:
        st.info("Nenhum sinal registrado ainda. Rode o Scanner para gerar os primeiros sinais.")
        st.stop()

    # ── Métricas ──────────────────────────────────────────────────────
    met = carteira_sinais.calcular_metricas(df_cart)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Total de sinais",   met["total"])
    mc2.metric("Em aberto",         met["abertos"])
    mc3.metric("Encerrados",        met["fechados"])
    mc4.metric("Taxa de acerto",    f"{met['taxa_acerto']:.0f}%" if met["fechados"] > 0 else "—")
    pnl_color = "normal"
    mc5.metric(
        "P&L total (R$)",
        f"R$ {met['pnl_total']:+.2f}",
        delta=f"{met['pnl_total']:+.2f}",
    )

    st.divider()

    # ── Helpers de exibição ───────────────────────────────────────────
    def _fmt_pnl_rs(v):
        try:
            f = float(v)
            return f"R$ {f:+.2f}"
        except Exception:
            return "—"

    def _fmt_pnl_pct(v):
        try:
            f = float(v)
            return f"{f:+.1f}%"
        except Exception:
            return "—"

    def _fmt_preco(v):
        try:
            return f"R$ {float(v):.2f}"
        except Exception:
            return "—"

    def _status_icon(s):
        if s == "Aberto":
            return "🟡 Aberto"
        if "Alvo" in s:
            return "✅ Alvo"
        if "Stop" in s:
            return "🔴 Stop"
        return "⬜ " + s

    def _montar_df_exibicao(df_in):
        rows = []
        for _, row in df_in.iterrows():
            rows.append({
                "Data":        row["data_sinal"],
                "Ativo":       row["ativo"],
                "Estratégia":  row["estrategia"],
                "Leg 1":       f"{row['leg1_acao']} {row['leg1_ticker']}",
                "Leg 2":       f"{row['leg2_acao']} {row['leg2_ticker']}" if str(row.get("leg2_ticker","")).strip() else "—",
                "Entrada":     _fmt_preco(row["preco_entrada"]),
                "Atual":       _fmt_preco(row["preco_atual"]),
                "Stop":        _fmt_preco(row["stop"]),
                "Alvo":        _fmt_preco(row["alvo"]),
                "P&L R$":      _fmt_pnl_rs(row["pnl_rs"]),
                "P&L %":       _fmt_pnl_pct(row["pnl_pct"]),
                "Status":      _status_icon(str(row["status"])),
                "Atualizado":  row["data_atualizacao"],
            })
        return pd.DataFrame(rows)

    # ── Sinais abertos ────────────────────────────────────────────────
    abertos = df_cart[df_cart["status"] == "Aberto"].copy()
    st.markdown(f"### 🟡 Posições abertas ({len(abertos)})")

    if abertos.empty:
        st.caption("Nenhuma posição aberta no momento.")
    else:
        df_ab_exib = _montar_df_exibicao(abertos)

        def _colorir_pnl(val):
            try:
                v = float(val.replace("R$","").replace("%","").replace("+","").strip())
                if v > 0:
                    return "color: #4ade80; font-weight: 700"
                if v < 0:
                    return "color: #f87171; font-weight: 700"
            except Exception:
                pass
            return "color: #94a3b8"

        st.dataframe(
            df_ab_exib.style
                .map(_colorir_pnl, subset=["P&L R$", "P&L %"]),
            use_container_width=True,
            hide_index=True,
            height=min(100 + len(abertos) * 40, 500),
        )

    st.divider()

    # ── Histórico (fechados) ──────────────────────────────────────────
    fechados = df_cart[df_cart["status"] != "Aberto"].copy()
    st.markdown(f"### 📜 Histórico — posições encerradas ({len(fechados)})")

    if fechados.empty:
        st.caption("Nenhuma posição encerrada ainda.")
    else:
        # Métricas do histórico
        try:
            pnls_hist = pd.to_numeric(fechados["pnl_rs"], errors="coerce").fillna(0)
            ganhos_hist  = fechados[pnls_hist > 0]
            perdas_hist  = fechados[pnls_hist < 0]
            hcol1, hcol2, hcol3, hcol4 = st.columns(4)
            hcol1.metric("Ganhos",  len(ganhos_hist))
            hcol2.metric("Perdas",  len(perdas_hist))
            hcol3.metric("P&L total (encerrados)", f"R$ {pnls_hist.sum():+.2f}")
            media = pnls_hist.mean()
            hcol4.metric("Média por trade", f"R$ {media:+.2f}")
        except Exception:
            pass

        df_fech_exib = _montar_df_exibicao(fechados.sort_values("data_fechamento", ascending=False))
        st.dataframe(
            df_fech_exib.style
                .map(_colorir_pnl, subset=["P&L R$", "P&L %"]),
            use_container_width=True,
            hide_index=True,
            height=min(100 + len(fechados) * 40, 500),
        )

    st.divider()

    # ── Exportar CSV ──────────────────────────────────────────────────
    csv_bytes = df_cart.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇ Exportar carteira (.csv)",
        data=csv_bytes,
        file_name=f"carteira_sinais_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.caption("Os preços são atualizados automaticamente a cada execução do Scanner usando o COTAHIST B3 do dia.")
