#!/usr/bin/env python3
"""
iv_b3.py
────────────────────────────────────────────────────────────────────────
Módulo de volatilidade implícita real — sem assinatura, sem custos.

Fluxo:
  1. Baixa o COTAHIST diário da B3 (arquivo público, gratuito).
  2. Parseia opções de ações do mercado à vista (TPMERC = 070/080).
  3. Para cada ativo, seleciona séries ATM com liquidez mínima.
  4. Calcula IV via Black-Scholes + bisseção para cada série selecionada.
  5. Retorna IV média ponderada pelo volume financeiro do dia.

Usado pelo scanner_opcoes_supertrend.py — pode ser importado
independentemente.

Dependências: requests, numpy, pandas (já instaladas pelo scanner)
────────────────────────────────────────────────────────────────────────
"""

import io
import math
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ══════════════════════════════════════════════════════════════════════
# CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════

CACHE_DIR   = Path("cotahist_cache")   # pasta local de cache dos ZIPs
ATM_BANDA   = 0.08    # ±8% do spot = opções consideradas "ATM"
MIN_VOL_NEG = 20      # mínimo de contratos negociados no dia
MIN_DU      = 5       # ignora opções com < 5 du (theta explosivo)
MAX_DU      = 60      # ignora opções com > 60 du (premios muito pequenos)
CDI_PADRAO  = 0.1325  # 13,25% a.a. — fallback se API do BACEN falhar

# Letras de mês: A-L = calls; M-X = puts (padrão B3)
CALL_LETTERS = frozenset("ABCDEFGHIJKL")
PUT_LETTERS  = frozenset("MNOPQRSTUVWX")

# Layout fixo do COTAHIST — posições 0-indexed (Python slice)
# Fonte: Manual de Layout de Arquivos – B3 COTAHIST
_COL = {
    "TIPREG": ( 0,   2),   # tipo de registro ("01" = cotação)
    "CODBDI": (10,  12),   # código BDI
    "CODNEG": (12,  24),   # código de negociação (ticker, 12 chars)
    "TPMERC": (24,  27),   # tipo de mercado ("070"/"080" = opções)
    "PREULT": (108, 121),  # preço do último negócio  (÷100 = R$)
    "QUATOT": (152, 170),  # quantidade total negociada (contratos)
    "VOLTOT": (170, 188),  # volume financeiro total    (÷100 = R$)
    "PREEXE": (188, 201),  # preço de exercício/strike  (÷100 = R$)
    "DATVEN": (202, 210),  # data de vencimento (YYYYMMDD)
}


# ══════════════════════════════════════════════════════════════════════
# TAXA DE JUROS — BACEN API pública
# ══════════════════════════════════════════════════════════════════════

def _cdi_bacen() -> float:
    """
    Busca a taxa SELIC/CDI anualizada atual via API pública do BACEN.
    Série 432 = SELIC acumulada diariamente, anualizada.
    Retorna CDI_PADRAO em caso de falha de rede.
    """
    try:
        url = (
            "https://api.bcb.gov.br/dados/serie/"
            "bcdata.sgs.432/dados/ultimos/1?formato=json"
        )
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        valor = resp.json()[0]["valor"].replace(",", ".")
        return float(valor) / 100.0
    except Exception:
        return CDI_PADRAO


# ══════════════════════════════════════════════════════════════════════
# BLACK-SCHOLES + BISSEÇÃO
# ══════════════════════════════════════════════════════════════════════

def _N(x: float) -> float:
    """CDF da distribuição Normal padrão."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_preco(S: float, K: float, T: float, r: float, sigma: float, tipo: str) -> float:
    """
    Preço de uma opção europeia pelo modelo Black-Scholes.

    Parâmetros
    ----------
    S     : preço spot do ativo (R$)
    K     : strike / preço de exercício (R$)
    T     : tempo até vencimento em ANOS (du / 252)
    r     : taxa de juros sem risco anualizada (ex.: 0.1325)
    sigma : volatilidade implícita anualizada (ex.: 0.35)
    tipo  : "CALL" ou "PUT"
    """
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0) if tipo == "CALL" else max(K - S, 0.0)

    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if tipo == "CALL":
        return S * _N(d1) - K * math.exp(-r * T) * _N(d2)
    return K * math.exp(-r * T) * _N(-d2) - S * _N(-d1)


def iv_bisecao(
    preco_mkt: float,
    S: float,
    K: float,
    T: float,
    r: float,
    tipo: str,
    tol: float = 1e-5,
    max_iter: int = 300,
) -> float | None:
    """
    Volatilidade implícita via método da bisseção.

    Resolve: BS(sigma) = preco_mkt  para sigma ∈ [1e-6, 8.0].
    Retorna None se o prêmio de mercado for inconsistente ou não convergir.
    """
    # Preço de mercado deve superar o valor intrínseco
    intrinseco = max(S - K, 0.0) if tipo == "CALL" else max(K - S, 0.0)
    if preco_mkt <= 0 or preco_mkt < intrinseco * 0.99:
        return None

    lo, hi = 1e-6, 8.0

    # Verifica se o preço de mercado está dentro do intervalo de busca
    if bs_preco(S, K, T, r, hi, tipo) < preco_mkt:
        return None  # prêmio maior que BS com 800% de vol — dado suspeito

    for _ in range(max_iter):
        mid  = (lo + hi) * 0.5
        diff = bs_preco(S, K, T, r, mid, tipo) - preco_mkt
        if abs(diff) < tol:
            return mid
        if diff < 0:
            lo = mid
        else:
            hi = mid

    return None  # não convergiu


# ══════════════════════════════════════════════════════════════════════
# COTAHIST — download e parse
# ══════════════════════════════════════════════════════════════════════

def _ultimo_pregao(data: date | None = None) -> date:
    """Retorna a data do último pregão (ignora fins de semana)."""
    d = data or date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def _cache_path(data: date) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"COTAHIST_{data.strftime('%d%m%Y')}.ZIP"


def _baixar_cotahist(data: date, forcar: bool) -> bytes:
    """
    Baixa (ou lê do cache) o ZIP do COTAHIST.
    Se o arquivo do dia indicado não existir (B3 publica com atraso de ~1 dia),
    tenta automaticamente os 5 pregões anteriores.
    """
    # Tenta a data solicitada e, em caso de 404, regride dia a dia
    tentativas = [data]
    d = data
    for _ in range(4):
        d -= timedelta(days=1)
        while d.weekday() >= 5:
            d -= timedelta(days=1)
        tentativas.append(d)

    ultimo_erro: Exception | None = None
    for tentativa in tentativas:
        cache = _cache_path(tentativa)
        if not forcar and cache.exists() and cache.stat().st_size > 5_000:
            return cache.read_bytes()

        url = (
            "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/"
            f"COTAHIST_D{tentativa.strftime('%d%m%Y')}.ZIP"
        )
        try:
            resp = requests.get(url, timeout=90, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            cache.write_bytes(resp.content)
            return resp.content
        except Exception as exc:
            ultimo_erro = exc
            continue  # tenta o pregão anterior

    raise ultimo_erro  # nenhuma data funcionou


def _parse_preco(s: str) -> float | None:
    """Converte campo de preço do COTAHIST (inteiro com 2 decimais implícitos)."""
    try:
        return int(s.strip()) / 100.0
    except (ValueError, AttributeError):
        return None


def parsear_opcoes(conteudo: bytes) -> pd.DataFrame:
    """
    Parseia o COTAHIST (ZIP) e retorna DataFrame com todas as opções
    de ações à vista (TPMERC 070 e 080).

    Colunas:
        subjacente  — primeiros 5 chars do ticker da opção (ex.: "PETR4")
        codneg      — ticker completo da série de opção
        tipo        — "CALL" ou "PUT"
        preult      — prêmio de fechamento (R$)
        strike      — preço de exercício (R$)
        vencimento  — data de vencimento (date)
        volume      — quantidade de contratos negociados
        financeiro  — volume financeiro (R$)
    """
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        nome_txt = next(
            n for n in zf.namelist() if n.upper().endswith(".TXT")
        )
        texto = zf.open(nome_txt).read().decode("latin-1")

    registros = []

    for linha in texto.splitlines():
        if len(linha) < 245:
            continue
        if linha[_COL["TIPREG"][0]:_COL["TIPREG"][1]] != "01":
            continue

        tpmerc = linha[_COL["TPMERC"][0]:_COL["TPMERC"][1]]
        if tpmerc not in ("070", "080"):
            continue

        codneg = linha[_COL["CODNEG"][0]:_COL["CODNEG"][1]].strip()
        if len(codneg) < 5:
            continue

        # No COTAHIST da B3, options usam prefixo de 4 chars do ativo-base
        # (ex.: PETR4 → "PETR", VALE3 → "VALE") e a letra de mês fica em [4].
        # A-L = calls (Jan-Dez); M-X = puts (Jan-Dez).
        letra = codneg[4].upper()
        if letra in CALL_LETTERS:
            tipo = "CALL"
        elif letra in PUT_LETTERS:
            tipo = "PUT"
        else:
            continue

        preult = _parse_preco(linha[_COL["PREULT"][0]:_COL["PREULT"][1]])
        strike = _parse_preco(linha[_COL["PREEXE"][0]:_COL["PREEXE"][1]])

        if not preult or preult <= 0 or not strike or strike <= 0:
            continue

        datven_str = linha[_COL["DATVEN"][0]:_COL["DATVEN"][1]].strip()
        try:
            vencimento = datetime.strptime(datven_str, "%Y%m%d").date()
        except ValueError:
            continue

        try:
            volume = int(linha[_COL["QUATOT"][0]:_COL["QUATOT"][1]].strip() or "0")
        except ValueError:
            volume = 0

        try:
            financeiro = int(
                linha[_COL["VOLTOT"][0]:_COL["VOLTOT"][1]].strip() or "0"
            ) / 100.0
        except ValueError:
            financeiro = 0.0

        registros.append({
            "subjacente": codneg[:4],   # prefixo de 4 chars = ativo-base (padrão B3)
            "codneg":     codneg,
            "tipo":       tipo,
            "preult":     preult,
            "strike":     strike,
            "vencimento": vencimento,
            "volume":     volume,
            "financeiro": financeiro,
        })

    if not registros:
        return pd.DataFrame(
            columns=[
                "subjacente", "codneg", "tipo", "preult",
                "strike", "vencimento", "volume", "financeiro",
            ]
        )
    return pd.DataFrame(registros)


# ══════════════════════════════════════════════════════════════════════
# IV POR ATIVO
# ══════════════════════════════════════════════════════════════════════

def _du_restantes(vencimento: date) -> int:
    hoje = date.today()
    return sum(
        1
        for i in range((vencimento - hoje).days)
        if (hoje + timedelta(days=i)).weekday() < 5
    )


def iv_ativo(
    ticker: str,
    spot: float,
    opcoes_df: pd.DataFrame,
    r: float,
) -> float | None:
    """
    IV média ponderada (peso = volume financeiro) das séries ATM do ativo.

    Critérios de seleção:
      • Prefixo do subjacente = primeiros 4 chars do ticker (padrão B3)
      • Vencimento entre MIN_DU e MAX_DU dias úteis
      • Volume >= MIN_VOL_NEG contratos
      • Strike dentro de ±ATM_BANDA do spot

    Retorna IV anualizada (ex.: 0.35 = 35%) ou None se sem dados.
    """
    prefixo = ticker[:4]  # B3 usa 4 chars: PETR4 → "PETR", VALE3 → "VALE"
    df = opcoes_df[opcoes_df["subjacente"] == prefixo].copy()
    if df.empty:
        return None

    df["du"] = df["vencimento"].apply(_du_restantes)
    df = df[(df["du"] >= MIN_DU) & (df["du"] <= MAX_DU)]
    df = df[df["volume"] >= MIN_VOL_NEG]
    df = df[((df["strike"] - spot).abs() / spot) <= ATM_BANDA]

    if df.empty:
        return None

    ivs, pesos = [], []
    for _, row in df.iterrows():
        T  = row["du"] / 252.0
        iv = iv_bisecao(row["preult"], spot, row["strike"], T, r, row["tipo"])
        if iv and 0.02 <= iv <= 5.0:
            ivs.append(iv)
            pesos.append(max(row["financeiro"], 1.0))

    if not ivs:
        return None

    return float(np.average(ivs, weights=pesos))


# ══════════════════════════════════════════════════════════════════════
# SELEÇÃO DE LEGS POR ESTRATÉGIA
# ══════════════════════════════════════════════════════════════════════

def _selecionar_leg(
    opcoes_df: pd.DataFrame,
    prefixo: str,
    tipo: str,
    spot: float,
    du_min: int,
    du_max: int,
    otm_pct: float = 0.0,
    min_vol: int = 5,
) -> dict | None:
    """
    Seleciona a melhor opção (single leg).
    otm_pct == 0 → ATM; otm_pct > 0 → OTM afastado nessa direção.
    """
    df = opcoes_df[opcoes_df["subjacente"] == prefixo].copy()
    df = df[df["tipo"] == tipo]
    df["du"] = df["vencimento"].apply(_du_restantes)
    df = df[(df["du"] >= du_min) & (df["du"] <= du_max)]
    df = df[df["volume"] >= min_vol]

    if df.empty:
        return None

    target = spot * (1 + otm_pct) if tipo == "CALL" else spot * (1 - otm_pct)
    df = df.copy()
    df["dist"] = (df["strike"] - target).abs()
    row = df.nsmallest(1, "dist").iloc[0]

    return {
        "codneg":     row["codneg"],
        "tipo":       row["tipo"],
        "strike":     float(row["strike"]),
        "vencimento": row["vencimento"],
        "preult":     float(row["preult"]),
        "du":         int(row["du"]),
        "volume":     int(row["volume"]),
    }


def _montar_leg(lg: dict, acao: str, label: str) -> dict:
    d = lg.copy()
    d["acao"]  = acao
    d["label"] = label
    return d


def _preco_compra_simples(lg: dict) -> dict:
    p = lg["preult"]
    return {
        "entrada":   p,
        "stop":      round(p * 0.50, 2),
        "alvo":      round(p * 2.00, 2),
        "rr":        "1 : 2",
        "descricao": f"Paga R$ {p:.2f}/contrato. Stop: −50% do prêmio. Alvo: dobrar o prêmio.",
    }


def _preco_venda_simples(lg: dict) -> dict:
    c = lg["preult"]
    return {
        "entrada":   c,
        "stop":      round(c * 3.00, 2),
        "alvo":      round(c * 0.20, 2),
        "rr":        "1 : 0.25  (alta prob.)",
        "descricao": f"Recebe R$ {c:.2f}/contrato. Stop: recomprar se chegar a 3×. Alvo: fechar a 20% do crédito.",
    }


def _preco_spread_debito(compra: dict, venda: dict) -> dict:
    debito   = round(compra["preult"] - venda["preult"], 2)
    if debito <= 0:
        debito = compra["preult"]  # proteção se OTM tiver prêmio maior que ATM
    amplitude = round(abs(compra["strike"] - venda["strike"]), 2)
    max_ganho = round(max(amplitude - debito, 0.01), 2)
    rr_val    = round(max_ganho / debito, 1) if debito > 0 else 0
    return {
        "entrada":   debito,
        "stop":      debito,   # perde o débito inteiro
        "alvo":      max_ganho,
        "rr":        f"1 : {rr_val}",
        "descricao": (
            f"Débito líquido: R$ {debito:.2f}/contrato "
            f"(compra R$ {compra['preult']:.2f} − venda R$ {venda['preult']:.2f}). "
            f"Ganho máximo: R$ {max_ganho:.2f} se ação fechar além do strike vendido."
        ),
    }


def _preco_spread_credito(venda: dict, compra: dict) -> dict:
    credito  = round(venda["preult"] - compra["preult"], 2)
    if credito <= 0:
        credito = venda["preult"]
    amplitude = round(abs(venda["strike"] - compra["strike"]), 2)
    max_perda = round(max(amplitude - credito, 0.01), 2)
    rr_val    = round(credito / max_perda, 2) if max_perda > 0 else 0
    return {
        "entrada":   credito,
        "stop":      max_perda,
        "alvo":      round(credito * 0.20, 2),
        "rr":        f"1 : {rr_val}  (crédito/risco)",
        "descricao": (
            f"Crédito líquido: R$ {credito:.2f}/contrato "
            f"(venda R$ {venda['preult']:.2f} − compra R$ {compra['preult']:.2f}). "
            f"Risco máximo: R$ {max_perda:.2f}. Alvo: fechar a 20% do crédito."
        ),
    }


def selecionar_detalhes_estrategias(
    ticker: str,
    spot: float,
    opcoes_df: pd.DataFrame,
    chave: str,
    du_min: int = 20,
    du_max: int = 40,
) -> list[dict]:
    """
    Retorna lista de dicts — um por estratégia do catálogo para esse cenário.

    Cada item:
        nome      — nome da estratégia
        nivel     — nível de complexidade
        legs      — lista de legs [{codneg, tipo, strike, vencimento, preult, du, acao, label}]
        preco_op  — {entrada, stop, alvo, rr, descricao}
        completa  — bool: True se todos os legs foram encontrados
    """
    from scanner_opcoes_supertrend import CATALOGO  # evita importação circular
    estrategias_catalogo = CATALOGO.get(chave, [])

    prefixo = ticker[:4]

    def leg(tipo, otm=0.0):
        return _selecionar_leg(opcoes_df, prefixo, tipo, spot, du_min, du_max, otm)

    resultado = []

    for nome, nivel in estrategias_catalogo:
        num = nome.split("–")[0].strip()

        legs_raw = []
        preco_op = {}

        # ── Compras simples ──────────────────────────────────────────
        if num == "1":
            atm_call = leg("CALL")
            if atm_call:
                legs_raw = [_montar_leg(atm_call, "COMPRAR", "Call ATM")]
                preco_op = _preco_compra_simples(atm_call)

        elif num == "5":
            atm_put = leg("PUT")
            if atm_put:
                legs_raw = [_montar_leg(atm_put, "COMPRAR", "Put ATM")]
                preco_op = _preco_compra_simples(atm_put)

        # ── Vendas simples ───────────────────────────────────────────
        elif num == "2":
            atm_put = leg("PUT")
            if atm_put:
                legs_raw = [_montar_leg(atm_put, "VENDER", "Put ATM")]
                preco_op = _preco_venda_simples(atm_put)

        elif num == "6":
            atm_call = leg("CALL")
            if atm_call:
                legs_raw = [_montar_leg(atm_call, "VENDER", "Call ATM")]
                preco_op = _preco_venda_simples(atm_call)

        elif num == "16":
            atm_call = leg("CALL")
            if atm_call:
                legs_raw = [_montar_leg(atm_call, "VENDER", "Call ATM (coberta)")]
                p = atm_call["preult"]
                preco_op = {
                    "entrada":   p,
                    "stop":      None,
                    "alvo":      p,
                    "rr":        "—  (protegido pela ação)",
                    "descricao": f"Recebe R$ {p:.2f}/contrato. Requer ter 100 ações. Stop gerenciado pela posição em ações.",
                }

        # ── Spreads de débito ────────────────────────────────────────
        elif num == "3":
            # Trava de Alta c/ Call: compra ATM + vende OTM (~5% acima)
            atm_call = leg("CALL", 0.00)
            otm_call = leg("CALL", 0.05)
            if atm_call and otm_call and atm_call["codneg"] != otm_call["codneg"]:
                legs_raw = [
                    _montar_leg(atm_call, "COMPRAR", "Call ATM (perna comprada)"),
                    _montar_leg(otm_call, "VENDER",  "Call OTM +5% (perna vendida)"),
                ]
                preco_op = _preco_spread_debito(atm_call, otm_call)

        elif num == "7":
            # Trava de Baixa c/ Put: compra ATM + vende OTM (~5% abaixo)
            atm_put = leg("PUT", 0.00)
            otm_put = leg("PUT", 0.05)
            if atm_put and otm_put and atm_put["codneg"] != otm_put["codneg"]:
                legs_raw = [
                    _montar_leg(atm_put, "COMPRAR", "Put ATM (perna comprada)"),
                    _montar_leg(otm_put, "VENDER",  "Put OTM −5% (perna vendida)"),
                ]
                preco_op = _preco_spread_debito(atm_put, otm_put)

        # ── Spreads de crédito ───────────────────────────────────────
        elif num == "4":
            # Trava de Alta c/ Put: vende ATM + compra OTM abaixo
            atm_put = leg("PUT", 0.00)
            otm_put = leg("PUT", 0.05)
            if atm_put and otm_put and atm_put["codneg"] != otm_put["codneg"]:
                legs_raw = [
                    _montar_leg(atm_put, "VENDER",  "Put ATM (perna vendida)"),
                    _montar_leg(otm_put, "COMPRAR", "Put OTM −5% (proteção)"),
                ]
                preco_op = _preco_spread_credito(atm_put, otm_put)

        elif num == "8":
            # Trava de Baixa c/ Call: vende ATM + compra OTM acima
            atm_call = leg("CALL", 0.00)
            otm_call = leg("CALL", 0.05)
            if atm_call and otm_call and atm_call["codneg"] != otm_call["codneg"]:
                legs_raw = [
                    _montar_leg(atm_call, "VENDER",  "Call ATM (perna vendida)"),
                    _montar_leg(otm_call, "COMPRAR", "Call OTM +5% (proteção)"),
                ]
                preco_op = _preco_spread_credito(atm_call, otm_call)

        # ── Straddles ────────────────────────────────────────────────
        elif num == "10":
            atm_call = leg("CALL")
            atm_put  = leg("PUT")
            if atm_call and atm_put:
                legs_raw = [
                    _montar_leg(atm_call, "VENDER", "Call ATM"),
                    _montar_leg(atm_put,  "VENDER", "Put ATM"),
                ]
                credito = atm_call["preult"] + atm_put["preult"]
                preco_op = {
                    "entrada":   round(credito, 2),
                    "stop":      round(credito * 3.00, 2),
                    "alvo":      round(credito * 0.20, 2),
                    "rr":        "1 : 0.25  (alta prob.)",
                    "descricao": f"Recebe R$ {credito:.2f} total. Stop: recomprar se dobrar. Alvo: fechar a 20%.",
                }

        elif num == "12":
            atm_call = leg("CALL")
            atm_put  = leg("PUT")
            if atm_call and atm_put:
                legs_raw = [
                    _montar_leg(atm_call, "COMPRAR", "Call ATM"),
                    _montar_leg(atm_put,  "COMPRAR", "Put ATM"),
                ]
                debito = atm_call["preult"] + atm_put["preult"]
                preco_op = {
                    "entrada":   round(debito, 2),
                    "stop":      round(debito * 0.50, 2),
                    "alvo":      round(debito * 2.00, 2),
                    "rr":        "1 : 2",
                    "descricao": f"Paga R$ {debito:.2f} total. Stop: −50%. Alvo: dobrar.",
                }

        # Estratégias sem legs automáticos (requerem ação/estrutura especial)
        else:
            legs_raw = []
            preco_op = {"descricao": "Estrutura complexa — montar manualmente no homebroker."}

        resultado.append({
            "nome":     nome,
            "nivel":    nivel,
            "legs":     legs_raw,
            "preco_op": preco_op,
            "completa": len(legs_raw) > 0,
        })

    return resultado


# ══════════════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA
# ══════════════════════════════════════════════════════════════════════

def obter_iv_todos(
    tickers: list[str],
    precos_spot: dict[str, float],
    data: date | None = None,
    forcar_download: bool = False,
    verbose: bool = True,
    retornar_opcoes: bool = False,
) -> dict[str, float | None] | tuple:
    """
    Baixa o COTAHIST e retorna a IV real de cada ativo.

    Parâmetros
    ----------
    tickers         : lista de tickers sem .SA  (ex.: ["PETR4", "VALE3"])
    precos_spot     : {ticker: último preço de fechamento}
    data            : pregão desejado (None = último dia útil)
    forcar_download : ignora cache e baixa novamente
    verbose         : imprime progresso no console
    retornar_opcoes : se True, retorna (iv_dict, opcoes_df, cdi)

    Retorna
    -------
    {ticker: IV anualizada (float) | None}
    ou, se retornar_opcoes=True: (dict_iv, opcoes_df, cdi_float)
    """
    pregao = _ultimo_pregao(data)

    # ── Download ──────────────────────────────────────────────────────
    if verbose:
        print(
            f"  [COTAHIST] Buscando a partir de {pregao.strftime('%d/%m/%Y')}...",
            end=" ", flush=True,
        )
    try:
        conteudo = _baixar_cotahist(pregao, forcar_download)
        if verbose:
            print(f"{len(conteudo)/1024:.0f} KB  OK")
    except Exception as e:
        if verbose:
            print(f"FALHOU ({e})")
        return {t: None for t in tickers}

    # ── Parse ─────────────────────────────────────────────────────────
    if verbose:
        print("  [COTAHIST] Parseando opções...", end=" ", flush=True)
    try:
        opcoes_df = parsear_opcoes(conteudo)
        if verbose:
            print(f"{len(opcoes_df):,} séries encontradas")
    except Exception as e:
        if verbose:
            print(f"FALHOU ({e})")
        return {t: None for t in tickers}

    # ── CDI / taxa sem risco ──────────────────────────────────────────
    if verbose:
        print("  [BACEN]    Taxa CDI...", end=" ", flush=True)
    r = _cdi_bacen()
    if verbose:
        print(f"{r*100:.2f}% a.a.")

    # ── IV por ativo ──────────────────────────────────────────────────
    resultado: dict[str, float | None] = {}
    for ticker in tickers:
        spot = precos_spot.get(ticker)
        if spot is None:
            resultado[ticker] = None
        else:
            resultado[ticker] = iv_ativo(ticker, spot, opcoes_df, r)

    n_ok = sum(1 for v in resultado.values() if v is not None)
    if verbose:
        print(f"  [IV]       {n_ok}/{len(tickers)} ativos com IV calculada")

    if retornar_opcoes:
        return resultado, opcoes_df, r
    return resultado
