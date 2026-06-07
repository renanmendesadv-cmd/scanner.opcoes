#!/usr/bin/env python3
"""
carteira_sinais.py — Rastreamento de sinais gerados pelo scanner
────────────────────────────────────────────────────────────────
Registra cada sinal gerado, acompanha o preço das opções via
COTAHIST e calcula o P&L diário de cada posição.

Status possíveis:
  Aberto           — posição ainda em aberto
  Alvo atingido    — preço atual >= alvo
  Stop atingido    — preço atual <= stop
  Expirado         — opção venceu sem fechar antes
"""

import hashlib
import pandas as pd
from datetime import date
from pathlib import Path

PASTA   = Path(__file__).parent
ARQUIVO = PASTA / "sinais_rastreados.csv"

COLUNAS = [
    "id",
    "data_sinal",
    "ativo",
    "estrategia",
    # Leg 1
    "leg1_ticker",
    "leg1_acao",           # COMPRAR / VENDER
    "leg1_preco_entrada",
    # Leg 2 (spread)
    "leg2_ticker",
    "leg2_acao",
    "leg2_preco_entrada",
    # Preços da operação
    "preco_entrada",       # Net debit (debito) ou net credit
    "stop",
    "alvo",
    # Rastreamento
    "status",              # Aberto / Alvo atingido / Stop atingido / Expirado
    "data_atualizacao",
    "preco_atual",
    "pnl_rs",
    "pnl_pct",
    # Fechamento
    "data_fechamento",
    "preco_fechamento",
]


# ──────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────

def _gerar_id(data_sinal: str, ativo: str, estrategia: str) -> str:
    chave = f"{data_sinal}|{ativo}|{estrategia}"
    return hashlib.md5(chave.encode()).hexdigest()[:8].upper()


def carregar() -> pd.DataFrame:
    """Carrega o CSV de sinais. Cria vazio se não existir."""
    if not ARQUIVO.exists():
        return pd.DataFrame(columns=COLUNAS)
    df = pd.read_csv(ARQUIVO, dtype=str)
    for col in COLUNAS:
        if col not in df.columns:
            df[col] = None
    return df[COLUNAS].copy()


def salvar(df: pd.DataFrame) -> None:
    df[COLUNAS].to_csv(ARQUIVO, index=False)


# ──────────────────────────────────────────────────────────────────────
# REGISTRO
# ──────────────────────────────────────────────────────────────────────

def registrar_sinais(resultados: list) -> int:
    """
    Percorre os resultados do scanner e registra sinais novos no CSV.
    Só registra estratégias completas (com legs reais encontradas).
    Retorna o número de novos sinais adicionados.
    """
    df   = carregar()
    hoje = date.today().isoformat()
    novos = 0

    for r in resultados:
        if not r.get("ok"):
            continue
        ativo = r["ticker"].replace(".SA", "")

        for det in r.get("estrategias_detalhadas", []):
            if not det.get("completa") or not det.get("legs"):
                continue

            po         = det.get("preco_op", {})
            entrada    = po.get("entrada")
            if entrada is None:
                continue

            estrategia = det["nome"]
            id_sinal   = _gerar_id(hoje, ativo, estrategia)

            # Não duplicar mesmo sinal no mesmo dia
            if id_sinal in df["id"].values:
                continue

            legs = det["legs"]
            leg1 = legs[0] if len(legs) > 0 else {}
            leg2 = legs[1] if len(legs) > 1 else {}

            nova = {
                "id":                  id_sinal,
                "data_sinal":          hoje,
                "ativo":               ativo,
                "estrategia":          estrategia,
                "leg1_ticker":         leg1.get("codneg", ""),
                "leg1_acao":           leg1.get("acao", ""),
                "leg1_preco_entrada":  str(round(float(leg1.get("preult", 0)), 4)),
                "leg2_ticker":         leg2.get("codneg", "") if leg2 else "",
                "leg2_acao":           leg2.get("acao", "") if leg2 else "",
                "leg2_preco_entrada":  str(round(float(leg2.get("preult", 0)), 4)) if leg2 else "",
                "preco_entrada":       str(round(float(entrada), 4)),
                "stop":                str(round(float(po["stop"]), 4))  if po.get("stop")  is not None else "",
                "alvo":                str(round(float(po["alvo"]), 4))  if po.get("alvo")  is not None else "",
                "status":              "Aberto",
                "data_atualizacao":    hoje,
                "preco_atual":         str(round(float(entrada), 4)),
                "pnl_rs":              "0.00",
                "pnl_pct":             "0.00",
                "data_fechamento":     "",
                "preco_fechamento":    "",
            }

            df    = pd.concat([df, pd.DataFrame([nova])], ignore_index=True)
            novos += 1

    if novos > 0:
        salvar(df)

    return novos


# ──────────────────────────────────────────────────────────────────────
# ATUALIZAÇÃO DE PREÇOS
# ──────────────────────────────────────────────────────────────────────

def _preco_leg(codneg: str, opcoes_df) -> float | None:
    """Retorna o último preço de uma opção no COTAHIST."""
    if not codneg or opcoes_df is None:
        return None
    codneg = codneg.strip()
    mask = opcoes_df["codneg"].str.strip() == codneg
    if not mask.any():
        return None
    return float(opcoes_df.loc[mask, "preult"].iloc[0])


def _calcular_pnl(
    leg1_acao: str, leg1_entry: float, leg1_atual: float,
    leg2_acao: str | None = None,
    leg2_entry: float | None = None,
    leg2_atual: float | None = None,
) -> float:
    """
    P&L por unidade considerando a ação de cada leg.
      COMPRAR: ganho = atual - entrada  (comprou, vale mais/menos agora)
      VENDER:  ganho = entrada - atual  (vendeu, custa mais/menos fechar)
    """
    pnl = 0.0
    if leg1_acao == "COMPRAR":
        pnl += leg1_atual - leg1_entry
    else:
        pnl += leg1_entry - leg1_atual

    if leg2_acao and leg2_entry is not None and leg2_atual is not None:
        if leg2_acao == "COMPRAR":
            pnl += leg2_atual - leg2_entry
        else:
            pnl += leg2_entry - leg2_atual

    return pnl


def atualizar_precos(opcoes_df) -> tuple[pd.DataFrame, int]:
    """
    Atualiza o preço atual e P&L de todos os sinais em aberto
    usando os preços do COTAHIST mais recente.

    Retorna (df_atualizado, n_atualizados).
    """
    df     = carregar()
    hoje   = date.today().isoformat()
    atualizados = 0

    abertos_idx = df.index[df["status"] == "Aberto"].tolist()

    for idx in abertos_idx:
        row = df.loc[idx]

        # Buscar preços atuais das legs
        p1 = _preco_leg(str(row["leg1_ticker"]), opcoes_df)
        if p1 is None:
            continue   # opção não encontrada no COTAHIST (vencida ou sem dados)

        leg2_tk = str(row["leg2_ticker"]).strip()
        p2 = _preco_leg(leg2_tk, opcoes_df) if leg2_tk else None

        try:
            leg1_entry = float(row["leg1_preco_entrada"])
            leg1_acao  = str(row["leg1_acao"])
        except (ValueError, TypeError):
            continue

        leg2_acao  = str(row["leg2_acao"]) if leg2_tk and row["leg2_acao"] else None
        try:
            leg2_entry = float(row["leg2_preco_entrada"]) if leg2_tk and row["leg2_preco_entrada"] else None
        except (ValueError, TypeError):
            leg2_entry = None

        # P&L por unidade
        pnl = _calcular_pnl(
            leg1_acao, leg1_entry, p1,
            leg2_acao, leg2_entry, p2,
        )

        try:
            entrada_ref = float(row["preco_entrada"])
        except (ValueError, TypeError):
            entrada_ref = abs(leg1_entry) or 1.0

        pnl_pct = (pnl / abs(entrada_ref)) * 100 if entrada_ref != 0 else 0.0

        # Preço líquido atual para comparar com stop/alvo
        # Usa o mesmo sinal que o preco_entrada (debit = negativo ao fechar, credit = positivo)
        preco_atual_net = entrada_ref + pnl   # simplificação: entrada + variação

        # Verificar stop/alvo
        try:
            stop_str = str(row["stop"]).strip()
            alvo_str = str(row["alvo"]).strip()
            stop_val = float(stop_str) if stop_str not in ("", "nan") else None
            alvo_val = float(alvo_str) if alvo_str not in ("", "nan") else None
        except (ValueError, TypeError):
            stop_val = alvo_val = None

        status = "Aberto"
        has_leg2 = bool(leg2_tk and leg2_tk not in ("", "nan"))

        if leg1_acao == "COMPRAR":
            # ── Posição de débito (compra de opção ou spread de débito) ──
            # preco_atual_net = valor atual da posição (entrada + variação)
            # alvo: posição vale >= alvo_val
            if alvo_val is not None and preco_atual_net >= alvo_val:
                status = "Alvo atingido"
            # stop: só aciona quando stop_val < entrada (faz sentido para compra)
            # Spreads de débito têm stop = net_debit = entrada → ignorar nesse caso
            elif stop_val is not None and stop_val < entrada_ref * 0.99 and preco_atual_net <= stop_val:
                status = "Stop atingido"
            elif pnl_pct <= -90:
                # Rede de segurança: perdeu ≥ 90 % do investimento
                status = "Stop atingido"

        else:
            # ── Posição de crédito (venda de opção ou spread de crédito) ──
            # P&L positivo = bom (opção caiu, recompra mais barata)
            # alvo: pnl ≥ (crédito recebido − preço-alvo de saída)
            if alvo_val is not None:
                if alvo_val < entrada_ref:
                    # Venda normal / spread crédito: alvo é preço de recompra
                    alvo_ok = pnl >= (entrada_ref - alvo_val)
                else:
                    # Lançamento coberto / alvo = crédito cheio: expirar sem valor
                    alvo_ok = pnl >= entrada_ref * 0.90
                if alvo_ok:
                    status = "Alvo atingido"

            # stop: depende se há segunda perna (spread) ou não (venda seca)
            if status == "Aberto" and stop_val is not None:
                if has_leg2:
                    # Spread de crédito: stop_val = perda máxima em R$
                    if pnl <= -stop_val:
                        status = "Stop atingido"
                else:
                    # Venda seca: stop_val = preço da opção que aciona saída
                    if pnl <= (entrada_ref - stop_val):
                        status = "Stop atingido"

        df.at[idx, "preco_atual"]       = str(round(preco_atual_net, 4))
        df.at[idx, "pnl_rs"]            = str(round(pnl, 4))
        df.at[idx, "pnl_pct"]           = str(round(pnl_pct, 2))
        df.at[idx, "data_atualizacao"]  = hoje
        df.at[idx, "status"]            = status

        if status != "Aberto":
            df.at[idx, "data_fechamento"]  = hoje
            df.at[idx, "preco_fechamento"] = str(round(preco_atual_net, 4))

        atualizados += 1

    if atualizados > 0:
        salvar(df)

    return df, atualizados


# ──────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ──────────────────────────────────────────────────────────────────────

def calcular_metricas(df: pd.DataFrame) -> dict:
    """Retorna métricas resumidas da carteira."""
    total     = len(df)
    abertos   = int((df["status"] == "Aberto").sum())
    fechados  = df[df["status"] != "Aberto"].copy()

    n_fechados = len(fechados)
    n_ganhos   = 0
    pnl_total  = 0.0

    if n_fechados > 0:
        try:
            pnls = pd.to_numeric(fechados["pnl_rs"], errors="coerce").fillna(0)
            n_ganhos  = int((pnls > 0).sum())
            pnl_total = float(pnls.sum())
        except Exception:
            pass

    taxa_acerto = (n_ganhos / n_fechados * 100) if n_fechados > 0 else 0.0

    # P&L total incluindo abertos
    try:
        pnl_abertos = pd.to_numeric(df.loc[df["status"] == "Aberto", "pnl_rs"], errors="coerce").fillna(0).sum()
    except Exception:
        pnl_abertos = 0.0

    return {
        "total":        total,
        "abertos":      abertos,
        "fechados":     n_fechados,
        "n_ganhos":     n_ganhos,
        "taxa_acerto":  taxa_acerto,
        "pnl_fechados": pnl_total,
        "pnl_abertos":  float(pnl_abertos),
        "pnl_total":    pnl_total + float(pnl_abertos),
    }
