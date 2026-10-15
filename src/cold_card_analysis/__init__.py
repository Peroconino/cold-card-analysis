import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import orjson

    loads = orjson.loads
except ImportError:
    loads = json.loads


def lorenz(values, max_points=2000):
    """Retorna (p, L): fração acumulada de endereços e fração acumulada do volume."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x > 0]
    cum = np.insert(np.cumsum(x), 0, 0.0) / x.sum()
    p = np.linspace(0, 1, len(cum))
    if len(cum) > max_points:  # reduz pontos para o gráfico não ficar pesado
        idx = np.linspace(0, len(cum) - 1, max_points).astype(int)
        p, cum = p[idx], cum[idx]
    return p, cum


def gini(values):
    """Gini de uma lista de valores >= 0 (0 = igualdade total, 1 = desigualdade máxima)."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x > 0]
    n = len(x)
    if n == 0:
        return float("nan")
    cum = np.cumsum(x)
    return (n + 1 - 2 * (cum / cum[-1]).sum()) / n


def volume_by_address(all_txs):
    """Volume total gasto (inputs) por endereço em todos os blocos carregados."""
    vol = defaultdict(int)
    for tx in all_txs:
        for i in tx.get("inputs", []):
            po = i.get("prev_out")
            if po and po.get("addr"):
                vol[po["addr"]] += po["value"]
    return vol


def plot_gini(all_txs, senders_total, out="gini.png"):
    vol = volume_by_address(all_txs)
    chain_vol = list(vol.values())
    victim_vol = [vol[a] for a in senders_total if a in vol]

    g_chain, g_victim = gini(chain_vol), gini(victim_vol)
    pc, lc = lorenz(chain_vol)
    pv, lv = lorenz(victim_vol)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.6, 1]}
    )

    # Painel 1: curvas de Lorenz
    ax1.plot([0, 1], [0, 1], "--", color="gray", label="Igualdade perfeita")
    ax1.plot(
        pc,
        lc,
        color="tab:blue",
        lw=2,
        label=f"Blockchain (Gini = {g_chain:.3f}, n = {len(chain_vol):,})",
    )
    ax1.plot(
        pv,
        lv,
        color="tab:red",
        lw=2,
        label=f"Vítimas (Gini = {g_victim:.3f}, n = {len(victim_vol):,})",
    )
    ax1.set_xlabel("Fração acumulada de endereços")
    ax1.set_ylabel("Fração acumulada do volume movimentado")
    ax1.set_title("Curvas de Lorenz")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Painel 2: barras com o Gini
    bars = ax2.bar(
        ["Blockchain", "Vítimas"], [g_chain, g_victim], color=["tab:blue", "tab:red"]
    )
    ax2.bar_label(bars, fmt="%.3f", padding=3)
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("Gini")
    ax2.set_title("Coeficiente de Gini")
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("Desigualdade do volume movimentado por endereço", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.show()


def parse_block(path: Path):
    with open(path, "rb") as f:
        block = loads(f.read())
    txs = block["tx"] if isinstance(block, dict) else block

    # Remove só os campos pesados que não entram nas estatísticas
    for tx in txs:
        for i in tx.get("inputs", []):
            i.pop("witness", None)
            i.pop("script", None)
            po = i.get("prev_out")
            if po:
                po.pop("script", None)
                po.pop("spending_outpoints", None)
        for o in tx.get("out", []):
            o.pop("script", None)
            o.pop("spending_outpoints", None)
    return txs


def load_dir(directory: str, workers: int = 4):
    files = sorted(Path(directory).glob("*"))
    all_txs = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for txs in ex.map(parse_block, files, chunksize=4):
            all_txs.extend(txs)

    return all_txs


def analyze(all_txs, target="bc1qnk4zh9qcnap2mycp56qjrgza3cc8ylrh8fecp0"):
    utxos = []  # outputs pagos ao target
    senders_total = defaultdict(int)  # endereço de origem -> valor enviado
    tx_hashes = set()

    for tx in all_txs:
        outs = [o for o in tx.get("out", []) if o.get("addr") == target]
        if not outs:
            continue

        in_by_addr = defaultdict(int)
        for i in tx.get("inputs", []):
            po = i.get("prev_out")
            if po and po.get("addr"):
                in_by_addr[po["addr"]] += po["value"]

        # ignora troco / autotransferência da própria carteira
        if target in in_by_addr:
            continue

        tx_hashes.add(tx["hash"])
        for o in outs:
            utxos.append(
                {
                    "tx_hash": tx["hash"],
                    "n": o["n"],
                    "value": o["value"],
                    "spent": o["spent"],
                    "time": tx["time"],
                }
            )
        for addr, val in in_by_addr.items():
            senders_total[addr] += val

    return utxos, senders_total, tx_hashes


SAT = 100_000_000


def report(utxos, senders_total, tx_hashes):
    total = sum(u["value"] for u in utxos)
    unspent = [u for u in utxos if not u["spent"]]

    print(f"Total recebido:              {total / SAT:.8f} BTC")
    print(f"UTXOs recebidos:             {len(utxos)}")
    print(
        f"  ainda não gastos:          {len(unspent)} "
        f"({sum(u['value'] for u in unspent) / SAT:.8f} BTC)"
    )
    print(f"Transações de entrada:       {len(tx_hashes)}")
    print(f"Endereços de origem:         {len(senders_total)}")

    if utxos:
        times = [u["time"] for u in utxos]
        print(f"Período (unix):              {min(times)} -> {max(times)}")

    print("\nTop 10 endereços por valor enviado:")
    top = sorted(senders_total.items(), key=lambda kv: kv[1], reverse=True)[:10]
    for addr, val in top:
        print(f"  {addr}  {val / SAT:.8f} BTC")


BENFORD = np.log10(1 + 1 / np.arange(1, 10))  # P(d) = log10(1 + 1/d)


def first_digits(values):
    """Primeiro dígito significativo de cada valor > 0."""
    return np.fromiter((int(str(v)[0]) for v in values if v > 0), dtype=np.int8)


def digit_freq(values):
    d = first_digits(values)
    counts = np.bincount(d, minlength=10)[1:10]
    return counts / counts.sum(), len(d)


def chain_output_values(all_txs):
    """Valor de todos os outputs da blockchain (ignora value == 0, ex.: OP_RETURN)."""
    return [
        o["value"] for tx in all_txs for o in tx.get("out", []) if o.get("value", 0) > 0
    ]


def mad(freq):
    """Mean Absolute Deviation em relação a Benford (critério de Nigrini)."""
    return np.mean(np.abs(freq - BENFORD))


def mad_label(m):
    if m < 0.006:
        return "conformidade alta"
    if m < 0.012:
        return "conformidade aceitável"
    if m < 0.015:
        return "conformidade marginal"
    return "não conforme"


def chi2_stat(freq, n):
    obs = freq * n
    exp = BENFORD * n
    return ((obs - exp) ** 2 / exp).sum()  # 8 graus de liberdade; crítico a 5% = 15.51


def plot_benford(all_txs, utxos, out="benford.png"):
    stolen_vals = [u["value"] for u in utxos]
    chain_vals = chain_output_values(all_txs)

    f_stolen, n_stolen = digit_freq(stolen_vals)
    f_chain, n_chain = digit_freq(chain_vals)

    m_s, m_c = mad(f_stolen), mad(f_chain)
    x2_s, x2_c = chi2_stat(f_stolen, n_stolen), chi2_stat(f_chain, n_chain)

    digits = np.arange(1, 10)
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.5, 1]}
    )

    # Painel 1: distribuições
    ax1.bar(
        digits - w / 2,
        f_stolen * 100,
        w,
        color="tab:red",
        alpha=0.85,
        label=f"Roubadas (n = {n_stolen:,})",
    )
    ax1.bar(
        digits + w / 2,
        f_chain * 100,
        w,
        color="tab:blue",
        alpha=0.85,
        label=f"Blockchain (n = {n_chain:,})",
    )
    ax1.plot(digits, BENFORD * 100, "ko--", lw=1.8, label="Benford teórico")
    ax1.set_xticks(digits)
    ax1.set_xlabel("Primeiro dígito")
    ax1.set_ylabel("Frequência (%)")
    ax1.set_title("Distribuição do primeiro dígito")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Painel 2: desvio em relação a Benford
    ax2.axhline(0, color="k", lw=1)
    ax2.plot(
        digits,
        (f_stolen - BENFORD) * 100,
        "o-",
        color="tab:red",
        label=f"Roubadas (MAD = {m_s:.4f})",
    )
    ax2.plot(
        digits,
        (f_chain - BENFORD) * 100,
        "s-",
        color="tab:blue",
        label=f"Blockchain (MAD = {m_c:.4f})",
    )
    ax2.set_xticks(digits)
    ax2.set_xlabel("Primeiro dígito")
    ax2.set_ylabel("Desvio de Benford (p.p.)")
    ax2.set_title("Desvio em relação ao esperado")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("Lei de Benford: valores de outputs (satoshis)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.show()

    print(f"{'':16}{'n':>12}{'MAD':>10}{'chi²':>12}   conformidade (Nigrini)")
    print(f"{'Roubadas':16}{n_stolen:>12,}{m_s:>10.4f}{x2_s:>12.1f}   {mad_label(m_s)}")
    print(
        f"{'Blockchain':16}{n_chain:>12,}{m_c:>10.4f}{x2_c:>12.1f}   {mad_label(m_c)}"
    )
    print("(chi² crítico a 5% com 8 g.l. = 15.51)")


def main() -> None:
    all_txs = load_dir("blocos/")
    print(f"{len(all_txs)} transações carregadas\n")

    utxos, senders_total, tx_hashes = analyze(all_txs)
    report(utxos, senders_total, tx_hashes)
    plot_gini(all_txs, senders_total)
    plot_benford(all_txs, utxos)
