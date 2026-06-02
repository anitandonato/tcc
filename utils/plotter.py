"""
Gera todos os graficos comparativos do TCC em preto e branco (ABNT).

Fontes de dados:
  config.PERF_CSV          -> performance_combined.png, memory_usage.png
  config.ACC_CSV           -> eer_comparison.png
  results/scores/*_roc.csv -> roc_lfw.png, roc_basepropr.png

Uso:
  python utils/plotter.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # sem janela grafica
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------------------------
# Estilos por biblioteca (preto e branco, ABNT)
# ---------------------------------------------------------------------------

LIB_STYLES = {
    "OpenCV (Haar+HOG)":   {"hatch": "",   "color": "#cccccc", "linestyle": "-",  "edgecolor": "black"},
    "Dlib (HOG)":          {"hatch": "//", "color": "#888888", "linestyle": "--", "edgecolor": "black"},
    "DeepFace (VGG+RetinaFace)": {"hatch": "xx", "color": "#444444", "linestyle": "-.", "edgecolor": "black"},
}

LIB_KEYS = {
    "OpenCV (Haar+HOG)":   "opencv",
    "Dlib (HOG)":          "dlib",
    "DeepFace (VGG+RetinaFace)": "deepface",
}

SCORES_DIR = config.RESULTS_DIR / "scores"

# ---------------------------------------------------------------------------
# Utilitario
# ---------------------------------------------------------------------------

def _save(fig, filename: str):
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CHARTS_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"  Salvo: {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Performance: tempo (ms) e FPS
# ---------------------------------------------------------------------------

def plot_performance(df: pd.DataFrame):
    libs = df["Biblioteca"].tolist()
    x = np.arange(len(libs))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for i, lib in enumerate(libs):
        style = LIB_STYLES.get(lib, {"hatch": "", "color": "#aaaaaa", "edgecolor": "black"})
        tempo = df.loc[df["Biblioteca"] == lib, "Tempo_medio_ms"].values[0]
        fps   = df.loc[df["Biblioteca"] == lib, "FPS"].values[0]

        b1 = ax1.bar(x[i] - width / 2, tempo, width,
                     color=style["color"], hatch=style["hatch"],
                     edgecolor=style["edgecolor"], label=lib if i == 0 else "")
        ax2.bar(x[i] + width / 2, fps, width,
                color="white", hatch=style["hatch"],
                edgecolor=style["edgecolor"])

    ax1.set_ylabel("Tempo medio de inferencia (ms)")
    ax2.set_ylabel("FPS (quadros por segundo)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(libs, rotation=20, ha="right", fontsize=9)
    ax1.set_title("Desempenho: Tempo de Inferencia e FPS por Biblioteca")

    # Legenda manual de hatches
    from matplotlib.patches import Patch
    patches = [
        Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="black", label=lib)
        for lib, s in LIB_STYLES.items()
    ]
    legend_extra = [
        Patch(facecolor="#cccccc", edgecolor="black", label="Barra esquerda = Tempo (ms)"),
        Patch(facecolor="white",   edgecolor="black", label="Barra direita  = FPS"),
    ]
    ax1.legend(handles=patches + legend_extra, fontsize=8, loc="upper left")

    _save(fig, "performance_combined.png")


# ---------------------------------------------------------------------------
# 2. Memoria
# ---------------------------------------------------------------------------

def plot_memory(df: pd.DataFrame):
    df = df.dropna(subset=["Memoria_pico_MB"])
    if df.empty:
        print("  [AVISO] Sem dados de memoria. Pulando grafico.")
        return

    libs = df["Biblioteca"].tolist()
    x = np.arange(len(libs))
    fig, ax = plt.subplots(figsize=(7, 4))

    for i, lib in enumerate(libs):
        style = LIB_STYLES.get(lib, {"hatch": "", "color": "#aaaaaa", "edgecolor": "black"})
        val = df.loc[df["Biblioteca"] == lib, "Memoria_pico_MB"].values[0]
        bar = ax.bar(x[i], val, color=style["color"], hatch=style["hatch"],
                     edgecolor=style["edgecolor"])
        ax.bar_label(bar, fmt="%.1f MB", fontsize=8)

    ax.set_ylabel("Pico de memoria (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(libs, rotation=20, ha="right", fontsize=9)
    ax.set_title("Uso de Memoria por Biblioteca")

    from matplotlib.patches import Patch
    patches = [
        Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="black", label=lib)
        for lib, s in LIB_STYLES.items()
    ]
    ax.legend(handles=patches, fontsize=8)
    _save(fig, "memory_usage.png")


# ---------------------------------------------------------------------------
# 3. Curva ROC
# ---------------------------------------------------------------------------

def plot_roc(dataset_name: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    plotted = 0

    for lib_name, lib_key in LIB_KEYS.items():
        roc_csv = SCORES_DIR / f"{dataset_name.lower()}_{lib_key}_roc.csv"
        if not roc_csv.exists():
            continue
        df = pd.read_csv(roc_csv)
        style = LIB_STYLES[lib_name]
        ax.plot(df["fpr"], df["tpr"],
                linestyle=style["linestyle"],
                color="black",
                linewidth=1.5,
                label=lib_name)
        plotted += 1

    if plotted == 0:
        print(f"  [AVISO] Nenhum ROC CSV encontrado para dataset '{dataset_name}'. Pulando.")
        plt.close(fig)
        return

    # Linha de referencia (classificador aleatorio)
    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1, label="Aleatorio (EER ref.)")

    ax.set_xlabel("FAR (False Acceptance Rate)")
    ax.set_ylabel("1 - FRR (True Positive Rate)")
    ax.set_title(f"Curva ROC Comparativa - {dataset_name}")
    ax.legend(fontsize=8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, linestyle=":", linewidth=0.5, color="gray")

    filename = f"roc_{dataset_name.lower().replace(' ', '_')}.png"
    _save(fig, filename)


# ---------------------------------------------------------------------------
# 4. EER comparativo
# ---------------------------------------------------------------------------

def plot_eer(df_acc: pd.DataFrame):
    datasets = df_acc["Dataset"].unique()
    libs = list(LIB_STYLES.keys())
    n_libs = len(libs)
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, lib in enumerate(libs):
        style = LIB_STYLES[lib]
        vals = []
        for ds in datasets:
            row = df_acc[(df_acc["Dataset"] == ds) & (df_acc["Biblioteca"] == lib)]
            vals.append(row["EER"].values[0] if not row.empty else 0.0)

        offset = (i - n_libs / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width,
                      color=style["color"], hatch=style["hatch"],
                      edgecolor=style["edgecolor"], label=lib)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

    ax.set_ylabel("Equal Error Rate (EER)")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_title("EER por Biblioteca e Dataset (menor e melhor)")
    ax.set_ylim(0, min(1.0, df_acc["EER"].max() * 1.3 + 0.05))
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="gray")

    _save(fig, "eer_comparison.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Gerando graficos...\n")
    generated = []

    # Performance + Memoria
    if config.PERF_CSV.exists():
        df_perf = pd.read_csv(config.PERF_CSV)
        df_perf = df_perf[df_perf["Tempo_medio_ms"].notna()].copy()
        df_perf["Tempo_medio_ms"] = pd.to_numeric(df_perf["Tempo_medio_ms"], errors="coerce")
        df_perf["FPS"] = pd.to_numeric(df_perf["FPS"], errors="coerce")
        df_perf = df_perf.dropna(subset=["Tempo_medio_ms"])

        print("[Performance]")
        plot_performance(df_perf)

        print("[Memoria]")
        plot_memory(df_perf)
    else:
        print(f"[AVISO] {config.PERF_CSV} nao encontrado. Execute tests/performance_tests.py primeiro.")

    # EER + ROC (por dataset)
    if config.ACC_CSV.exists():
        df_acc = pd.read_csv(config.ACC_CSV)

        print("[EER comparativo]")
        plot_eer(df_acc)

        for dataset in df_acc["Dataset"].unique():
            print(f"[ROC - {dataset}]")
            plot_roc(dataset)
    else:
        print(f"[AVISO] {config.ACC_CSV} nao encontrado. Execute tests/accuracy_tests.py primeiro.")

    print("\nConcluido. Graficos salvos em:", config.CHARTS_DIR)


if __name__ == "__main__":
    main()
