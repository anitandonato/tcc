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
    "OpenCV (Haar+HOG)":         {"hatch": "",   "color": "#cccccc", "linestyle": "-",  "edgecolor": "black"},
    "Dlib (HOG)":                {"hatch": "//", "color": "#888888", "linestyle": "--", "edgecolor": "black"},
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
# 1. Performance: tempo (ms) com desvio padrao, por dataset
# ---------------------------------------------------------------------------

def _short_lib(name: str) -> str:
    return name.split("(")[0].strip()


def plot_performance(df: pd.DataFrame):
    from matplotlib.patches import Patch

    has_dataset = "Dataset" in df.columns and df["Dataset"].nunique() > 1

    if has_dataset:
        datasets = df["Dataset"].unique()
        libs = df["Biblioteca"].unique().tolist()
        short_libs = [_short_lib(l) for l in libs]
        x = np.arange(len(libs))
        n_ds = len(datasets)
        width = 0.35
        ds_colors  = ["#cccccc", "#888888"]
        ds_hatches = ["", "//"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for j, ds in enumerate(datasets):
            df_ds = df[df["Dataset"] == ds]
            tempos, stds, fpss = [], [], []
            for lib in libs:
                row = df_ds[df_ds["Biblioteca"] == lib]
                tempos.append(float(row["Tempo_medio_ms"].values[0]) if not row.empty else 0)
                stds.append(float(row["Std_ms"].values[0]) if not row.empty and "Std_ms" in row.columns else 0)
                fpss.append(float(row["FPS"].values[0]) if not row.empty else 0)

            offset = (j - n_ds / 2 + 0.5) * width
            ax1.bar(x + offset, tempos, width, yerr=stds, capsize=4,
                    color=ds_colors[j], hatch=ds_hatches[j],
                    edgecolor="black", label=ds, error_kw={"elinewidth": 1})
            ax2.bar(x + offset, fpss, width,
                    color=ds_colors[j], hatch=ds_hatches[j],
                    edgecolor="black", label=ds)

        for ax, ylabel, title in [
            (ax1, "Tempo medio de inferencia (ms)", "Tempo de Inferencia por Biblioteca"),
            (ax2, "FPS (quadros por segundo)",       "FPS por Biblioteca"),
        ]:
            ax.set_xticks(x)
            ax.set_xticklabels(short_libs, rotation=15, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=8)
            ax.grid(axis="y", linestyle=":", linewidth=0.5, color="gray")

        ax1.set_yscale("log")
        fig.suptitle("Desempenho por Biblioteca e Dataset", fontsize=11)
    else:
        libs = df["Biblioteca"].tolist()
        x = np.arange(len(libs))
        width = 0.35
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()

        for i, lib in enumerate(libs):
            style = LIB_STYLES.get(lib, {"hatch": "", "color": "#aaaaaa", "edgecolor": "black"})
            tempo = df.loc[df["Biblioteca"] == lib, "Tempo_medio_ms"].values[0]
            fps   = df.loc[df["Biblioteca"] == lib, "FPS"].values[0]
            ax1.bar(x[i] - width / 2, tempo, width,
                    color=style["color"], hatch=style["hatch"], edgecolor=style["edgecolor"])
            ax2.bar(x[i] + width / 2, fps, width,
                    color="white", hatch=style["hatch"], edgecolor=style["edgecolor"])

        ax1.set_ylabel("Tempo medio de inferencia (ms)")
        ax2.set_ylabel("FPS (quadros por segundo)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(libs, rotation=20, ha="right", fontsize=9)
        ax1.set_title("Desempenho: Tempo de Inferencia e FPS por Biblioteca")

        patches = [
            Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="black", label=lib)
            for lib, s in LIB_STYLES.items() if lib in libs
        ]
        ax1.legend(handles=patches, fontsize=8, loc="upper left")

    _save(fig, "performance_combined.png")


# ---------------------------------------------------------------------------
# 2. Memoria
# ---------------------------------------------------------------------------

def plot_memory(df: pd.DataFrame):
    from matplotlib.patches import Patch

    df = df.dropna(subset=["Memoria_pico_MB"])
    if df.empty:
        print("  [AVISO] Sem dados de memoria. Pulando grafico.")
        return

    has_dataset = "Dataset" in df.columns and df["Dataset"].nunique() > 1

    if has_dataset:
        datasets = df["Dataset"].unique()
        libs = df["Biblioteca"].unique().tolist()
        short_libs = [_short_lib(l) for l in libs]
        x = np.arange(len(libs))
        n_ds = len(datasets)
        width = 0.35
        ds_colors  = ["#cccccc", "#888888"]
        ds_hatches = ["", "//"]

        fig, ax = plt.subplots(figsize=(8, 5))

        for j, ds in enumerate(datasets):
            df_ds = df[df["Dataset"] == ds]
            vals = []
            for lib in libs:
                row = df_ds[df_ds["Biblioteca"] == lib]
                vals.append(float(row["Memoria_pico_MB"].values[0]) if not row.empty else 0)

            offset = (j - n_ds / 2 + 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          color=ds_colors[j], hatch=ds_hatches[j],
                          edgecolor="black", label=ds)
            ax.bar_label(bars, fmt="%.0f MB", fontsize=7, padding=2)

        ax.set_ylabel("Pico de memoria (MB)")
        ax.set_xticks(x)
        ax.set_xticklabels(short_libs, rotation=15, ha="right", fontsize=9)
        ax.set_title("Uso de Memoria por Biblioteca e Dataset")
        ax.legend(fontsize=8)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, color="gray")
    else:
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

        patches = [
            Patch(facecolor=s["color"], hatch=s["hatch"], edgecolor="black", label=lib)
            for lib, s in LIB_STYLES.items() if lib in libs
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
