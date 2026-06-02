"""
Pipeline de verificacao 1:1 para as 3 bibliotecas ativas.

Avalia LFW e base propria separadamente.
Para cada biblioteca x dataset calcula: EER, FAR, FRR, Acuracia,
Confusion Matrix e curva ROC.

Saidas:
  results/accuracy_summary.csv
  results/scores/{dataset}_{lib_key}_scores.csv
  results/raw_logs/{lib_key}_log.txt
"""

import os
import sys
import cv2
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from libs.opencv_lib import OpenCVLib
from libs.dlib_lib import DlibLib
from libs.deepface_lib import DeepFaceLib
from utils.metrics import (
    compute_eer,
    compute_far_frr,
    compute_roc,
    compute_confusion_matrix,
    compute_accuracy,
)
from utils.logger import get_logger

SCORES_DIR = config.RESULTS_DIR / "scores"


def _load_pairs(genuine_csv: Path, impostor_csv: Path) -> pd.DataFrame:
    """Carrega e concatena pares genuinos e impostores em um unico DataFrame."""
    gen = pd.read_csv(genuine_csv)
    imp = pd.read_csv(impostor_csv)
    return pd.concat([gen, imp], ignore_index=True)


def _build_wrappers() -> dict:
    return {
        "OpenCV (Haar+HOG)": ("opencv", OpenCVLib()),
        "Dlib (HOG)": ("dlib",   DlibLib()),
        "DeepFace (VGG+RetinaFace)": ("deepface", DeepFaceLib(model_name="VGG-Face", detector_backend="retinaface")),
    }


def _compute_scores(wrapper, pairs: pd.DataFrame, log) -> tuple[list, list, int]:
    """
    Gera scores para todos os pares.

    Retorna (scores, labels, n_skipped).
    Pares com embedding None sao descartados e contados em n_skipped.
    """
    scores, labels = [], []
    n_skipped = 0
    total = len(pairs)

    for i, row in pairs.iterrows():
        img1 = cv2.imread(str(row["img1_path"]))
        img2 = cv2.imread(str(row["img2_path"]))

        if img1 is None or img2 is None:
            log.error(f"Imagem nao encontrada: {row['img1_path']} ou {row['img2_path']}")
            n_skipped += 1
            continue

        try:
            emb1 = wrapper.get_embedding(img1)
            emb2 = wrapper.get_embedding(img2)
            score = wrapper.compare(emb1, emb2)
        except Exception as e:
            log.error(f"Erro no par {i}: {e}")
            n_skipped += 1
            continue

        if score is None:
            n_skipped += 1
            decision = "SKIPPED"
            log.log_pair(row["img1_path"], row["img2_path"], float("nan"), row["label"], decision)
            continue

        label = int(row["label"])
        decision = "MATCH" if score < 0.5 else "NO_MATCH"  # threshold provisorio para o log
        log.log_pair(row["img1_path"], row["img2_path"], score, label, decision)

        scores.append(score)
        labels.append(label)

        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{total} pares processados...")

    return scores, labels, n_skipped


def run_accuracy_test(dataset_name: str, genuine_csv: Path, impostor_csv: Path) -> list[dict]:
    """Executa verificacao 1:1 em um dataset e retorna lista de resultados por biblioteca."""
    if not genuine_csv.exists() or not impostor_csv.exists():
        print(f"[AVISO] CSVs do dataset '{dataset_name}' nao encontrados. Pulando.")
        print(f"  Execute: python data/prepare_lfw.py  ou  python data/prepare_own_base.py")
        return []

    pairs = _load_pairs(genuine_csv, impostor_csv)
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}  |  Pares: {len(pairs)} ({pairs['label'].sum()} genuinos)")
    print(f"{'='*60}")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    wrappers = _build_wrappers()
    dataset_results = []

    for lib_name, (lib_key, wrapper) in wrappers.items():
        print(f"\n  [{lib_name}]")

        with get_logger(f"{dataset_name.lower()}_{lib_key}") as log:
            log.info(f"Dataset={dataset_name} | Pares={len(pairs)}")

            scores, labels, n_skipped = _compute_scores(wrapper, pairs, log)
            n_valid = len(scores)

            if n_valid < 10:
                print(f"    Pares validos insuficientes ({n_valid}). Pulando metricas.")
                log.error(f"Pares validos insuficientes: {n_valid}")
                continue

            eer, thr_eer      = compute_eer(scores, labels)
            far, frr          = compute_far_frr(scores, labels, thr_eer)
            acc               = compute_accuracy(scores, labels, thr_eer)
            cm                = compute_confusion_matrix(scores, labels, thr_eer)
            fpr, tpr, thrs    = compute_roc(scores, labels)

            print(f"    EER       : {eer:.4f}")
            print(f"    FAR       : {far:.4f}")
            print(f"    FRR       : {frr:.4f}")
            print(f"    Acuracia  : {acc:.4f}")
            print(f"    Threshold : {thr_eer:.4f}")
            print(f"    Validos   : {n_valid}/{len(pairs)}  (ignorados: {n_skipped})")

            log.log_summary({
                "EER": eer, "FAR": far, "FRR": frr,
                "Acuracia": acc, "Threshold_EER": thr_eer,
                "TP": cm["TP"], "TN": cm["TN"], "FP": cm["FP"], "FN": cm["FN"],
                "Pares_validos": n_valid, "Pares_total": len(pairs),
            })

            # Scores brutos para reanalise
            scores_csv = SCORES_DIR / f"{dataset_name.lower()}_{lib_key}_scores.csv"
            pd.DataFrame({
                "img1_path": pairs["img1_path"].iloc[:n_valid + n_skipped],
                "score": scores,
                "label": labels,
            }).to_csv(scores_csv, index=False)

            # ROC para plotter
            roc_csv = SCORES_DIR / f"{dataset_name.lower()}_{lib_key}_roc.csv"
            pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thrs}).to_csv(roc_csv, index=False)

            dataset_results.append({
                "Dataset": dataset_name,
                "Biblioteca": lib_name,
                "EER": round(eer, 4),
                "FAR": round(far, 4),
                "FRR": round(frr, 4),
                "Acuracia": round(acc, 4),
                "Threshold_EER": round(thr_eer, 4),
                "TP": cm["TP"], "TN": cm["TN"], "FP": cm["FP"], "FN": cm["FN"],
                "Pares_validos": n_valid,
                "Pares_total": len(pairs),
            })

    return dataset_results


def main():
    all_results = []

    # LFW
    all_results += run_accuracy_test(
        "LFW",
        config.PAIRS_DIR / "lfw_pairs_genuine.csv",
        config.PAIRS_DIR / "lfw_pairs_impostor.csv",
    )

    # Base propria controlada
    all_results += run_accuracy_test(
        "BasePropr",
        config.PAIRS_DIR / "own_pairs_genuine.csv",
        config.PAIRS_DIR / "own_pairs_impostor.csv",
    )

    if not all_results:
        print("\nNenhum resultado gerado. Verifique se os datasets foram preparados.")
        return

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(config.ACC_CSV, index=False)

    print(f"\n{'='*60}")
    print(f"Resultados salvos em: {config.ACC_CSV}")
    print(df[["Dataset", "Biblioteca", "EER", "FAR", "FRR", "Acuracia"]].to_string(index=False))


if __name__ == "__main__":
    main()
