"""
Benchmark de desempenho das 3 bibliotecas ativas.

Estrategia: roda detect() uma vez por imagem em cada dataset.
  - Base propria : todas as imagens em data/known_faces/
  - LFW          : LFW_SAMPLE imagens amostradas com seed=42

Metricas calculadas por biblioteca x dataset:
  - Tempo medio (ms) e desvio padrao (ms)
  - FPS estimado
  - Pico de memoria (MB)
  - Taxa de deteccao (faces detectadas / imagens processadas)

Saida: results/performance_summary.csv
"""

import os
import sys
import time
import random
import statistics
import cv2
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import config
from libs.opencv_lib import OpenCVLib
from libs.dlib_lib import DlibLib
from libs.deepface_lib import DeepFaceLib
from utils.metrics import MemoryTracker
from utils.logger import get_logger

LFW_SAMPLE = 300


def _deepface_device() -> str:
    try:
        import tensorflow as tf
        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    except Exception:
        return "N/A"


def _build_wrappers() -> dict:
    return {
        "OpenCV (Haar+HOG)": (OpenCVLib(), "CPU"),
        "Dlib (HOG)": (DlibLib(), "CPU"),
        "DeepFace (VGG+RetinaFace)": (
            DeepFaceLib(model_name="VGG-Face", detector_backend="retinaface"),
            _deepface_device(),
        ),
    }


def collect_images(directory: Path, max_count: int = None, seed: int = 42) -> list:
    images = sorted([
        p for p in directory.rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file()
    ])
    if max_count and len(images) > max_count:
        rng = random.Random(seed)
        images = sorted(rng.sample(images, max_count))
    return images


def _benchmark_dataset(wrapper, images: list, ds_name: str, lib_name: str,
                        device: str, log) -> dict:
    """Roda detect() em cada imagem do dataset e retorna estatisticas."""
    log.info(f"Dataset={ds_name} | n_imagens={len(images)} | device={device}")

    # Warm-up com a primeira imagem
    first = cv2.imread(str(images[0]))
    if first is not None:
        try:
            wrapper.detect(first.copy())
            log.info("Warm-up concluido")
        except Exception as e:
            log.error(f"Warm-up falhou: {e}")

    # Medicao de memoria (1 run isolado)
    peak_mb = None
    try:
        img = cv2.imread(str(images[0]))
        if img is not None:
            with MemoryTracker() as mem:
                wrapper.detect(img.copy())
            peak_mb = round(mem.peak_mb, 2)
            log.info(f"Memoria pico: {peak_mb} MB")
    except Exception as e:
        log.error(f"Medicao de memoria falhou: {e}")

    # Tempo por imagem
    times_ms = []
    faces_total = 0
    errors = 0
    n = len(images)

    for i, img_path in enumerate(images, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            errors += 1
            continue
        try:
            start = time.perf_counter()
            boxes = wrapper.detect(img)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times_ms.append(elapsed_ms)
            faces_total += len(boxes)
        except Exception as e:
            log.error(f"Imagem {i} ({img_path.name}) falhou: {e}")
            errors += 1

        if i % 50 == 0 or i == n:
            print(f"  {i}/{n}...", end="\r", flush=True)

    print()

    avg_ms = round(statistics.mean(times_ms), 2) if times_ms else None
    std_ms = round(statistics.stdev(times_ms), 2) if len(times_ms) > 1 else None
    fps = round(1000 / avg_ms, 2) if avg_ms else 0.0
    det_rate = round(faces_total / len(times_ms), 2) if times_ms else None

    log.log_summary({
        "Dataset": ds_name,
        "N_imagens": n,
        "Processadas": len(times_ms),
        "Tempo_medio_ms": avg_ms,
        "Std_ms": std_ms,
        "FPS": fps,
        "Memoria_pico_MB": peak_mb,
        "Faces_detectadas": faces_total,
        "Taxa_deteccao_media": det_rate,
        "Erros": errors,
    })

    return {
        "Dataset": ds_name,
        "Biblioteca": lib_name,
        "N_imagens": n,
        "Tempo_medio_ms": avg_ms,
        "Std_ms": std_ms,
        "FPS": fps,
        "Memoria_pico_MB": peak_mb,
        "Device": device,
        "Faces_detectadas": faces_total,
        "Taxa_deteccao_media": det_rate,
        "Erros": errors,
    }


def run_performance_test():
    own_images = collect_images(config.OWN_BASE_DIR)
    lfw_images = collect_images(config.LFW_DIR, max_count=LFW_SAMPLE,
                                seed=config.RANDOM_SEED)

    print(f"Base propria : {len(own_images)} imagens")
    print(f"LFW (amostra): {len(lfw_images)} imagens\n")

    if not own_images and not lfw_images:
        print("[ERRO] Nenhuma imagem encontrada nos datasets.")
        return

    datasets = []
    if own_images:
        datasets.append(("BasePropr", own_images))
    if lfw_images:
        datasets.append(("LFW", lfw_images))

    wrappers = _build_wrappers()
    all_results = []

    for lib_name, (wrapper, device) in wrappers.items():
        lib_key = lib_name.split()[0].lower()
        print(f"\n{'='*60}")
        print(f"Biblioteca: {lib_name}  device={device}")
        print("=" * 60)

        with get_logger(lib_key) as log:
            log.info(f"Benchmark iniciado | device={device}")
            for ds_name, images in datasets:
                print(f"\n  Dataset: {ds_name} ({len(images)} imagens)")
                row = _benchmark_dataset(wrapper, images, ds_name,
                                         lib_name, device, log)
                all_results.append(row)
                print(f"  Tempo medio : {row['Tempo_medio_ms']} ms  "
                      f"(std={row['Std_ms']} ms)")
                print(f"  FPS         : {row['FPS']}")
                print(f"  Memoria     : {row['Memoria_pico_MB']} MB")
                print(f"  Faces       : {row['Faces_detectadas']} "
                      f"({row['Taxa_deteccao_media']:.2f} face/img)")
                print(f"  Erros       : {row['Erros']}/{len(images)}")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cols = ["Dataset", "Biblioteca", "N_imagens", "Tempo_medio_ms", "Std_ms",
            "FPS", "Memoria_pico_MB", "Device", "Faces_detectadas",
            "Taxa_deteccao_media", "Erros"]
    df = pd.DataFrame(all_results)[cols]
    df.to_csv(config.PERF_CSV, index=False)
    print(f"\nResultados salvos em: {config.PERF_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_performance_test()
