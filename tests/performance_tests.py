"""
Benchmark de desempenho das 3 bibliotecas ativas.

Estrategia de medicao por biblioteca:
  1. Warm-up  : 1 execucao de detect() descartada
  2. Memoria  : 1 execucao com MemoryTracker -> peak_mb
  3. Tempo    : NUM_RUNS execucoes com perf_counter -> avg_ms, fps

Saida: results/performance_summary.csv e results/raw_logs/{lib}_log.txt
"""

import os
import sys
import time
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


def _deepface_device() -> str:
    try:
        import tensorflow as tf
        return "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
    except Exception:
        return "N/A"


def _build_wrappers() -> dict:
    return {
        "OpenCV (Haar)": (OpenCVLib(), "CPU"),
        "Dlib (HOG)": (DlibLib(), "CPU"),
        "DeepFace (VGG+RetinaFace)": (
            DeepFaceLib(model_name="VGG-Face", detector_backend="retinaface"),
            _deepface_device(),
        ),
    }


def run_performance_test(image_path=None, num_runs: int = None):
    image_path = Path(image_path) if image_path else config.TEST_IMAGE
    num_runs = num_runs if num_runs is not None else config.NUM_RUNS

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[ERRO] Imagem nao encontrada: {image_path}")
        print("Adicione uma imagem em data/test_images/test3.jpg")
        return

    print(f"Imagem: {image_path}  {image.shape}")
    print(f"Execucoes: {config.WARMUP_RUNS} warm-up + {num_runs} medicoes\n")

    wrappers = _build_wrappers()
    results = []

    for name, (wrapper, device) in wrappers.items():
        lib_key = name.split()[0].lower()
        print(f"[{name}]  device={device}")

        with get_logger(lib_key) as log:
            log.info(f"Benchmark iniciado | device={device} | num_runs={num_runs}")

            # 1. Warm-up
            try:
                wrapper.detect(image.copy())
                log.info("Warm-up concluido")
            except Exception as e:
                log.error(f"Warm-up falhou: {e}")
                print(f"  Warm-up ERRO: {e}")

            # 2. Memoria (1 run isolado)
            try:
                with MemoryTracker() as mem:
                    wrapper.detect(image.copy())
                peak_mb = round(mem.peak_mb, 2)
                log.info(f"Memoria pico: {peak_mb} MB")
            except Exception as e:
                log.error(f"Medicao de memoria falhou: {e}")
                peak_mb = None

            # 3. Tempo (NUM_RUNS runs)
            total_time = 0.0
            faces_found = 0
            errors = 0

            for i in range(num_runs):
                try:
                    start = time.perf_counter()
                    boxes = wrapper.detect(image.copy())
                    end = time.perf_counter()
                    total_time += (end - start)
                    if i == num_runs - 1:
                        faces_found = len(boxes)
                except Exception as e:
                    log.error(f"Run {i+1} falhou: {e}")
                    errors += 1

            successful = num_runs - errors
            if successful > 0:
                avg_ms = round((total_time / successful) * 1000, 2)
                fps = round(1000 / avg_ms, 2) if avg_ms > 0 else 0.0
            else:
                avg_ms = fps = None

            print(f"  Tempo medio : {avg_ms} ms")
            print(f"  FPS         : {fps}")
            print(f"  Memoria     : {peak_mb} MB")
            print(f"  Faces       : {faces_found}")
            print(f"  Erros       : {errors}/{num_runs}\n")

            log.log_summary({
                "Tempo_medio_ms": avg_ms,
                "FPS": fps,
                "Memoria_pico_MB": peak_mb,
                "Faces_encontradas": faces_found,
                "Device": device,
                "Erros": errors,
            })

        results.append({
            "Biblioteca": name,
            "Tempo_medio_ms": avg_ms,
            "FPS": fps,
            "Memoria_pico_MB": peak_mb,
            "Device": device,
            "Faces_encontradas": faces_found,
            "Erros": errors,
        })

    # Salvar CSV
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(config.PERF_CSV, index=False)
    print(f"Resultados salvos em: {config.PERF_CSV}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_performance_test()
