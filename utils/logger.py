"""
Logger de experimentos para verificacao facial 1:1.

Grava um arquivo de log por biblioteca em results/raw_logs/{lib}_log.txt.

Uso basico:
    with get_logger("dlib") as log:
        log.info("Iniciando experimento")
        log.log_pair("img1.jpg", "img2.jpg", 0.42, 1, "MATCH")
        log.log_summary({"EER": 0.05, "FAR": 0.04, "FRR": 0.06})
"""

import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import config


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ExperimentLogger:

    def __init__(self, lib_name: str):
        self._lib = lib_name
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = config.LOGS_DIR / f"{lib_name}_log.txt"
        self._file = open(log_path, "w", encoding="utf-8")
        self._write_header()

    # --- context manager ---

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # --- escrita ---

    def _write(self, line: str):
        self._file.write(line + "\n")
        self._file.flush()

    def _write_header(self):
        sep = "=" * 60
        self._write(sep)
        self._write(f"Biblioteca : {self._lib}")
        self._write(f"Inicio     : {_timestamp()}")
        self._write(sep)

    def info(self, msg: str):
        self._write(f"[{_timestamp()}] INFO: {msg}")

    def error(self, msg: str):
        self._write(f"[{_timestamp()}] ERROR: {msg}")

    def log_pair(self, img1, img2, score: float, label: int, decision: str):
        """Registra resultado de um par de verificacao."""
        name1 = Path(str(img1)).name
        name2 = Path(str(img2)).name
        self._write(
            f"[{_timestamp()}] PAIR: "
            f"img1={name1}, img2={name2}, "
            f"score={score:.4f}, label={label}, decision={decision}"
        )

    def log_summary(self, metrics: dict):
        """Registra bloco de metricas ao final do experimento."""
        sep = "-" * 60
        self._write(sep)
        self._write(f"[{_timestamp()}] SUMMARY")
        for key, value in metrics.items():
            if isinstance(value, float):
                self._write(f"  {key}: {value:.4f}")
            else:
                self._write(f"  {key}: {value}")
        self._write(sep)

    def close(self):
        if not self._file.closed:
            self._write("=" * 60)
            self._write(f"Fim        : {_timestamp()}")
            self._write("=" * 60)
            self._file.close()


def get_logger(lib_name: str) -> ExperimentLogger:
    """Retorna um ExperimentLogger para a biblioteca indicada."""
    return ExperimentLogger(lib_name)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with get_logger("smoke_test") as log:
        log.info("Iniciando teste de logger")
        log.log_pair("alice_01.jpg", "alice_02.jpg", 0.312, 1, "MATCH")
        log.log_pair("alice_01.jpg", "bob_01.jpg",   0.871, 0, "NO_MATCH")
        log.error("Exemplo de erro registrado")
        log.log_summary({
            "EER": 0.05,
            "FAR": 0.04,
            "FRR": 0.06,
            "Acuracia": 0.95,
            "Tempo_medio_ms": 123.4,
            "Memoria_pico_MB": 45.2,
        })

    log_path = config.LOGS_DIR / "smoke_test_log.txt"
    assert log_path.exists(), "Arquivo de log nao foi criado"
    content = log_path.read_text(encoding="utf-8")
    assert "PAIR" in content
    assert "SUMMARY" in content
    assert "EER" in content

    print(f"Log criado em: {log_path}")
    print("Conteudo:")
    print(content)
    print("Smoke test passou.")
