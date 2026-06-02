"""
Métricas de avaliação para verificação facial 1:1.

Convenção de scores: distância (menor = mais similar).
Decisão de match: score < threshold → aceito (match).

  FAR (False Acceptance Rate): impostores aceitos / total impostores
  FRR (False Rejection Rate) : genuínos rejeitados / total genuínos
  EER (Equal Error Rate)     : ponto onde FAR == FRR
"""

import tracemalloc
import numpy as np


# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------

def _validate(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if scores.shape != labels.shape:
        raise ValueError("scores e labels devem ter o mesmo tamanho.")
    if not np.any(labels == 1):
        raise ValueError("Nenhum par genuíno (label=1) encontrado.")
    if not np.any(labels == 0):
        raise ValueError("Nenhum par impostor (label=0) encontrado.")
    return scores, labels


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def compute_roc(scores, labels):
    """
    Calcula a curva ROC varrendo todos os thresholds únicos.

    Retorna:
        fpr        : np.ndarray — False Positive Rate (FAR) por threshold
        tpr        : np.ndarray — True Positive Rate (1 - FRR) por threshold
        thresholds : np.ndarray — valores de threshold correspondentes
    """
    scores, labels = _validate(scores, labels)

    # Sentinelas garantem pontos (FAR=1,TPR=1) e (FAR=0,TPR=0) na curva
    thresholds = np.concatenate(([-np.inf], np.unique(scores), [np.inf]))

    n_genuine  = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    fpr_list = []
    tpr_list = []

    for thr in thresholds:
        predicted_match = scores < thr          # True = aceito como match
        tp = np.sum(predicted_match & (labels == 1))
        fp = np.sum(predicted_match & (labels == 0))
        fpr_list.append(fp / n_impostor)        # FAR
        tpr_list.append(tp / n_genuine)         # 1 - FRR

    return np.array(fpr_list), np.array(tpr_list), thresholds


# ---------------------------------------------------------------------------
# EER
# ---------------------------------------------------------------------------

def compute_eer(scores, labels):
    """
    Calcula o Equal Error Rate (EER) e o threshold correspondente.

    EER é o ponto onde FAR == FRR, ou seja, FAR == 1 - TPR.
    Usa interpolação linear entre os dois pontos mais próximos do cruzamento.

    Retorna:
        eer           : float — valor do EER (0 a 1)
        threshold_eer : float — threshold no ponto de EER
    """
    fpr, tpr, thresholds = compute_roc(scores, labels)

    frr = 1.0 - tpr          # FRR = 1 - TPR
    diff = frr - fpr         # EER ocorre onde diff == 0

    # Encontra índice onde diff muda de sinal
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) == 0:
        # Sem cruzamento exato: retorna o ponto de mínima diferença
        idx = np.argmin(np.abs(diff))
        return float((fpr[idx] + frr[idx]) / 2), float(thresholds[idx])

    idx = sign_changes[0]

    # Interpolação linear entre idx e idx+1
    d0, d1 = diff[idx], diff[idx + 1]
    t0, t1 = thresholds[idx], thresholds[idx + 1]

    if d1 == d0:
        t_eer = (t0 + t1) / 2
    else:
        t_eer = t0 - d0 * (t1 - t0) / (d1 - d0)

    # EER é a média de FAR e FRR no ponto interpolado
    alpha = 0 if (t1 == t0) else (t_eer - t0) / (t1 - t0)
    far_eer = fpr[idx] + alpha * (fpr[idx + 1] - fpr[idx])
    frr_eer = frr[idx] + alpha * (frr[idx + 1] - frr[idx])
    eer = (far_eer + frr_eer) / 2

    return float(eer), float(t_eer)


# ---------------------------------------------------------------------------
# FAR / FRR para threshold fixo
# ---------------------------------------------------------------------------

def compute_far_frr(scores, labels, threshold):
    """
    Calcula FAR e FRR para um threshold específico.

    Retorna:
        far : float
        frr : float
    """
    scores, labels = _validate(scores, labels)

    predicted_match = scores < threshold
    n_genuine  = np.sum(labels == 1)
    n_impostor = np.sum(labels == 0)

    fp = np.sum(predicted_match & (labels == 0))
    fn = np.sum(~predicted_match & (labels == 1))

    far = fp / n_impostor
    frr = fn / n_genuine
    return float(far), float(frr)


# ---------------------------------------------------------------------------
# Matriz de confusão
# ---------------------------------------------------------------------------

def compute_confusion_matrix(scores, labels, threshold):
    """
    Retorna dict com TP, TN, FP, FN para um threshold dado.

      TP : genuíno aceito (correto)
      TN : impostor rejeitado (correto)
      FP : impostor aceito  → contribui para FAR
      FN : genuíno rejeitado → contribui para FRR
    """
    scores, labels = _validate(scores, labels)

    predicted_match = scores < threshold
    tp = int(np.sum(predicted_match  & (labels == 1)))
    tn = int(np.sum(~predicted_match & (labels == 0)))
    fp = int(np.sum(predicted_match  & (labels == 0)))
    fn = int(np.sum(~predicted_match & (labels == 1)))

    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


# ---------------------------------------------------------------------------
# Acurácia
# ---------------------------------------------------------------------------

def compute_accuracy(scores, labels, threshold):
    """Retorna acurácia geral: (TP + TN) / total."""
    cm = compute_confusion_matrix(scores, labels, threshold)
    total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
    return (cm["TP"] + cm["TN"]) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Medição de memória
# ---------------------------------------------------------------------------

class MemoryTracker:
    """
    Context manager para medir pico de memória de um bloco de código.

    Uso:
        with MemoryTracker() as mem:
            embedding = wrapper.get_embedding(image)
        print(f"Pico: {mem.peak_mb:.2f} MB")
    """

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, *_):
        _, self._peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    @property
    def peak_mb(self):
        return self._peak_bytes / 1024 / 1024


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=== Smoke test: dados quase perfeitos ===")
    genuine_scores  = rng.uniform(0.0, 0.3, 500)   # distâncias baixas
    impostor_scores = rng.uniform(0.7, 1.0, 500)   # distâncias altas
    scores = np.concatenate([genuine_scores, impostor_scores])
    labels = np.array([1] * 500 + [0] * 500)

    eer, thr = compute_eer(scores, labels)
    print(f"  EER={eer:.4f}  (esperado ~0.00)  threshold_eer={thr:.4f}")
    assert eer < 0.05, f"EER inesperadamente alto: {eer}"

    far, frr = compute_far_frr(scores, labels, thr)
    print(f"  FAR={far:.4f}  FRR={frr:.4f}  (ambos devem ser ~0)")

    cm = compute_confusion_matrix(scores, labels, thr)
    print(f"  Confusion matrix: {cm}")

    acc = compute_accuracy(scores, labels, thr)
    print(f"  Acuracia={acc:.4f}  (esperado ~1.00)")
    assert acc > 0.95, f"Acurácia inesperadamente baixa: {acc}"

    fpr, tpr, _ = compute_roc(scores, labels)
    print(f"  ROC: {len(fpr)} pontos gerados")

    print("\n=== Smoke test: MemoryTracker ===")
    with MemoryTracker() as mem:
        _ = np.zeros((1000, 1000))
    print(f"  Pico de memória: {mem.peak_mb:.2f} MB  (esperado > 0)")
    assert mem.peak_mb > 0

    print("\nTodos os testes passaram.")
