# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## MODO DE TRABALHO OBRIGATÓRIO

**Antes de qualquer implementação, SEMPRE entre em plan mode (`EnterPlanMode`) para apresentar e validar a abordagem. Só implemente após aprovação explícita.**

---

## Visão Geral do Projeto

Estudo comparativo de **três** bibliotecas biométricas open-source de reconhecimento facial no modelo de **verificação 1:1**, avaliando acurácia (EER, FAR, FRR, ROC), desempenho (tempo de processamento, FPS) e uso de memória.

**Bibliotecas e perfis de uso (justificativa acadêmica):**

| Biblioteca | Perfil de Aplicação | Wrapper |
|-----------|---------------------|---------|
| **OpenCV** (Haar Cascade + LBPH) | Dispositivos simples, sistemas embarcados | `libs/opencv_lib.py` |
| **Dlib** (HOG + ResNet) | Aplicações desktop com hardware moderado | `libs/dlib_lib.py` |
| **DeepFace** (VGG-Face / ArcFace) | Servidores e sistemas corporativos | `libs/deepface_lib.py` |

> MTCNN, FaceRecognition e InsightFace foram descontinuados do escopo. Manter os arquivos em `libs/`, mas não usá-los nos pipelines.

---

## Ambiente de Execução

```
Dispositivo : ad-v15
Processador : 13th Gen Intel Core i5-13420H (2.10 GHz)
RAM         : 16 GB (utilizável: 15,7 GB)
GPU         : NVIDIA GeForce RTX 4050 Laptop (6 GB VRAM)
SO          : Windows 11 Pro 64-bit
Python      : 3.10

py -3.10 -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

O DeepFace deve usar a GPU (CUDA) quando disponível. Os logs devem registrar explicitamente se cada execução usou CPU ou GPU.

---

## Comandos Principais

```bash
# Prepara o dataset LFW (executar uma vez após baixar do Kaggle)
python data/prepare_lfw.py

# Prepara os pares da base própria controlada
python data/prepare_own_base.py

# Pipeline completo de verificação 1:1
python tests/accuracy_tests.py

# Benchmark de desempenho (1 warm-up + 50 execuções medidas)
python tests/performance_tests.py

# Gera todos os gráficos em results/charts/
python utils/plotter.py

# Smoke test rápido
python main.py
```

---

## Estratégia de Dataset

### LFW — Quantidade (validade estatística)

- **Fonte:** https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- **Subconjunto usado:** 200 indivíduos com exatamente 5 imagens cada = 1.000 imagens
  - Filtrar pessoas com 5+ imagens (~547 disponíveis no LFW), selecionar 200 aleatoriamente com seed fixo
- **Pares gerados:**
  - Genuínos: C(5,2) × 200 = **2.000 pares** (mesma pessoa, fotos diferentes)
  - Impostores: ~2.000 pares amostrados (pessoas diferentes)
  - Total: ~4.000 pares balanceados
- Seed aleatória fixa para reprodutibilidade

### Base Própria Controlada — Qualidade (análise em ambiente controlado)

- **10 indivíduos × 20 imagens = 200 imagens**
- 20 imagens cobrem: 3 iluminações × 2 ângulos × 2 distâncias = 12 combinações (~1–2 por combinação)
- **Pares gerados:**
  - Genuínos: C(20,2) × 10 = **1.900 pares**
  - Impostores: ~1.900 pares amostrados
  - Total: ~3.800 pares balanceados
- Finalidade: demonstrar comportamento das bibliotecas sob condições controladas (análise qualitativa complementar)
- **Limitação a declarar no TCC:** base de 10 indivíduos serve como análise qualitativa; a validade estatística principal vem do LFW

### Estrutura de Dados

```
data/
  lfw/                            # dataset Kaggle descompactado
    lfw_funneled/
      Aaron_Eckhart/
      ...
    pairs.txt                     # pares oficiais LFW (referência)
  known_faces/                    # base própria controlada
    pessoa_01/
      img_01.jpg ... img_20.jpg
    ...
    pessoa_10/
  test_pairs/                     # gerado pelos scripts prepare_*.py
    lfw_pairs_genuine.csv         # img1_path, img2_path, label=1
    lfw_pairs_impostor.csv        # img1_path, img2_path, label=0
    own_pairs_genuine.csv
    own_pairs_impostor.csv
  prepare_lfw.py                  # script de preparação LFW
  prepare_own_base.py             # script de preparação base própria
```

---

## Módulos Pendentes de Implementação

### 1. `config.py` — **VAZIO, implementar primeiro**
Centralizar todos os parâmetros:
- Caminhos: `LFW_DIR`, `OWN_BASE_DIR`, `PAIRS_DIR`, `RESULTS_DIR`
- `ACTIVE_LIBS = ["OpenCV", "Dlib", "DeepFace"]`
- `NUM_RUNS = 50`, `WARMUP_RUNS = 1`, `RANDOM_SEED = 42`
- `LFW_PEOPLE = 200`, `LFW_IMAGES_PER_PERSON = 5`
- `OWN_PEOPLE = 10`, `OWN_IMAGES_PER_PERSON = 20`
- `USE_GPU = True`

### 2. `utils/metrics.py` — **VAZIO, implementar**
```python
compute_far_frr(scores, labels, threshold) -> (far, frr)
compute_eer(scores, labels)               -> (eer, threshold_eer)
compute_roc(scores, labels)               -> (fpr[], tpr[], thresholds[])
compute_confusion_matrix(scores, labels, threshold) -> dict(TP, TN, FP, FN)
peak_memory_mb()                          -> float   # via tracemalloc
```

### 3. `tests/accuracy_tests.py` — **VAZIO, implementar**
Pipeline 1:1:
1. Carregar pares dos CSVs (LFW e base própria separadamente)
2. Para cada par: `get_embedding(img)` nos 3 wrappers
3. Calcular distância por wrapper (euclidiana ou cosseno, conforme a lib)
4. Coletar scores + labels
5. Chamar `utils/metrics.py` → FAR, FRR, EER, ROC
6. Salvar `results/accuracy_summary.csv` e gráficos ROC

### 4. `utils/logger.py` — **VAZIO, implementar**
Logger com timestamp gravando em `results/raw_logs/{lib}_log.txt`.

### 5. `utils/plotter.py` — **EXISTE, estender**
Adicionar:
- Curva ROC comparativa das 3 bibliotecas no mesmo gráfico
- Gráfico EER comparativo
- Gráfico de uso de memória (MB)
- **Todos os gráficos em preto e branco** (linhas tracejadas/pontilhadas para diferenciar)

### 6. `tests/performance_tests.py` — **EXISTE, refatorar**
- Remover MTCNN, FaceRecognition e InsightFace
- Manter OpenCV, Dlib e DeepFace
- 1 execução de warm-up descartada + 50 execuções medidas
- Adicionar medição de **pico de memória** via `tracemalloc`
- Registrar CPU ou GPU no resultado do DeepFace

### 7. `main.py` — **EXISTE, corrigir**
- Substituir caminho absoluto hardcoded por `config.py`
- Remover bibliotecas descontinuadas

---

## Interface dos Wrappers (não alterar)

```python
def detect(image)           -> list[tuple[x, y, w, h]]
def get_embedding(image)    -> np.ndarray | None
def compare(emb1, emb2)     -> float  # distância (menor = mais similar)
```

Wrappers **ativos:** `opencv_lib.py`, `dlib_lib.py`, `deepface_lib.py`
Wrappers **inativos** (manter arquivos, excluir dos pipelines): `mtcnn_lib.py`, `face_recognition_lib.py`, `insightface_lib.py`

---

## Saídas Esperadas

```
results/
  accuracy_summary.csv          # EER, FAR, FRR por biblioteca × dataset (LFW e base própria)
  performance_summary.csv       # tempo médio (ms), FPS, memória (MB), CPU/GPU
  charts/
    roc_comparativa.png         # preto e branco, 3 bibliotecas
    eer_comparison.png          # preto e branco
    performance_combined.png    # preto e branco
    memory_usage.png            # preto e branco
  raw_logs/
    opencv_log.txt
    dlib_log.txt
    deepface_log.txt
```

---

## Regras de Desenvolvimento

- Confirmar ambiente: `python -c "import cv2, dlib, deepface"`
- Toda função nova em `utils/` deve ter smoke test antes de integrar ao pipeline
- Benchmark: 1 warm-up descartado + 50 execuções medidas (justificativa: TCL, n≥30)
- Gráficos sempre em **preto e branco** (compatível com impressão ABNT)
- Somente imagens — sem vídeo
- Não modificar a interface `detect / get_embedding / compare` dos wrappers
- Registrar versão de cada biblioteca nos logs
- Seed aleatória fixada em `RANDOM_SEED = 42` para reprodutibilidade dos pares
