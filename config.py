from pathlib import Path

ROOT = Path(__file__).parent

# --- Caminhos de dados ---
DATA_DIR     = ROOT / "data"
LFW_DIR      = DATA_DIR / "lfw" / "lfw-funneled"
OWN_BASE_DIR = DATA_DIR / "known_faces"
PAIRS_DIR    = DATA_DIR / "test_pairs"
MODELS_DIR   = ROOT / "models"

DLIB_PREDICTOR = MODELS_DIR / "shape_predictor_68_face_landmarks.dat"
DLIB_FACE_REC  = MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"

TEST_IMAGE = DATA_DIR / "test_images" / "test3.jpg"

# --- Caminhos de saída ---
RESULTS_DIR = ROOT / "results"
LOGS_DIR    = RESULTS_DIR / "raw_logs"
CHARTS_DIR  = RESULTS_DIR / "charts"
PERF_CSV    = RESULTS_DIR / "performance_summary.csv"
ACC_CSV     = RESULTS_DIR / "accuracy_summary.csv"

# --- Bibliotecas ativas ---
ACTIVE_LIBS = ["OpenCV", "Dlib", "DeepFace"]

# --- Parâmetros de benchmark ---
NUM_RUNS     = 50
WARMUP_RUNS  = 1

# --- Parâmetros de dataset ---
RANDOM_SEED           = 42
LFW_PEOPLE            = 200
LFW_IMAGES_PER_PERSON = 5
OWN_PEOPLE            = 10
OWN_IMAGES_PER_PERSON = 20

# --- Hardware ---
USE_GPU = True  # DeepFace usará CUDA se disponível
