import cv2
import time
import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libs.opencv_lib import OpenCVLib
from libs.face_recognition_lib import FaceRecognitionLib
from libs.dlib_lib import DlibLib
from libs.mtcnn_lib import MTCNNLib
from libs.deepface_lib import DeepFaceLib
from libs.insightface_lib import InsightFaceLib
from utils.timer import Timer

def run_performance_test(image_path, num_runs=50):
    try:
        image_path_abs = os.path.abspath(image_path)
        image = cv2.imread(image_path_abs)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem em: {image_path_abs}")
            return
    except Exception as e:
        print(f"Erro ao ler imagem: {e}")
        return

    print(f"Imagem de teste '{image_path_abs}' carregada. Dimensões: {image.shape}")
    print(f"Iniciando teste de performance com {num_runs} execuções por biblioteca...\n")

    # --- Inicialização ---
    init_timer = Timer()
    
    init_timer.checkpoint("Iniciando OpenCV")
    wrappers = {"OpenCV (Haar)": OpenCVLib()}
    
    init_timer.checkpoint("Iniciando FaceRecognition")
    wrappers["FaceRecognition (HOG)"] = FaceRecognitionLib(model='hog')
    
    init_timer.checkpoint("Iniciando Dlib")
    wrappers["Dlib (HOG)"] = DlibLib()
    
    init_timer.checkpoint("Iniciando MTCNN")
    wrappers["MTCNN"] = MTCNNLib()
    
    init_timer.checkpoint("Iniciando DeepFace")
    wrappers["DeepFace (Dlib)"] = DeepFaceLib(detector_backend='dlib')
    
    init_timer.checkpoint("Iniciando InsightFace")
    wrappers["InsightFace (Buffalo_L)"] = InsightFaceLib()
    
    init_timer.checkpoint("Fim das Inicializações")
    
    print("\n--- Iniciando Teste de Inferência ---")

    results = []
    
    for name, wrapper in wrappers.items():
        print(f"Testando: {name}")
        total_time = 0
        faces_found = 0 # Variável para guardar o nº de faces
        
        try:
            # Executa N vezes
            for i in range(num_runs):
                start = time.perf_counter()
                bounding_boxes = wrapper.detect(image.copy())
                end = time.perf_counter()
                total_time += (end - start)
                
                # Na última execução, guardamos quantas faces achou
                if i == num_runs - 1:
                    faces_found = len(bounding_boxes)
            
            avg_time_s = total_time / num_runs
            avg_time_ms = avg_time_s * 1000
            fps = 1.0 / avg_time_s if avg_time_s > 0 else 0
            
            print(f"  -> Tempo Médio: {avg_time_ms:.2f} ms")
            print(f"  -> FPS: {fps:.2f}")
            print(f"  -> Faces Encontradas: {faces_found}\n") # Mostra no log
            
            results.append({
                "Biblioteca": name,
                "Tempo Medio (ms)": avg_time_ms,
                "FPS": fps,
                "Faces Encontradas": faces_found # <--- NOVA COLUNA
            })
            
        except Exception as e:
            print(f"  -> ERRO AO TESTAR {name}: {e}")
            results.append({
                "Biblioteca": name,
                "Tempo Medio (ms)": "ERRO",
                "FPS": "ERRO",
                "Faces Encontradas": 0
            })

    # --- Salvar Resultados ---
    df = pd.DataFrame(results)
    save_path = "results/performance_summary.csv"
    os.makedirs("results", exist_ok=True)
    df.to_csv(save_path, index=False)
    
    print(f"\nTeste concluído! Resultados salvos em '{save_path}'")
    print(df.to_string())

if __name__ == "__main__":
    test_image_file = "data/test_images/test3.jpg"
    if os.path.exists(test_image_file):
        run_performance_test(test_image_file, num_runs=50)
    else:
        print("Imagem não encontrada.")