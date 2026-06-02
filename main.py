import cv2
import time

import config
from libs.opencv_lib import OpenCVLib
from libs.dlib_lib import DlibLib
from libs.deepface_lib import DeepFaceLib

def main():
    print("Iniciando o TCC de Comparação de Reconhecimento Facial...")

    image_path = config.TEST_IMAGE
    
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Erro: Não foi possível carregar a imagem em: {image_path}")
            print("Por favor, adicione uma imagem de teste nesta pasta.")
            return
    except Exception as e:
        print(f"Erro ao ler imagem: {e}")
        return

    print(f"Imagem de teste '{image_path}' carregada com sucesso.\n")
    
    # --- Inicialize os wrappers ---
    wrappers = {
        "OpenCV (Haar)": OpenCVLib(),
        "Dlib (HOG)": DlibLib(),
        "DeepFace (VGG+RetinaFace)": DeepFaceLib(model_name='VGG-Face', detector_backend='retinaface'),
    }

    # --- Execute o teste de detecção ---
    print("\n--- INICIANDO TESTE DE DETECÇÃO ---")
    for name, wrapper in wrappers.items():
        print(f"Testando: {name}")
        
        # Medindo o tempo
        start_time = time.perf_counter()
        
        # Chamando o método padronizado 'detect'
        bounding_boxes = wrapper.detect(image)
        
        end_time = time.perf_counter()
        
        elapsed_ms = (end_time - start_time) * 1000
        
        print(f"  -> Faces encontradas: {len(bounding_boxes)}")
        print(f"  -> Tempo de execução: {elapsed_ms:.2f} ms")
        
        # (Opcional) Desenha as caixas na imagem para visualização
        image_copy = image.copy()
        for (x, y, w, h) in bounding_boxes:
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Salva a imagem resultante
        cv2.imwrite(f"results/{name}_detection_result.jpg", image_copy)

if __name__ == "__main__":
    main()