import cv2
import time
from libs.opencv_lib import OpenCVLib
from libs.face_recognition_lib import FaceRecognitionLib
from libs.dlib_lib import DlibLib
from libs.mtcnn_lib import MTCNNLib

def main():
    print("Iniciando o TCC de Comparação de Reconhecimento Facial...")

    # --- DEFINA A VARIÁVEL AQUI (NO TOPO) ---
    image_path = 'data/test_images/test1.jpg'
    
    # --- Carregue uma imagem de teste ---
    try:
        image = cv2.imread(image_path)
        if image is None:
            # Agora 'image_path' está definida e pode ser usada no erro
            print(f"Erro: Não foi possível carregar a imagem em: {image_path}")
            print("Por favor, adicione uma imagem de teste nesta pasta.")
            return
    except Exception as e:
        print(f"Erro ao ler imagem: {e}")
        return

    # --- ESTA LINHA (A SUA LINHA 9) AGORA VAI FUNCIONAR ---
    print(f"Imagem de teste '{image_path}' carregada com sucesso.\n")
    
    # --- Inicialize os wrappers ---
    opencv_wrapper = OpenCVLib()
    face_rec_wrapper = FaceRecognitionLib(model='hog')
    dlib_wrapper = DlibLib()
    mtcnn_wrapper = MTCNNLib()
    
    wrappers = {
        "OpenCV (Haar)": opencv_wrapper,
        "FaceRecognition (Dlib-HOG)": face_rec_wrapper,
        "Dlib (HOG)": dlib_wrapper,
        "MTCNN": mtcnn_wrapper
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