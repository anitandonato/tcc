import cv2
import os

class OpenCVLib:
    def __init__(self):
        # Encontra o caminho para os classificadores Haar do OpenCV
        # (Isso evita erros de caminho absoluto)
        haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.detector = cv2.CascadeClassifier(haar_path)
        print("Wrapper OpenCV (Haar Cascade) inicializado.")

    def detect(self, image):
        """
        Detecta faces usando o Haar Cascade.
        
        Args:
            image (numpy.ndarray): Imagem carregada pelo OpenCV (em formato BGR).

        Returns:
            list: Uma lista de tuplas, onde cada tupla é (x, y, w, h)
                  representando a bounding box de uma face detectada.
        """
        # O detector Haar funciona melhor em escala de cinza
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # O retorno do detector já é uma lista de (x, y, w, h)
        faces = self.detector.detectMultiScale(
            gray_image, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Formato (x, y, w, h) já está correto.
        return list(faces)

    def get_embedding(self, image):
        """
        OpenCV Haar/LBP não gera embeddings faciais. 
        Retorna None como placeholder.
        """
        print("Aviso: OpenCV Haar Cascade não é um modelo de reconhecimento (não gera embeddings).")
        return None

    def compare(self, embedding1, embedding2):
        """
        Não aplicável para este detector.
        """
        return None