import face_recognition
import cv2
import numpy as np

class FaceRecognitionLib:
    def __init__(self, model='hog'):
        """
        Inicializa o wrapper da biblioteca face_recognition.
        
        Args:
            model (str): 'hog' (mais rápido, CPU) ou 'cnn' (mais preciso, GPU/dlib).
                         Usaremos 'hog' para o TCC para comparar com dlib puro.
        """
        self.model = model
        print(f"Wrapper FaceRecognition (backend Dlib, model={self.model}) inicializado.")

    def detect(self, image):
        """
        Detecta faces usando a biblioteca face_recognition.
        
        Args:
            image (numpy.ndarray): Imagem carregada pelo OpenCV (em formato BGR).

        Returns:
            list: Uma lista de tuplas (x, y, w, h)
        """
        # A biblioteca espera imagens em RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Retorna (top, right, bottom, left)
        locations = face_recognition.face_locations(rgb_image, model=self.model)
        
        # Converte de (top, right, bottom, left) para (x, y, w, h)
        boxes = []
        for (top, right, bottom, left) in locations:
            x = left
            y = top
            w = right - left
            h = bottom - top
            boxes.append((x, y, w, h))
            
        return boxes

    def get_embedding(self, image, known_locations=None):
        """
        Extrai o embedding facial de 128-d.
        
        Args:
            image (numpy.ndarray): Imagem BGR.
            known_locations (list, opcional): Bounding boxes já detectadas.

        Returns:
            numpy.ndarray: O vetor de embedding de 128 dimensões (ou None).
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Se as localizações não foram passadas, detecta primeiro
        if not known_locations:
            # Converte (x, y, w, h) de volta para (top, right, bottom, left)
            # (Este é um exemplo simples, idealmente você usaria a detecção de 'detect')
            raw_locations = face_recognition.face_locations(rgb_image, model=self.model)
        else:
            # Converte (x, y, w, h) para (top, right, bottom, left)
            raw_locations = []
            for (x, y, w, h) in known_locations:
                raw_locations.append((y, x + w, y + h, x))

        # Extrai os embeddings
        embeddings = face_recognition.face_encodings(rgb_image, known_locations=raw_locations)
        
        if len(embeddings) > 0:
            return embeddings[0]  # Retorna o embedding da primeira face
        else:
            return None

    def compare(self, embedding1, embedding2):
        """
        Compara dois embeddings e retorna a distância Euclidiana.
        
        Args:
            embedding1 (np.ndarray): Primeiro embedding.
            embedding2 (np.ndarray): Segundo embedding.

        Returns:
            float: A distância (menor = mais similar).
        """
        if embedding1 is None or embedding2 is None:
            return None
            
        # Calcula a distância euclidiana. A biblioteca já fornece isso.
        distance = face_recognition.face_distance([embedding1], embedding2)
        return distance[0]