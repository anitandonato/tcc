import cv2
import os
import numpy as np

# Tamanho padrao da face recortada para o HOG descriptor
_FACE_SIZE = (64, 64)
# HOGDescriptor configurado para janela 64x64
_HOG = cv2.HOGDescriptor(_FACE_SIZE, (16, 16), (8, 8), (8, 8), 9)


class OpenCVLib:
    def __init__(self):
        haar_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        self.detector = cv2.CascadeClassifier(haar_path)
        print("Wrapper OpenCV (Haar Cascade + HOG) inicializado.")

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
        Extrai HOG descriptor da maior face detectada.

        Fluxo: Haar Cascade -> recorte -> 64x64 -> grayscale -> HOG (~1764D).
        Adequado para dispositivos embarcados: sem rede neural, sem GPU.

        Returns:
            np.ndarray de ~1764 floats, ou None se nenhuma face for detectada.
        """
        boxes = self.detect(image)
        if not boxes:
            return None

        x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
        face = image[y:y + h, x:x + w]
        if face.size == 0:
            return None

        face = cv2.resize(face, _FACE_SIZE)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        descriptor = _HOG.compute(gray)
        return descriptor.flatten()

    def compare(self, embedding1, embedding2):
        """
        Distancia cosseno entre dois HOG descriptors.

        Range: [0, 2]  (0 = identico, 2 = oposto).
        Usa a mesma metrica do DeepFace (cosine) para comparabilidade.

        Returns:
            float em [0, 2], ou None se qualquer embedding for None.
        """
        if embedding1 is None or embedding2 is None:
            return None
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(1.0 - cosine_sim)