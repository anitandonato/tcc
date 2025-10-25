import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np

class MTCNNLib:
    def __init__(self):
        try:
            self.detector = MTCNN()
            print("Wrapper MTCNN inicializado.")
        except Exception as e:
            print(f"Erro ao inicializar MTCNN: {e}")
            self.detector = None

    def detect(self, image):
        if self.detector is None: return []
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.detect_faces(rgb_image)
        boxes = []
        for res in results:
            x, y, w, h = res['box']
            boxes.append((x, y, w, h))
        return boxes

    def get_embedding(self, image, known_locations=None):
        print("Aviso: MTCNN é um detector de faces, não gera embeddings por si só.")
        return None

    def compare(self, embedding1, embedding2):
        return None