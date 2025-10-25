import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np

class InsightFaceLib:
    def __init__(self, model_name='buffalo_l'):
        """
        Inicializa o wrapper da biblioteca InsightFace.
        """
        try:
            self.app = FaceAnalysis(name=model_name, 
                                    providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            print(f"Wrapper InsightFace (Model: {model_name}, Provider: CPU) inicializado.")
        except Exception as e:
            print(f"Erro ao inicializar InsightFace: {e}")
            self.app = None

    def detect(self, image):
        """
        Detecta faces usando o detector do InsightFace.
        """
        if self.app is None:
            return []
        try:
            faces = self.app.get(image)
            boxes = []
            for face in faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                boxes.append((x1, y1, x2 - x1, y2 - y1))
            return boxes
        except Exception as e:
            return []

    def get_embedding(self, image):
        """
        Extrai o embedding facial (normalmente 512-d).
        """
        if self.app is None:
            return None
        try:
            faces = self.app.get(image)
            if len(faces) > 0:
                return faces[0].embedding
            else:
                return None
        except Exception as e:
            return None

    def compare(self, embedding1, embedding2):
        """
        Compara dois embeddings (Vetores de 512-d).
        InsightFace usa distância de Cosseno.
        """
        if embedding1 is None or embedding2 is None:
            return None
            
        norm_emb1 = embedding1 / np.linalg.norm(embedding1)
        norm_emb2 = embedding2 / np.linalg.norm(embedding2)
        similarity = np.dot(norm_emb1, norm_emb2)
        distance = 1 - similarity # Distância de Cosseno
        return distance