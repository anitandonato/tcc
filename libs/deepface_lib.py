import cv2
from deepface import DeepFace
import numpy as np

class DeepFaceLib:
    def __init__(self,
                 model_name='VGG-Face',
                 detector_backend='retinaface',
                 distance_metric='cosine'):
        """
        Inicializa o wrapper da biblioteca DeepFace.
        """
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        
        try:
            DeepFace.build_model(self.model_name)
            print(f"Wrapper DeepFace (Model: {self.model_name}, Detector: {self.detector_backend}) inicializado.")
        except Exception as e:
            print(f"Erro ao inicializar modelo DeepFace ({self.model_name}): {e}")

    def detect(self, image):
        """
        Detecta faces usando o backend especificado no DeepFace.
        Usa a nova função 'extract_faces' em vez da 'detectFace' obsoleta.
        """
        try:
            # extract_faces retorna uma LISTA de dicionários
            # Cada dicionário tem uma chave 'facial_area' com (x, y, w, h)
            face_objs = DeepFace.extract_faces(
                img_path=image,
                detector_backend=self.detector_backend,
                enforce_detection=False # Não falha se não achar rosto
            )
            
            boxes = []
            for face_obj in face_objs:
                area = face_obj['facial_area']   # dict: {'x':…,'y':…,'w':…,'h':…}
                x, y, w, h = area['x'], area['y'], area['w'], area['h']
                if w > 0 and h > 0:
                    boxes.append((x, y, w, h))

            return boxes

        except Exception as e:
            print(f"DeepFace detect error: {e}")
            return []

    def get_embedding(self, image):
        """
        Extrai o embedding facial (agora usando extract_faces).
        """
        try:
            # A função represent() é a forma correta de pegar embeddings
            embedding_obj = DeepFace.represent(
                img_path=image, 
                model_name=self.model_name,
                detector_backend=self.detector_backend,
                enforce_detection=False
            )
            
            if len(embedding_obj) > 0:
                return np.array(embedding_obj[0]["embedding"])
            else:
                return None
        except Exception as e:
            print(f"DeepFace get_embedding error: {e}")
            return None

    def compare(self, embedding1, embedding2):
        """
        Compara dois embeddings usando a métrica de distância configurada.
        """
        if embedding1 is None or embedding2 is None:
            return None

        a = np.asarray(embedding1, dtype=float)
        b = np.asarray(embedding2, dtype=float)

        if self.distance_metric == 'cosine':
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return None
            return float(1.0 - np.dot(a, b) / (norm_a * norm_b))
        else:
            return float(np.linalg.norm(a - b))