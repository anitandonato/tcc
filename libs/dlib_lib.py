import dlib
import cv2
import numpy as np

class DlibLib:
    def __init__(self, predictor_path="models/shape_predictor_68_face_landmarks.dat", 
                   face_rec_model_path="models/dlib_face_recognition_resnet_model_v1.dat"):
        try:
            self.detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(predictor_path)
            self.face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
            print("Wrapper Dlib (HOG) inicializado.")
        except Exception as e:
            print(f"Erro ao carregar modelos do Dlib: {e}")
            print(f"Certifique-se que os arquivos .dat estão na pasta 'models/'.")
            self.detector = None

    def detect(self, image):
        if self.detector is None: return []
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = self.detector(gray_image, 1)
        boxes = []
        for d in detections:
            boxes.append((d.left(), d.top(), d.width(), d.height()))
        return boxes

    def get_embedding(self, image, known_locations=None):
        if self.detector is None: return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if not known_locations:
            boxes_xywh = self.detect(image)
            if not boxes_xywh: return None
            x, y, w, h = boxes_xywh[0]
            detection_dlib = dlib.rectangle(x, y, x + w, y + h)
        else:
            x, y, w, h = known_locations[0]
            detection_dlib = dlib.rectangle(x, y, x + w, y + h)

        shape = self.shape_predictor(rgb_image, detection_dlib)
        embedding = self.face_rec_model.compute_face_descriptor(rgb_image, shape)
        return np.array(embedding)

    def compare(self, embedding1, embedding2):
        if embedding1 is None or embedding2 is None: return None
        return np.linalg.norm(embedding1 - embedding2)