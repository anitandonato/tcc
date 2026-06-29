import sys, cv2, numpy as np
sys.path.insert(0, '.')
from libs.deepface_lib import DeepFaceLib

lib = DeepFaceLib()

img1 = cv2.imread('data/lfw/lfw-funneled/Richard_Myers/Richard_Myers_0001.jpg')
img2 = cv2.imread('data/lfw/lfw-funneled/Richard_Myers/Richard_Myers_0004.jpg')

emb1 = lib.get_embedding(img1)
emb2 = lib.get_embedding(img2)

print('emb1 shape:', emb1.shape if emb1 is not None else None)
print('emb1 non-zeros:', np.count_nonzero(emb1) if emb1 is not None else None)
print('emb2 non-zeros:', np.count_nonzero(emb2) if emb2 is not None else None)

score = lib.compare(emb1, emb2)
print('score (genuino, deve ser baixo):', score)
