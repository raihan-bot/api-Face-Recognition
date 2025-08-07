import os
import torch
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# Inisialisasi MTCNN dan FaceNet
mtcnn = MTCNN(image_size=160, margin=20)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

dataset_path = 'dataset'
embedding_dict = {}

for person in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person)
    if not os.path.isdir(person_dir):
        continue

    embeddings = []

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        face = mtcnn(img)

        if face is not None:
            emb = resnet(face.unsqueeze(0)).detach().numpy()  # shape (1, 512)
            embeddings.append(emb[0])  # ambil shape (512,)

    if embeddings:
        stacked = np.vstack(embeddings)  # shape (n, 512)
        mean_emb = np.mean(stacked, axis=0)  # shape (512,)
        embedding_dict[person] = mean_emb

# Simpan ke file pickle
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embedding_dict, f)

print("✅ Embedding wajah berhasil disimpan ke embeddings.pkl")
