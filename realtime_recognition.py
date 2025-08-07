import cv2
import numpy as np
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

# Inisialisasi
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
with open('embeddings.pkl', 'rb') as f:
    known_embeddings = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_face_embedding(image):
    face = mtcnn(image)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        emb = resnet(face).detach().cpu().numpy()[0]
        return emb
    return None

# Buka kamera
cap = cv2.VideoCapture(0)
print("🎥 Kamera dinyalakan... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    embedding = get_face_embedding(pil_img)

    label = "Wajah tidak dikenali"
    if embedding is not None:
        best_match = None
        max_sim = 0.0
        threshold = 0.75

        for name, emb in known_embeddings.items():
            sim = cosine_similarity(embedding, emb)
            if sim > max_sim:
                max_sim = sim
                best_match = name

        if max_sim > threshold:
            label = f"{best_match} ({max_sim:.2f})"
        else:
            label = f"Tidak dikenali ({max_sim:.2f})"

    # Tampilkan hasil di frame
    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Face Recognition - Kamera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
