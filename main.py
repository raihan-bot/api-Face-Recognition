import os
import io
import json
import pickle
import numpy as np
from PIL import Image
from typing import List

from fastapi import FastAPI, UploadFile, Form, File, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# -------------------- Model Setup --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -------------------- Load Embeddings --------------------
try:
    with open('embeddings.pkl', 'rb') as f:
        known_embeddings = pickle.load(f)
except FileNotFoundError:
    known_embeddings = {}

# -------------------- Helper Functions --------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_face_embedding(image):
    face = mtcnn(image)
    if face is not None:
        face = face.unsqueeze(0).to(device)
        emb = resnet(face).detach().cpu().numpy()[0]
        return emb
    return None

# -------------------- FastAPI App --------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# -------------------- WebSocket Endpoint --------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🟢 Koneksi WebSocket diterima")

    while True:
        try:
            message = await websocket.receive_bytes()
            img = Image.open(io.BytesIO(message)).convert("RGB")
            emb = get_face_embedding(img)

            result = {"match": False}

            if emb is not None:
                best_match = None
                max_sim = 0.0
                threshold = 0.75
                for name, known_emb in known_embeddings.items():
                    sim = cosine_similarity(emb, known_emb)
                    if sim > max_sim:
                        max_sim = sim
                        best_match = name
                if max_sim > threshold:
                    result = {"match": True, "name": best_match, "score": float(max_sim)}

            await websocket.send_json(result)
        except Exception as e:
            print("❌ WebSocket error:", e)
            break

# -------------------- Single Image Enrollment --------------------
@app.post("/enroll")
async def enroll_face(file: UploadFile, name: str = Form(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    embedding = get_face_embedding(image)

    if embedding is None:
        return {"success": False, "message": "Wajah tidak terdeteksi"}

    known_embeddings[name] = embedding
    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)

    return {"success": True, "message": f"Berhasil menambahkan wajah untuk {name}"}

# -------------------- Multi Image Enrollment --------------------
@app.post("/enroll_batch")
async def enroll_batch(name: str = Form(...), files: List[UploadFile] = File(...)):
    folder_path = f"dataset/{name}"
    os.makedirs(folder_path, exist_ok=True)

    embeddings = []

    for idx, file in enumerate(files, 1):
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Simpan gambar ke folder dataset/nama/1.jpg, 2.jpg, ...
        image.save(os.path.join(folder_path, f"{idx}.jpg"))

        # Dapatkan embedding
        emb = get_face_embedding(image)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return {"success": False, "message": "Tidak ada wajah yang terdeteksi di gambar manapun."}

    # Simpan average embedding untuk orang ini
    avg_embedding = np.mean(embeddings, axis=0)
    known_embeddings[name] = avg_embedding

    with open('embeddings.pkl', 'wb') as f:
        pickle.dump(known_embeddings, f)

    return {"success": True, "message": f"Berhasil mendaftarkan wajah untuk {name} dari {len(embeddings)} gambar."}

# -------------------- Run FastAPI --------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))  # Default port 8001 jika dijalankan lokal
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
