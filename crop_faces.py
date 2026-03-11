import cv2
import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
from utils import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(
    image_size=224,      # output size direttamente 224x224 per ResNet
    margin=20,           # un po' di contesto intorno al volto
    keep_all=False,      # tieni solo il volto più grande
    device=device
)

def extract_face(image_path, save_path):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)  # restituisce tensor normalizzato o None se non trova volti
    
    if face is not None:
        # face è un tensor (3, 224, 224), salvalo come immagine
        face_img = (face.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8')
        Image.fromarray(face_img).save(save_path)
        return True
    return False

def process_dataset():

    # REAL
    real_path = os.path.join(
        ROOT_DIR, "original_sequences", "youtube", "c23", "frames"
    )

    out_real_path = os.path.join(
        ROOT_DIR, "original_sequences", "youtube", "c23", "cropped_frames"
    )

    # FAKE
    fake_path = os.path.join(
        ROOT_DIR, "manipulated_sequences", MANIPULATION, "c23", "frames"
    )

    out_fake_path = os.path.join(
        ROOT_DIR, "manipulated_sequences", MANIPULATION, "c23", "cropped_frames"
    )

    os.makedirs(out_real_path, exist_ok=True)
    os.makedirs(out_fake_path, exist_ok=True)
    real_skipped = 0

    print("Starting cropping REAL images.......")
    count = 0

    for root, dirs, files in os.walk(real_path):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            src = os.path.join(root, fname)
            # mantieni la struttura delle cartelle
            rel_path = os.path.relpath(root, real_path)
            dst_dir = os.path.join(out_real_path, rel_path)

            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
            if os.path.exists(f"{dst}"):
                continue
            else:
                if not extract_face(src, dst):
                    real_skipped += 1

            count += 1
            print(f"{count}")

    print(f"Done. Skipped real frames (no face found): {real_skipped}")

    print("Starting cropping FAKE images.......")
    fake_skipped = 0
    count = 0

    for root, dirs, files in os.walk(fake_path):
        for fname in files:
            if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            src = os.path.join(root, fname)
            # mantieni la struttura delle cartelle
            rel_path = os.path.relpath(root, fake_path)
            dst_dir = os.path.join(out_fake_path, rel_path)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, fname)
        
            if not extract_face(src, dst):
                fake_skipped += 1
                print(f"Skipping image {src}")

        count += 1
        print(f"{count}")
            
    
    print(f"Done. Skipped fake frames (no face found): {fake_skipped}")

# Esempio d'uso
process_dataset()