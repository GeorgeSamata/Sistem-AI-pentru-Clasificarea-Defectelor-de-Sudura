import os
import cv2
import numpy as np
import random

# Configurare Cai
BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")
GEN_DIR = os.path.join(BASE_DIR, "generated")
CLASSES = ["defective", "good"] # Clasele din datasetul Kaggle

def ensure_structure():
    """Creeaza structura de foldere."""
    for cls in CLASSES:
        os.makedirs(os.path.join(RAW_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(GEN_DIR, cls), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    print("[INFO] Structura de foldere verificata.")

def create_dummy_data_if_empty(samples=5):
    """
    Daca nu exista date Kaggle descarcate, generam cateva imagini dummy
    doar ca sa nu crape aplicatia la prima rulare.
    """
    for cls in CLASSES:
        path = os.path.join(RAW_DIR, cls)
        if not os.listdir(path):
            print(f"[WARN] Nu am gasit imagini in {path}. Generare dummy...")
            for i in range(samples):
                # Imagine random noise color
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                # Daca e defect, desenam un cerc rosu
                if cls == "defective":
                    cv2.circle(img, (112, 112), 50, (0, 0, 255), -1)
                cv2.imwrite(os.path.join(path, f"dummy_{i}.jpg"), img)

def augment_data(target_count=50):
    """
    CONTRIBUTIA ORIGINALA (40%):
    Generam imagini noi aplicand transformari pe cele existente.
    """
    print("[INFO] Pornire generare date sintetice...")
    
    total_generated = 0
    
    for cls in CLASSES:
        src_path = os.path.join(RAW_DIR, cls)
        dst_path = os.path.join(GEN_DIR, cls)
        
        images = os.listdir(src_path)
        if not images:
            continue
            
        # Generam pana atingem target_count
        for i in range(target_count):
            img_name = random.choice(images)
            img = cv2.imread(os.path.join(src_path, img_name))
            
            if img is None: continue

            # Augmentare 1: Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            
            # Augmentare 2: Modificare luminozitate
            value = random.randint(-50, 50)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, value)
            v[v > 255] = 255
            v[v < 0] = 0
            final_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            # Salvare
            cv2.imwrite(os.path.join(dst_path, f"aug_{i}_{img_name}"), img)
            total_generated += 1
            
    print(f"[SUCCESS] Am generat {total_generated} imagini sintetice in {GEN_DIR}")

if __name__ == "__main__":
    ensure_structure()
    create_dummy_data_if_empty() # Doar pentru demo
    augment_data(target_count=20) # Genereaza 20 imagini noi