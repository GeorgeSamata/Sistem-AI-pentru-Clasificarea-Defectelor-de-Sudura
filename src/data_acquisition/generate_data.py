import os
import cv2
import numpy as np
import glob
import shutil

BASE_DIR = "data"
RAW_DIR = os.path.join(BASE_DIR, "raw")         
GEN_DIR = os.path.join(BASE_DIR, "generated")   
TRAIN_DIR = os.path.join(BASE_DIR, "train")    
IMG_SIZE = (224, 224)

def add_noise(image):
    """Adauga zgomot 'sare si piper' sau Gaussian pentru a simula senzori industriali slabi."""
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss * 50
    return np.clip(noisy, 0, 255).astype(np.uint8)

def augment_brightness_flip(image):
    """Schimba luminozitatea si face flip orizontal aleatoriu."""
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    val = np.random.randint(-40, 40)
    v = cv2.add(v, val)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def main():
    print("=== START GENERARE DATE (REGULA 40%) ===")

    if os.path.exists(GEN_DIR): shutil.rmtree(GEN_DIR)
    if os.path.exists(TRAIN_DIR): shutil.rmtree(TRAIN_DIR)
    
    if not os.path.exists(RAW_DIR):
        print(f"[EROARE] Nu exista folderul {RAW_DIR}. Creeaza-l si pune pozele acolo!")
        return

    classes = [d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))]
    if not classes:
        print("[EROARE] Nu am gasit foldere de clase in data/raw!")
        return
    
    print(f"[INFO] Clase detectate: {classes}")

    count_original = 0
    count_generated = 0

    for cls in classes:
        os.makedirs(os.path.join(GEN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)

        src_path = os.path.join(RAW_DIR, cls)
        images = glob.glob(os.path.join(src_path, "*.*")) 

        print(f" Procesez clasa '{cls}' ({len(images)} imagini)...")

        for img_path in images:
            try:
                img = cv2.imread(img_path)
                if img is None: continue
                
                img = cv2.resize(img, IMG_SIZE)
                filename = os.path.basename(img_path)

                cv2.imwrite(os.path.join(TRAIN_DIR, cls, f"org_{filename}"), img)
                count_original += 1

                img_noise = add_noise(img)
                cv2.imwrite(os.path.join(GEN_DIR, cls, f"gen_noise_{filename}"), img_noise)
                cv2.imwrite(os.path.join(TRAIN_DIR, cls, f"gen_noise_{filename}"), img_noise)
                
                img_aug = augment_brightness_flip(img)
                cv2.imwrite(os.path.join(GEN_DIR, cls, f"gen_aug_{filename}"), img_aug)
                cv2.imwrite(os.path.join(TRAIN_DIR, cls, f"gen_aug_{filename}"), img_aug)

                count_generated += 2 

            except Exception as e:
                print(f" Eroare la {filename}: {e}")

    total = count_original + count_generated
    if total == 0:
        print("[EROARE] Nu am procesat nicio imagine.")
        return

    percent = (count_generated / total) * 100
    print("\n=== RAPORT FINAL ===")
    print(f" Imagini Originale: {count_original}")
    print(f" Imagini Generate (Tu): {count_generated}")
    print(f" TOTAL Dataset: {total}")
    print(f" Procent Contributie Proprie: {percent:.2f}% (Tinta era >40%)")
    print("====================")
    print(f" Datele de antrenare sunt gata in: {TRAIN_DIR}")

if __name__ == "__main__":
    main()