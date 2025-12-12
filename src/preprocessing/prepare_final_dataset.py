import os
import shutil
import random
import glob

# --- Configurare ---
DATA_ROOT = "data"
RAW_DIR = os.path.join(DATA_ROOT, "raw")          # Datele Kaggle
GEN_DIR = os.path.join(DATA_ROOT, "generated")    # Datele sintetice (Contribuția 40%)
EXTRA_DIR = os.path.join(DATA_ROOT, "extra")      # 17 imagini

# Lista cu toate sursele de date
BASE_DIRS = [RAW_DIR, GEN_DIR, EXTRA_DIR]

FINAL_DIRS = {
    "train": os.path.join(DATA_ROOT, "train"),
    "validation": os.path.join(DATA_ROOT, "validation"),
    "test": os.path.join(DATA_ROOT, "test")
}
CLASSES = ["defective", "good"]
SPLIT_RATIOS = (0.70, 0.15, 0.15) # 70% Train, 15% Val, 15% Test
SEED = 42

def setup_directories():
    """Curăță și recreează structura de foldere."""
    print("[INFO] Configurare directoare țintă...")
    for split_name, split_path in FINAL_DIRS.items():
        if os.path.exists(split_path):
            shutil.rmtree(split_path)
        for cls in CLASSES:
            os.makedirs(os.path.join(split_path, cls), exist_ok=True)

def collect_and_split():
    """Colectează fișierele din toate sursele (Raw, Generated, Extra)."""
    random.seed(SEED)
    print("[INFO] Colectare și împărțire fișiere...")

    total_files = 0
    
    for cls in CLASSES:
        all_cls_files = []
        
        # Căutăm în TOATE locațiile (inclusiv cele 17 extra)
        for base_dir in BASE_DIRS:
            src_path = os.path.join(base_dir, cls, "*")
            files = glob.glob(src_path)
            # Filtrăm doar imaginile
            files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if len(files) > 0:
                print(f"   -> Găsit {len(files)} imagini în {base_dir}/{cls}")
                
            all_cls_files.extend(files)
            
        # Amestecăm
        random.shuffle(all_cls_files)
        
        n = len(all_cls_files)
        total_files += n
        
        # Calculăm split-ul
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        
        train_files = all_cls_files[:n_train]
        val_files = all_cls_files[n_train:n_train+n_val]
        test_files = all_cls_files[n_train+n_val:]
        
        print(f"[TOTAL] Clasa '{cls}': {n} imagini -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Copiere fizică
        def copy_files(file_list, split_name):
            dest_dir = os.path.join(FINAL_DIRS[split_name], cls)
            for f in file_list:
                try:
                    shutil.copy(f, os.path.join(dest_dir, os.path.basename(f)))
                except Exception as e:
                    print(f"[WARN] Nu s-a putut copia {f}: {e}")
                
        copy_files(train_files, "train")
        copy_files(val_files, "validation")
        copy_files(test_files, "test")

    print(f"\n[SUCCESS] Dataset final pregătit cu {total_files} imagini total!")

if __name__ == "__main__":
    # Verificăm dacă userul a creat folderul extra
    if not os.path.exists(EXTRA_DIR):
        print(f"[INFO] Folderul {EXTRA_DIR} nu există. Se va crea automat.")
        os.makedirs(os.path.join(EXTRA_DIR, "good"), exist_ok=True)
        os.makedirs(os.path.join(EXTRA_DIR, "defective"), exist_ok=True)
        print(f"[IMPORTANT] Pune cele 17 poze în {EXTRA_DIR}/good sau {EXTRA_DIR}/defective înainte să rulezi din nou!")
    else:
        setup_directories()
        collect_and_split()