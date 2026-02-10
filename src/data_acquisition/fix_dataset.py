import os
import shutil
import glob

BASE_RAW_DIR = 'data/raw'
TARGET_CLASSES = ['bad_weld', 'crack', 'good_weld', 'porosity', 'spatter']
SEARCH_DIRS = [BASE_RAW_DIR, 
               os.path.join(BASE_RAW_DIR, 'train'), 
               os.path.join(BASE_RAW_DIR, 'valid'), 
               os.path.join(BASE_RAW_DIR, 'test')]

def normalize_name(name):
    """Transforma numele folderelor Roboflow in formatele noastre standard."""
    name = name.lower()
    if "crack" in name: return "crack"
    if "porosity" in name: return "porosity"
    if "spatter" in name: return "spatter"
   
    if "bad" in name: return "bad_weld" 
    if "good" in name: return "good_weld"
    return None

def fix_dataset_structure():
    print("=== 🧹 INCEPERE CURATENIE DATASET ===")
    
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(BASE_RAW_DIR, cls), exist_ok=True)

    moved_count = 0
    removed_folders = 0

    for current_dir in SEARCH_DIRS:
        if not os.path.exists(current_dir):
            continue
            
        print(f"\n🔍 Scanez folderul: {current_dir}")
        
        subfolders = [f for f in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, f))]

        for folder_name in subfolders:
            if folder_name in TARGET_CLASSES or folder_name in ['train', 'valid', 'test']:
                continue

            target_class = normalize_name(folder_name)
            folder_path = os.path.join(current_dir, folder_name)
            
            if target_class:
                dest_dir = os.path.join(BASE_RAW_DIR, target_class)
                images = glob.glob(os.path.join(folder_path, "*.*"))
                
                if images:
                    print(f"   ➡️ Mut {len(images)} poze din '{folder_name}' -> '{target_class}'")
                    for img in images:
                        file_name = os.path.basename(img)
                       
                        prefix = os.path.basename(current_dir) 
                        if prefix == "raw": prefix = "root"
                        
                        new_name = f"{prefix}_{folder_name}_{file_name}"
                        try:
                            shutil.move(img, os.path.join(dest_dir, new_name))
                            moved_count += 1
                        except Exception as e:
                            print(f"    [ERR] Nu am putut muta {file_name}: {e}")
                
                try:
                    shutil.rmtree(folder_path)
                    removed_folders += 1
                except:
                    pass
            else:
                if folder_name in ["_tokenization", "empty"] or not os.listdir(folder_path):
                     shutil.rmtree(folder_path)

    for extra_dir in ['train', 'valid', 'test']:
        dir_path = os.path.join(BASE_RAW_DIR, extra_dir)
        if os.path.exists(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"Am sters folderul gol: {extra_dir}")

    print("\n=== RAPORT FINAL ===")
    print(f" Imagini salvate corect: {moved_count}")
    print(f" Foldere inutile sterse: {removed_folders}")
    print(f" Verifica acum data/raw! Ar trebui sa ai doar: {', '.join(TARGET_CLASSES)}")

if __name__ == "__main__":
    fix_dataset_structure()