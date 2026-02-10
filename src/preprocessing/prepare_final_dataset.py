import tensorflow as tf
import os
import sys

DATA_DIR = "data"
TRAIN_REC = os.path.join(DATA_DIR, "train", "porosity-spatter-crack.tfrecord")
VAL_REC = os.path.join(DATA_DIR, "valid", "porosity-spatter-crack.tfrecord")
TEST_REC = os.path.join(DATA_DIR, "test", "porosity-spatter-crack.tfrecord")

def count_records(tfrecord_path):
    """Numara imaginile dintr-un fisier TFRecord."""
    if not os.path.exists(tfrecord_path):
        return 0
    count = 0
    try:
        for _ in tf.data.TFRecordDataset(tfrecord_path):
            count += 1
    except Exception as e:
        print(f"[WARN] Eroare la citirea {tfrecord_path}: {e}")
        return 0
    return count

def main():
    print("=== Verificare Structura Date (TFRecord) ===\n")
    
    files = {
        "Train": TRAIN_REC,
        "Validation": VAL_REC,
        "Test": TEST_REC
    }
    
    all_ok = True
    total_images = 0
    
    for name, path in files.items():
        if os.path.exists(path):
            n = count_records(path)
            if n > 0:
                print(f"[OK] {name}: Gasit ({n} imagini)")
                total_images += n
            else:
                print(f"[GOL] {name}: Fisierul exista dar pare gol!")
                all_ok = False
        else:
            print(f"[LIPSA] {name}: Nu exista la {path}")
            all_ok = False
            
    if all_ok:
        print(f"\n[SUCCES] Totul este pregatit! Total imagini: {total_images}")
    else:
        print("\n[EROARE] Verifica folderele data/train, data/valid, data/test!")

if __name__ == "__main__":
    main()