import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import sys

# --- Configurare ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = "data"
MODEL_PATH = os.path.join("models", "trained_model.keras")
RESULTS_DIR = "results"
DOCS_DIR = "docs"
CLASS_NAMES = ["Defective", "Good"] # 0 - Defective, 1 - Good (Ordinea alfabetica)

# Asiguram existenta folderelor de output
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def evaluate():
    # 1. Verificari preliminare
    if not os.path.exists(MODEL_PATH):
        print(f"[EROARE] Nu exista model antrenat la {MODEL_PATH}.")
        print("Te rog ruleaza intai 'src/neural_network/train_improved.py'!")
        return

    if not os.path.exists(os.path.join(DATA_DIR, 'test')):
        print(f"[EROARE] Folderul de test nu exista in {DATA_DIR}.")
        print("Te rog ruleaza intai 'src/preprocessing/prepare_final_dataset.py'!")
        return

    # 2. Incarcare Model
    print(f"[INFO] Incarcare model din {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"[EROARE] Nu s-a putut incarca modelul: {e}")
        return

    # 3. Pregatire Generator Date Test
    # IMPORTANT: shuffle=False este critic pentru a mentine ordinea corecta a etichetelor
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False, # NU amesteca, pentru a putea compara cu etichetele reale
        seed=42
    )

    # Afisare clase detectate pentru verificare
    print(f"[INFO] Clase detectate: {test_generator.class_indices}")
    
    # 4. Inferenta (Predictie)
    print("[INFO] Rulare predictii pe setul de Test...")
    y_pred_prob = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1) # Clasa prezisa (0 sau 1)
    y_true = test_generator.classes         # Clasa reala (0 sau 1)

    # 5. Calcul Metrici
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print("\n" + "="*40)
    print(f"REZULTATE FINALE PE TEST SET")
    print("="*40)
    print(f"Acuratete (Accuracy): {acc:.4f}")
    print(f"F1-Score (Macro):     {f1:.4f}")
    print("-" * 40)
    print("Raport Detaliat:")
    # Atentie la ordinea numelor claselor, trebuie sa fie alfabetica daca generatorul le-a luat asa
    # Generatorul le ia alfabetic: 'defective' (0), 'good' (1)
    # Deci target_names trebuie sa fie ['defective', 'good']
    report_names = list(test_generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=report_names))
    print("="*40 + "\n")

    # 6. Salvare Rezultate JSON
    metrics = {
        "test_accuracy": round(float(acc), 4),
        "test_f1_macro": round(float(f1), 4)
    }
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[INFO] Metrici salvate in {os.path.join(RESULTS_DIR, 'test_metrics.json')}")

    # 7. Generare Matrice de Confuzie
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=report_names,
                yticklabels=report_names)
    plt.xlabel('Predictie Model')
    plt.ylabel('Eticheta Reala')
    plt.title('Matrice de Confuzie')
    
    cm_path = os.path.join(DOCS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"[INFO] Matricea de confuzie salvata in {cm_path}")

if __name__ == "__main__":
    evaluate()