import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'trained_model.keras')
DOCS_DIR = os.path.join(PROJECT_ROOT, 'docs')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

def evaluate():
    print("=== GENERARE RAPORT PERFORMANTA ===")
    
    os.makedirs(DOCS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    
    try:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical', 
            shuffle=True              
        )
    except Exception as e:
        print(f"[EROARE] Nu pot incarca datele: {e}")
        return
    
    class_names = val_ds.class_names
    print(f"Clase detectate: {class_names}")

    if not os.path.exists(MODEL_PATH):
        print(f"[EROARE] Nu gasesc modelul la {MODEL_PATH}")
        return
        
    model = tf.keras.models.load_model(MODEL_PATH)

    print("Fac predictii pe setul de validare...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        images = images / 255.0
        
        preds = model.predict(images, verbose=0)
        

        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predictie Model')
    plt.ylabel('Adevarat (Real)')
    plt.title('Confusion Matrix - Defecte Sudura')
    
    save_path = os.path.join(DOCS_DIR, 'confusion_matrix.png')
    plt.savefig(save_path)
    print(f"[SUCCES] Matrice salvata in: {save_path}")


    report = classification_report(
        y_true, 
        y_pred, 
        target_names=class_names,
        zero_division=0
    )
    
    print("\n=== RAPORT DETALIAT ===")
    print(report)
    
    with open(os.path.join(RESULTS_DIR, 'final_metrics.txt'), 'w') as f:
        f.write(report)
        
    print(f"[SUCCES] Raport salvat in results/final_metrics.txt")

if __name__ == "__main__":
    evaluate()