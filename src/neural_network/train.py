import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import os
import sys

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_SCRIPT_DIR)) 
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'train')

sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

try:
    from neural_network.cnn_model import WeldingCNN
except ImportError:
    try:
        from cnn_model import WeldingCNN
    except ImportError:
        print("[CRITIC] Nu pot importa cnn_model. Verifica structura!")
        sys.exit(1)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30
LEARNING_RATE = 0.001

def train():
    print(f"=== START ANTRENARE ===")
    print(f"[DEBUG] Caut datele in folderul: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        print(f"[EROARE FATALA] Folderul nu exista pe disc!")
        print(f"Te rog verifica daca ai folderul 'train' in 'data' folosind File Explorer.")
        return

    if not os.listdir(DATA_DIR):
        print(f"[EROARE] Folderul {DATA_DIR} este GOL!")
        return

    print(f"[INFO] Folder gasit. Incep incarcarea imaginilor...")
    
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATA_DIR,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
    except ValueError as e:
        print(f"[EROARE DATASET] {e}")
        print("Verifica daca in 'data/train' ai folderele claselor (bad_weld, crack, etc).")
        return

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"[INFO] Clase detectate ({num_classes}): {class_names}")

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    print(f"[INFO] Initializare model CNN...")
    cnn = WeldingCNN(input_shape=IMG_SIZE + (3,), num_classes=num_classes)
    model = cnn.model
    
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    models_dir = os.path.join(PROJECT_ROOT, 'models')
    results_dir = os.path.join(PROJECT_ROOT, 'results')
    docs_dir = os.path.join(PROJECT_ROOT, 'docs')
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(os.path.join(models_dir, 'trained_model.keras'), save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=5, restore_best_weights=True),
        CSVLogger(os.path.join(results_dir, 'training_history.csv'))
    ]

    print("[INFO] Start antrenare (poate dura cateva minute)...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Evolutie Antrenare')
    plt.xlabel('Epoci')
    plt.ylabel('Acuratete')
    plt.legend()
    plt.savefig(os.path.join(docs_dir, 'loss_curve.png'))
    print(f"[SUCCES] Antrenare completa! Graficul salvat in docs/loss_curve.png")

if __name__ == "__main__":
    train()