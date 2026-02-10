import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parent.parent.parent  
DATA_DIR = PROJECT_ROOT / 'data' / 'train'

sys.path.append(str(PROJECT_ROOT / 'src'))

print(f"=== DIAGNOSTIC CAI ===")
print(f"Radacina proiect: {PROJECT_ROOT}")
print(f"Folder date tinta: {DATA_DIR}")
print(f"Folderul exista? {DATA_DIR.exists()}")

if DATA_DIR.exists():
    content = list(DATA_DIR.glob('*'))
    print(f"Continut folder data/train: {[p.name for p in content if p.is_dir()]}")
else:
    print("[EROARE CRITICA] Folderul data/train NU a fost gasit fizic pe disc!")
    data_root = PROJECT_ROOT / 'data'
    if data_root.exists():
        print(f"Continut folder 'data': {[p.name for p in data_root.glob('*')]}")
    sys.exit(1)

try:
    from neural_network.cnn_model import WeldingCNN
except ImportError:
    try:
        from cnn_model import WeldingCNN
    except ImportError:
        print("[CRITIC] Nu pot importa cnn_model!")
        sys.exit(1)

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 30

def train():
    print(f"\n=== START ANTRENARE ===")
    
   
    data_dir_str = str(DATA_DIR)

    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir_str,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir_str,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            label_mode='categorical'
        )
    except Exception as e:
        print(f"[EROARE TENSORFLOW] {e}")
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

    models_dir = PROJECT_ROOT / 'models'
    results_dir = PROJECT_ROOT / 'results'
    docs_dir = PROJECT_ROOT / 'docs'
    
    for d in [models_dir, results_dir, docs_dir]:
        d.mkdir(exist_ok=True)

    callbacks = [
        ModelCheckpoint(str(models_dir / 'trained_model.keras'), save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=5, restore_best_weights=True),
        CSVLogger(str(results_dir / 'training_history.csv'))
    ]

    print("[INFO] Start antrenare...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Evolutie Antrenare')
    plt.legend()
    plt.savefig(str(docs_dir / 'loss_curve.png'))
    print(f"[SUCCES] Gata! Grafic salvat.")

if __name__ == "__main__":
    train()