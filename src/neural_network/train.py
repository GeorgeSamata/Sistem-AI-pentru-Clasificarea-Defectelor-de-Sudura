import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import os
import sys

# Adăugăm calea către root pentru a putea importa modelul
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.neural_network.cnn_model import WeldingCNN

# --- Configurare ---
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50 
LEARNING_RATE = 0.001

# Căi
DATA_DIR = "data"
MODELS_DIR = "models"
RESULTS_DIR = "results"
DOCS_DIR = "docs"

# Asigurăm existența directoarelor de output
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

def plot_training_history(history):
    """Generează și salvează graficele de Loss și Accuracy."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    
    # Grafic Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, loss, label='Training Loss', marker='o')
    plt.plot(epochs_range, val_loss, label='Validation Loss', marker='o')
    plt.legend(loc='upper right')
    plt.title('Curba de Învățare - Loss')
    plt.xlabel('Epoci')
    plt.ylabel('Loss')
    plt.grid(True)

    # Grafic Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, acc, label='Training Accuracy', marker='o')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy', marker='o')
    plt.legend(loc='lower right')
    plt.title('Curba de Învățare - Acuratețe')
    plt.xlabel('Epoci')
    plt.ylabel('Acuratețe')
    plt.grid(True)
    
    plot_path = os.path.join(DOCS_DIR, "loss_curve.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"[INFO] Grafic salvat în {plot_path}")

def train():
    # 1. Pregătirea Generatorilor de Date (Rescaling 0-1)
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    print("[INFO] Încărcare date de antrenare și validare...")
    
    # Train Generator
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse', # 0 sau 1
        shuffle=True,
        seed=42
    )

    # Validation Generator
    validation_generator = val_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False,
        seed=42
    )

    # 2. Inițializare Model
    print("[INFO] Inițializare model CNN...")
    cnn_wrapper = WeldingCNN(input_shape=IMG_SIZE + (3,), num_classes=2)
    model = cnn_wrapper.model
    
    # Compilare cu Learning Rate specific
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.summary()

    # 3. Callbacks (Nivel 2)
    callbacks = [
        # Salvează modelul doar când val_loss e minim
        ModelCheckpoint(
            filepath=os.path.join(MODELS_DIR, 'trained_model.keras'),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        # Oprește antrenarea dacă nu mai învață
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce viteza de învățare la platou
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        # Log istoric în CSV
        CSVLogger(os.path.join(RESULTS_DIR, 'training_history.csv'))
    ]

    # 4. Start Antrenare
    print(f"[INFO] Pornire antrenare (Max {EPOCHS} epoci)...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )

    # 5. Salvare Grafice
    plot_training_history(history)
    print("[SUCCESS] Antrenare completă. Model salvat în 'models/trained_model.keras'.")

if __name__ == "__main__":
    # Verificare de siguranță
    if os.path.exists(os.path.join(DATA_DIR, 'train')):
        train()
    else:
        print("[EROARE] Nu ai date pregătite. Rulează întâi 'src/preprocessing/prepare_final_dataset.py'!")