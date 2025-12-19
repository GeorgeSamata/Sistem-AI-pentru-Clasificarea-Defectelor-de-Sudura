import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.neural_network.cnn_model import WeldingCNN

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 50
LEARNING_RATE = 0.001
NUM_CLASSES = 5 

TRAIN_REC = 'data/train/porosity-spatter-crack.tfrecord'
VAL_REC = 'data/valid/porosity-spatter-crack.tfrecord'
MODEL_SAVE_PATH = 'models/trained_model.keras'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

def parse_tfrecord_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    label_id = tf.cond(tf.size(labels) > 0, lambda: labels[0], lambda: tf.constant(3, dtype=tf.int64))
    
    label_index = label_id - 1
    label = tf.one_hot(label_index, NUM_CLASSES)
    
    return image, label

def get_dataset(path, is_train=False):
    if not os.path.exists(path):
        print(f"Eroare: Nu gasesc {path}")
        return None
    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if is_train:
        dataset = dataset.shuffle(1000)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def train():
    train_ds = get_dataset(TRAIN_REC, is_train=True)
    val_ds = get_dataset(VAL_REC)
    if not train_ds or not val_ds: return

    print(f"[INFO] Initializare model pentru {NUM_CLASSES} clase...")
    cnn = WeldingCNN(input_shape=IMG_SIZE + (3,), num_classes=NUM_CLASSES)
    model = cnn.model
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, monitor='val_loss'),
        EarlyStopping(patience=8, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
        CSVLogger(os.path.join(RESULTS_DIR, 'training_history.csv'))
    ]

    print("[INFO] Start Antrenare...")
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Acuratete')
    plt.savefig(os.path.join(DOCS_DIR, "loss_curve.png"))
    print("[INFO] Grafic salvat!")

if __name__ == "__main__":
    train()