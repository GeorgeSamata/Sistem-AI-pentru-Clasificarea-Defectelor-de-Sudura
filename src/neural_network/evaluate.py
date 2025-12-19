import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TEST_REC = 'data/test/porosity-spatter-crack.tfrecord'
MODEL_PATH = 'models/trained_model.keras'
RESULTS_DIR = 'results'
DOCS_DIR = 'docs'

CLASS_NAMES = ["Bad Weld", "Crack", "Good Weld", "Porosity", "Spatter"]

def parse_test_fn(example):
    feature_description = {
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }
    example = tf.io.parse_single_example(example, feature_description)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    labels = tf.sparse.to_dense(example['image/object/class/label'])
    label = tf.cond(tf.size(labels) > 0, lambda: labels[0], lambda: tf.constant(3, dtype=tf.int64))
    return image, label - 1

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("Nu exista model antrenat!")
        return

    print("[INFO] Incarcare model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("[INFO] Evaluare pe Test Set...")
    dataset = tf.data.TFRecordDataset(TEST_REC)
    dataset = dataset.map(parse_test_fn).batch(BATCH_SIZE)

    y_true = []
    y_pred = []

    for batch_imgs, batch_labels in dataset:
        preds = model.predict(batch_imgs, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        y_true.extend(batch_labels.numpy())
        y_pred.extend(pred_classes)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    print(f"\n[REZULTAT] Accuracy: {acc:.4f}")
    print(f"[REZULTAT] F1-Score: {f1:.4f}")
    
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump({"test_accuracy": float(acc), "test_f1_macro": float(f1)}, f)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matrice de Confuzie')
    plt.savefig(os.path.join(DOCS_DIR, "confusion_matrix.png"))
    print("[INFO] Matrice salvata!")

if __name__ == "__main__":
    evaluate()