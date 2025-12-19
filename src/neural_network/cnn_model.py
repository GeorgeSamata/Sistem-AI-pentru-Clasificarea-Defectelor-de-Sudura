import tensorflow as tf
from tensorflow.keras import layers, models
import os

class WeldingCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=2):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Defineste arhitectura retelei neuronale (CNN).
        Aceasta este o arhitectura secventiala clasica.
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),

            layers.Dense(64, activation='relu'),
            layers.Dense(self.num_classes, activation='softmax') 
        ])

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        return model

    def save_model(self, path='models/welding_model_v1.keras'):
        """Salveaza modelul (chiar daca e neantrenat) pentru a fi folosit in UI."""
        self.model.save(path)
        print(f"[INFO] Model salvat la {path}")

    def load_weights(self, path):
        """Incarca greutati antrenate."""
        if os.path.exists(path):
            self.model = tf.keras.models.load_model(path)
            print("[INFO] Model incarcat cu succes.")
        else:
            print("[WARN] Fisierul modelului nu exista. Folosesc initializare random.")

    def predict_image(self, img_array):
        """Primeste un numpy array (224,224,3) si returneaza clasa."""
        img_batch = tf.expand_dims(img_array, 0)
        
        predictions = self.model.predict(img_batch, verbose=0)
        score = tf.nn.softmax(predictions[0])
        
        class_id = np.argmax(predictions[0])
        confidence = 100 * np.max(score)
        
        return class_id, confidence

if __name__ == "__main__":
    import numpy as np
    
    print("[INFO] Initializare Model CNN...")
    net = WeldingCNN()
    net.model.summary() 
    
    net.save_model()