import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

class WeldingCNN:
    def __init__(self, input_shape=(224, 224, 3), num_classes=5):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            
            layers.Flatten(),

            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5), 
            
            layers.Dense(self.num_classes, activation='softmax') 
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def save_model(self, path='models/trained_model.keras'):
        self.model.save(path)
        print(f"[INFO] Model salvat la {path}")

    def predict_image(self, img_array):
        img_batch = np.expand_dims(img_array, axis=0) 
        predictions = self.model.predict(img_batch, verbose=0)
        score = predictions[0]
        class_id = np.argmax(score)
        confidence = 100 * np.max(score)
        return class_id, confidence