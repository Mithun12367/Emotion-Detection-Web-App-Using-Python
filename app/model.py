import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import os
import numpy as np
import cv2
import sys

class EmotionModel:
    def __init__(self, weights_path='models/model_weights.weights.h5'):
        self.weights_path = weights_path
        self.model = self._build_model()
        self.emotions = {
            0: "Angry",
            1: "Disgust",
            2: "Fear",
            3: "Happy",
            4: "Sad",
            5: "Surprise",
            6: "Neutral"
        }
        self.load_weights()

    def _build_model(self):
        """
        Defines a VGG-like CNN architecture suitable for FER-2013 (48x48)
        """
        model = Sequential()

        # Block 1
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal', input_shape=(48, 48, 1)))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())

        # Block 2
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())
        
        # Block 3
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())

        # Block 4
        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu', kernel_initializer='he_normal'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def load_weights(self):
        if os.path.exists(self.weights_path):
            try:
                self.model.load_weights(self.weights_path)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"Error loading weights: {e}")
        else:
            print(f"Warning: Weights file not found at {self.weights_path}. Generating random weights for demo purposes.")
            # Create models directory if not exists
            os.makedirs(os.path.dirname(self.weights_path), exist_ok=True)
            self.model.save_weights(self.weights_path)
            print(f"Created dummy weights at {self.weights_path}")

    def predict(self, face_image):
        """
        Predicts emotion from a preprocessed face image (48x48 grayscale).
        """
        try:
            # Check input
            if face_image is None or face_image.size == 0:
                print("Error: Empty face image passed to predict", file=sys.stderr)
                return "Unknown", 0.0

            # Resize using OpenCV (more robust for numpy arrays than tf.image.resize)
            if face_image.shape != (48, 48):
                face_image = cv2.resize(face_image, (48, 48))

            # Normalize 0-1
            face_image = face_image.astype('float32') / 255.0
            
            # Reshape to (1, 48, 48, 1)
            # Input is (48, 48)
            face_image = np.expand_dims(face_image, axis=-1) # (48, 48, 1)
            face_image = np.expand_dims(face_image, axis=0)  # (1, 48, 48, 1)

            predictions = self.model.predict(face_image, verbose=0)
            max_index = int(np.argmax(predictions))
            confidence = float(predictions[0][max_index])
            
            return self.emotions[max_index], confidence
        except Exception as e:
            import sys
            print(f"Prediction Error: {e}", file=sys.stderr)
            return "Error", 0.0
