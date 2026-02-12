"""
3-Layer ANN Model for Bitcoin Trading Prediction
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os


class BTCTradingModel:
    """3-layer ANN for predicting BTC price direction"""
    
    def __init__(self, sequence_length=20, n_features=6):
        """
        Initialize the model
        
        Args:
            sequence_length: Length of input sequences (default: 20)
            n_features: Number of features per timestep (default: 6)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build 3-layer ANN (512-256-128 neurons, ReLU activation)
        """
        model = models.Sequential([
            # Flatten the input sequences
            layers.Flatten(input_shape=(self.sequence_length, self.n_features)),
            
            # First layer: 512 neurons
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            
            # Second layer: 256 neurons
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            
            # Third layer: 128 neurons
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.2),
            
            # Output layer: Binary classification (up/down)
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile with Adam optimizer
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def train(self, train_X, train_y, validation_split=0.2, epochs=50, batch_size=64):
        """
        Train the model
        
        Args:
            train_X: Training features
            train_y: Training labels
            validation_split: Validation split ratio (default: 0.2)
            epochs: Number of training epochs (default: 50)
            batch_size: Batch size (default: 64)
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        self.history = self.model.fit(
            train_X, train_y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return self.history
    
    def predict_proba(self, X):
        """
        Predict probability of price going up
        
        Args:
            X: Input features
            
        Returns:
            Probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        return self.model.predict(X)
    
    def predict_with_threshold(self, X, threshold=0.65):
        """
        Predict with probability threshold
        Only returns predictions with confidence > threshold
        
        Args:
            X: Input features
            threshold: Minimum probability threshold (default: 0.65)
            
        Returns:
            Tuple of (predictions, probabilities, confident_mask)
            predictions: -1 (no action), 0 (down), 1 (up)
            probabilities: Raw probabilities
            confident_mask: Boolean mask of confident predictions
        """
        probabilities = self.predict_proba(X).flatten()
        
        predictions = np.full(len(probabilities), -1)  # -1 = no action
        
        # High confidence for UP (probability > threshold)
        up_mask = probabilities > threshold
        predictions[up_mask] = 1
        
        # High confidence for DOWN (probability < 1 - threshold)
        down_mask = probabilities < (1 - threshold)
        predictions[down_mask] = 0
        
        confident_mask = up_mask | down_mask
        
        return predictions, probabilities, confident_mask
    
    def evaluate(self, test_X, test_y):
        """
        Evaluate the model
        
        Args:
            test_X: Test features
            test_y: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        loss, accuracy, auc = self.model.evaluate(test_X, test_y, verbose=0)
        
        # Get predictions with threshold
        predictions, probabilities, confident_mask = self.predict_with_threshold(test_X)
        
        # Calculate accuracy on confident predictions only
        confident_predictions = predictions[confident_mask]
        confident_labels = test_y[confident_mask]
        
        if len(confident_predictions) > 0:
            confident_accuracy = (confident_predictions == confident_labels).mean()
        else:
            confident_accuracy = 0.0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'auc': auc,
            'confident_predictions_ratio': confident_mask.mean(),
            'confident_accuracy': confident_accuracy
        }
    
    def save_model(self, filepath='models/btc_ann_model.h5'):
        """Save model to file"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='models/btc_ann_model.h5'):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
