"""
Baseline Model Training Module
Trains RandomForest/XGBoost models on synthetic labeled data for rockfall prediction.
"""

import numpy as np
import pandas as pd
import os
import json
import logging
import joblib
from datetime import datetime
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Using RandomForest only.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineTrainer:
    def __init__(self, data_dir="data", models_dir="models"):
        """
        Initialize baseline model trainer.
        
        Args:
            data_dir (str): Data directory
            models_dir (str): Models directory
        """
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # Create models directory if it doesn't exist
        os.makedirs(models_dir, exist_ok=True)
        
        # Model configurations
        self.models = {}
        self.scalers = {}
        self.label_encoder = LabelEncoder()
        
        # Performance metrics storage
        self.training_results = {}
    
    def generate_synthetic_labels(self, features, anomaly_threshold=2.0):
        """
        Generate synthetic labels based on feature values.
        
        Args:
            features (np.ndarray): Feature matrix
            anomaly_threshold (float): Threshold for anomaly detection
        
        Returns:
            tuple: (labels, probabilities)
        """
        logger.info("Generating synthetic labels based on feature patterns")
        
        if features.size == 0:
            return np.array([]), np.array([])
        
        # Normalize features for labeling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Calculate anomaly scores based on feature deviations
        # This is a simplified approach - in practice, you'd have real labels
        
        labels = []
        probabilities = []
        
        for i, sample in enumerate(features_scaled):
            # Calculate composite anomaly score
            # Focus on vibration and motion features (first few features)
            vibration_score = 0
            motion_score = 0
            
            # Vibration-related features (sensor data)
            if features.shape[1] > 10:  # If we have sensor features
                vibration_features = sample[8:]  # Sensor features start after vision features
                vibration_score = np.mean(np.abs(vibration_features[:15]))  # First 15 are vibration-related
            
            # Motion-related features (vision data)
            if features.shape[1] > 5:  # If we have vision features
                motion_features = sample[:5]  # First 5 are motion-related
                motion_score = np.mean(np.abs(motion_features))
            
            # Combine scores
            composite_score = 0.7 * vibration_score + 0.3 * motion_score
            
            # Add some randomness to make it more realistic
            noise = np.random.normal(0, 0.1)
            composite_score += noise
            
            # Determine label based on thresholds
            if composite_score > anomaly_threshold:
                label = "critical"
                prob = min(0.8 + np.random.uniform(0, 0.2), 1.0)
            elif composite_score > anomaly_threshold * 0.6:
                label = "warning"
                prob = 0.4 + np.random.uniform(0, 0.3)
            else:
                label = "safe"
                prob = np.random.uniform(0, 0.3)
            
            labels.append(label)
            probabilities.append(prob)
        
        # Convert to numpy arrays
        labels = np.array(labels)
        probabilities = np.array(probabilities)
        
        # Log label distribution
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = dict(zip(unique, counts))
        logger.info(f"Generated label distribution: {label_dist}")
        
        return labels, probabilities
    
    def prepare_training_data(self, features_file=None, test_size=0.2, random_state=42):
        """
        Prepare training and test datasets.
        
        Args:
            features_file (str): Path to features file
            test_size (float): Test set proportion
            random_state (int): Random state for reproducibility
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        # Load features
        if features_file is None:
            features_file = os.path.join(self.processed_dir, "combined_features.npy")
        
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None, None, None, None
        
        features = np.load(features_file)
        logger.info(f"Loaded features shape: {features.shape}")
        
        # Generate synthetic labels
        labels, probabilities = self.generate_synthetic_labels(features)
        
        if len(labels) == 0:
            logger.error("No labels generated")
            return None, None, None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train, hyperparameter_tuning=True):
        """
        Train Random Forest classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
        Returns:
            RandomForestClassifier: Trained model
        """
        logger.info("Training Random Forest classifier")
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(
                rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_
            
            logger.info(f"Best Random Forest parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            best_rf.fit(X_train, y_train)
        
        return best_rf
    
    def train_xgboost(self, X_train, y_train, hyperparameter_tuning=True):
        """
        Train XGBoost classifier.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
        Returns:
            XGBClassifier: Trained model or None if XGBoost not available
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available")
            return None
        
        logger.info("Training XGBoost classifier")
        
        # Encode labels for XGBoost
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        if hyperparameter_tuning:
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_encoded)
            best_xgb = grid_search.best_estimator_
            
            logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Use default parameters
            best_xgb = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                random_state=42,
                eval_metric='mlogloss'
            )
            best_xgb.fit(X_train, y_encoded)
        
        return best_xgb
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            model_name (str): Model name for reporting
        
        Returns:
            dict: Evaluation metrics
        """
        logger.info(f"Evaluating {model_name}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Handle XGBoost label encoding
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            # For XGBoost, convert back to original labels
            y_test_encoded = self.label_encoder.transform(y_test)
            y_pred_original = self.label_encoder.inverse_transform(y_pred)
        else:
            y_test_encoded = y_test
            y_pred_original = y_pred
        
        # Classification report
        class_report = classification_report(y_test, y_pred_original, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_original)
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        # ROC AUC (for multiclass, use ovr strategy)
        try:
            if len(np.unique(y_test)) > 2:
                roc_auc = roc_auc_score(y_test_encoded, y_pred_proba, multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_test_encoded, y_pred_proba[:, 1])
        except:
            roc_auc = None
        
        # Store results
        results = {
            'model_name': model_name,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'roc_auc': roc_auc,
            'feature_importance': feature_importance.tolist() if feature_importance is not None else None,
            'accuracy': class_report['accuracy'],
            'weighted_f1': class_report['weighted avg']['f1-score']
        }
        
        # Log key metrics
        logger.info(f"{model_name} - Accuracy: {results['accuracy']:.4f}")
        logger.info(f"{model_name} - Weighted F1: {results['weighted_f1']:.4f}")
        if roc_auc:
            logger.info(f"{model_name} - ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def save_model(self, model, model_name, scaler=None):
        """
        Save trained model and associated components.
        
        Args:
            model: Trained model
            model_name (str): Model name
            scaler: Feature scaler (optional)
        """
        # Save model
        model_file = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        joblib.dump(model, model_file)
        logger.info(f"Saved model: {model_file}")
        
        # Save scaler if provided
        if scaler is not None:
            scaler_file = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
            joblib.dump(scaler, scaler_file)
            logger.info(f"Saved scaler: {scaler_file}")
        
        # Save label encoder for XGBoost
        if hasattr(self.label_encoder, 'classes_') and len(self.label_encoder.classes_) > 0:
            encoder_file = os.path.join(self.models_dir, f"{model_name}_label_encoder.joblib")
            joblib.dump(self.label_encoder, encoder_file)
            logger.info(f"Saved label encoder: {encoder_file}")
        
        # Save model metadata
        metadata = {
            'model_name': model_name,
            'model_type': type(model).__name__,
            'training_date': datetime.now().isoformat(),
            'feature_count': model.n_features_in_ if hasattr(model, 'n_features_in_') else None,
            'classes': model.classes_.tolist() if hasattr(model, 'classes_') else None
        }
        
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata: {metadata_file}")
    
    def train_baseline_models(self, features_file=None, hyperparameter_tuning=True):
        """
        Train all baseline models.
        
        Args:
            features_file (str): Path to features file
            hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
        Returns:
            dict: Training results
        """
        logger.info("Starting baseline model training")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(features_file)
        
        if X_train is None:
            logger.error("Failed to prepare training data")
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        
        # Train Random Forest
        try:
            rf_model = self.train_random_forest(X_train_scaled, y_train, hyperparameter_tuning)
            rf_results = self.evaluate_model(rf_model, X_test_scaled, y_test, "RandomForest")
            self.save_model(rf_model, "random_forest", scaler)
            results['random_forest'] = rf_results
            self.models['random_forest'] = rf_model
            self.scalers['random_forest'] = scaler
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
        
        # Train XGBoost (if available)
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = self.train_xgboost(X_train_scaled, y_train, hyperparameter_tuning)
                if xgb_model is not None:
                    xgb_results = self.evaluate_model(xgb_model, X_test_scaled, y_test, "XGBoost")
                    self.save_model(xgb_model, "xgboost", scaler)
                    results['xgboost'] = xgb_results
                    self.models['xgboost'] = xgb_model
                    self.scalers['xgboost'] = scaler
            except Exception as e:
                logger.error(f"Error training XGBoost: {e}")
        
        # Save training results
        results_file = os.path.join(self.models_dir, "training_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.training_results = results
        logger.info("Baseline model training completed")
        
        return results
    
    def load_model(self, model_name):
        """
        Load a trained model.
        
        Args:
            model_name (str): Name of the model to load
        
        Returns:
            tuple: (model, scaler, metadata)
        """
        model_file = os.path.join(self.models_dir, f"{model_name}_model.joblib")
        scaler_file = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
        metadata_file = os.path.join(self.models_dir, f"{model_name}_metadata.json")
        encoder_file = os.path.join(self.models_dir, f"{model_name}_label_encoder.joblib")
        
        try:
            model = joblib.load(model_file)
            scaler = joblib.load(scaler_file) if os.path.exists(scaler_file) else None
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load label encoder if exists
            if os.path.exists(encoder_file):
                label_encoder = joblib.load(encoder_file)
                self.label_encoder = label_encoder
            
            logger.info(f"Loaded model: {model_name}")
            return model, scaler, metadata
        
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None, None, None


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Baseline Model Training")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--features-file", type=str, help="Path to features file")
    parser.add_argument("--no-tuning", action="store_true", help="Skip hyperparameter tuning")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = BaselineTrainer(data_dir=args.data_dir, models_dir=args.models_dir)
    
    # Train models
    results = trainer.train_baseline_models(
        features_file=args.features_file,
        hyperparameter_tuning=not args.no_tuning
    )
    
    # Print results summary
    logger.info("\n" + "="*50)
    logger.info("TRAINING RESULTS SUMMARY")
    logger.info("="*50)
    
    for model_name, result in results.items():
        logger.info(f"\n{model_name.upper()}:")
        logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  Weighted F1: {result['weighted_f1']:.4f}")
        if result['roc_auc']:
            logger.info(f"  ROC AUC: {result['roc_auc']:.4f}")


if __name__ == "__main__":
    main()