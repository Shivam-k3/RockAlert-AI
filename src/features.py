"""
Feature Extraction Module
Computes features from camera frames (optical flow) and sensor data (statistical features).
"""

import cv2
import numpy as np
import pandas as pd
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, data_dir="data", processed_dir="data/processed"):
        """
        Initialize feature extraction system.
        
        Args:
            data_dir (str): Raw data directory
            processed_dir (str): Processed data directory
        """
        self.data_dir = data_dir
        self.processed_dir = processed_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        
        # Feature storage files
        self.vision_features_file = os.path.join(processed_dir, "vision_features.npy")
        self.sensor_features_file = os.path.join(processed_dir, "sensor_features.npy")
        self.feature_metadata_file = os.path.join(processed_dir, "feature_metadata.json")
        
        # Initialize optical flow detector
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Previous frame for optical flow
        self.prev_frame = None
        
        # Feature metadata
        self.feature_metadata = self.load_feature_metadata()
    
    def load_feature_metadata(self):
        """Load feature metadata from file."""
        if os.path.exists(self.feature_metadata_file):
            with open(self.feature_metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "vision_features": {
                    "feature_names": [
                        "optical_flow_magnitude_mean",
                        "optical_flow_magnitude_std",
                        "optical_flow_magnitude_max",
                        "optical_flow_direction_variance",
                        "motion_density",
                        "edge_density",
                        "brightness_mean",
                        "brightness_std"
                    ],
                    "feature_count": 8,
                    "last_update": None
                },
                "sensor_features": {
                    "feature_names": [
                        "vibration_x_mean", "vibration_x_std", "vibration_x_max",
                        "vibration_y_mean", "vibration_y_std", "vibration_y_max",
                        "vibration_z_mean", "vibration_z_std", "vibration_z_max",
                        "vibration_magnitude_mean", "vibration_magnitude_std",
                        "acceleration_x_mean", "acceleration_x_std", "acceleration_x_max",
                        "acceleration_y_mean", "acceleration_y_std", "acceleration_y_max",
                        "acceleration_z_mean", "acceleration_z_std", "acceleration_z_max",
                        "acceleration_magnitude_mean", "acceleration_magnitude_std",
                        "temperature_mean", "temperature_std",
                        "humidity_mean", "humidity_std",
                        "pressure_mean", "pressure_std"
                    ],
                    "feature_count": 26,
                    "last_update": None
                }
            }
    
    def save_feature_metadata(self):
        """Save feature metadata to file."""
        with open(self.feature_metadata_file, 'w') as f:
            json.dump(self.feature_metadata, f, indent=2)
    
    def extract_optical_flow_features(self, frame1, frame2):
        """
        Extract optical flow features between two frames.
        
        Args:
            frame1 (np.ndarray): Previous frame
            frame2 (np.ndarray): Current frame
        
        Returns:
            dict: Optical flow features
        """
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowPyrLK(gray1, gray2, **self.flow_params)
        
        # Calculate flow magnitude and direction
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        direction = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Extract features
        features = {
            "optical_flow_magnitude_mean": np.mean(magnitude),
            "optical_flow_magnitude_std": np.std(magnitude),
            "optical_flow_magnitude_max": np.max(magnitude),
            "optical_flow_direction_variance": np.var(direction),
            "motion_density": np.sum(magnitude > 1.0) / magnitude.size
        }
        
        return features
    
    def extract_frame_features(self, frame):
        """
        Extract additional features from a single frame.
        
        Args:
            frame (np.ndarray): Input frame
        
        Returns:
            dict: Frame features
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Brightness statistics
        brightness_mean = np.mean(gray)
        brightness_std = np.std(gray)
        
        features = {
            "edge_density": edge_density,
            "brightness_mean": brightness_mean,
            "brightness_std": brightness_std
        }
        
        return features
    
    def process_vision_data(self, frames_dir=None):
        """
        Process camera frames and extract vision features.
        
        Args:
            frames_dir (str): Directory containing frames
        
        Returns:
            np.ndarray: Vision features array
        """
        if frames_dir is None:
            frames_dir = os.path.join(self.raw_dir, "frames")
        
        if not os.path.exists(frames_dir):
            logger.warning(f"Frames directory not found: {frames_dir}")
            return np.array([])
        
        # Get list of frame files
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        
        if len(frame_files) < 2:
            logger.warning("Need at least 2 frames for optical flow computation")
            return np.array([])
        
        logger.info(f"Processing {len(frame_files)} frames for vision features")
        
        vision_features = []
        prev_frame = None
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            frame = cv2.imread(frame_path)
            
            if frame is None:
                logger.warning(f"Could not read frame: {frame_path}")
                continue
            
            # Extract single frame features
            frame_features = self.extract_frame_features(frame)
            
            # Extract optical flow features (if previous frame exists)
            if prev_frame is not None:
                flow_features = self.extract_optical_flow_features(prev_frame, frame)
                frame_features.update(flow_features)
            else:
                # For first frame, set optical flow features to zero
                flow_features = {
                    "optical_flow_magnitude_mean": 0.0,
                    "optical_flow_magnitude_std": 0.0,
                    "optical_flow_magnitude_max": 0.0,
                    "optical_flow_direction_variance": 0.0,
                    "motion_density": 0.0
                }
                frame_features.update(flow_features)
            
            # Convert to feature vector
            feature_vector = [
                frame_features["optical_flow_magnitude_mean"],
                frame_features["optical_flow_magnitude_std"],
                frame_features["optical_flow_magnitude_max"],
                frame_features["optical_flow_direction_variance"],
                frame_features["motion_density"],
                frame_features["edge_density"],
                frame_features["brightness_mean"],
                frame_features["brightness_std"]
            ]
            
            vision_features.append(feature_vector)
            prev_frame = frame
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(frame_files)} frames")
        
        vision_features = np.array(vision_features)
        
        # Save features
        np.save(self.vision_features_file, vision_features)
        self.feature_metadata["vision_features"]["last_update"] = datetime.now().isoformat()
        self.save_feature_metadata()
        
        logger.info(f"Extracted vision features: {vision_features.shape}")
        return vision_features
    
    def extract_sensor_window_features(self, data_window, window_size_seconds=30):
        """
        Extract statistical features from a sensor data window.
        
        Args:
            data_window (pd.DataFrame): Sensor data window
            window_size_seconds (int): Window size in seconds
        
        Returns:
            dict: Sensor features
        """
        features = {}
        
        # Vibration features
        for axis in ['x', 'y', 'z']:
            col = f'vibration_{axis}'
            if col in data_window.columns:
                features[f'vibration_{axis}_mean'] = data_window[col].mean()
                features[f'vibration_{axis}_std'] = data_window[col].std()
                features[f'vibration_{axis}_max'] = data_window[col].max()
            else:
                features[f'vibration_{axis}_mean'] = 0.0
                features[f'vibration_{axis}_std'] = 0.0
                features[f'vibration_{axis}_max'] = 0.0
        
        # Vibration magnitude
        if all(f'vibration_{axis}' in data_window.columns for axis in ['x', 'y', 'z']):
            vibration_magnitude = np.sqrt(
                data_window['vibration_x']**2 + 
                data_window['vibration_y']**2 + 
                data_window['vibration_z']**2
            )
            features['vibration_magnitude_mean'] = vibration_magnitude.mean()
            features['vibration_magnitude_std'] = vibration_magnitude.std()
        else:
            features['vibration_magnitude_mean'] = 0.0
            features['vibration_magnitude_std'] = 0.0
        
        # Acceleration features
        for axis in ['x', 'y', 'z']:
            col = f'acceleration_{axis}'
            if col in data_window.columns:
                features[f'acceleration_{axis}_mean'] = data_window[col].mean()
                features[f'acceleration_{axis}_std'] = data_window[col].std()
                features[f'acceleration_{axis}_max'] = data_window[col].max()
            else:
                features[f'acceleration_{axis}_mean'] = 0.0
                features[f'acceleration_{axis}_std'] = 0.0
                features[f'acceleration_{axis}_max'] = 0.0
        
        # Acceleration magnitude
        if all(f'acceleration_{axis}' in data_window.columns for axis in ['x', 'y', 'z']):
            acceleration_magnitude = np.sqrt(
                data_window['acceleration_x']**2 + 
                data_window['acceleration_y']**2 + 
                data_window['acceleration_z']**2
            )
            features['acceleration_magnitude_mean'] = acceleration_magnitude.mean()
            features['acceleration_magnitude_std'] = acceleration_magnitude.std()
        else:
            features['acceleration_magnitude_mean'] = 0.0
            features['acceleration_magnitude_std'] = 0.0
        
        # Environmental features
        for sensor in ['temperature', 'humidity', 'pressure']:
            if sensor in data_window.columns:
                features[f'{sensor}_mean'] = data_window[sensor].mean()
                features[f'{sensor}_std'] = data_window[sensor].std()
            else:
                features[f'{sensor}_mean'] = 0.0
                features[f'{sensor}_std'] = 0.0
        
        return features
    
    def process_sensor_data(self, sensor_file=None, window_size_seconds=30):
        """
        Process sensor data and extract statistical features using sliding windows.
        
        Args:
            sensor_file (str): Path to sensor data CSV file
            window_size_seconds (int): Window size in seconds
        
        Returns:
            np.ndarray: Sensor features array
        """
        if sensor_file is None:
            sensor_file = os.path.join(self.raw_dir, "sensor_data.csv")
        
        if not os.path.exists(sensor_file):
            logger.warning(f"Sensor data file not found: {sensor_file}")
            return np.array([])
        
        # Read sensor data
        try:
            df = pd.read_csv(sensor_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
        except Exception as e:
            logger.error(f"Error reading sensor data: {e}")
            return np.array([])
        
        if df.empty:
            logger.warning("Sensor data is empty")
            return np.array([])
        
        logger.info(f"Processing {len(df)} sensor samples with {window_size_seconds}s windows")
        
        # Calculate window parameters
        time_span = (df['timestamp'].max() - df['timestamp'].min()).total_seconds()
        window_step = window_size_seconds / 2  # 50% overlap
        num_windows = int(time_span / window_step) - 1
        
        sensor_features = []
        start_time = df['timestamp'].min()
        
        for i in range(num_windows):
            window_start = start_time + timedelta(seconds=i * window_step)
            window_end = window_start + timedelta(seconds=window_size_seconds)
            
            # Extract window data
            window_data = df[
                (df['timestamp'] >= window_start) & 
                (df['timestamp'] < window_end)
            ]
            
            if len(window_data) < 10:  # Skip windows with too few samples
                continue
            
            # Extract features from window
            window_features = self.extract_sensor_window_features(window_data, window_size_seconds)
            
            # Convert to feature vector
            feature_vector = [
                window_features["vibration_x_mean"],
                window_features["vibration_x_std"],
                window_features["vibration_x_max"],
                window_features["vibration_y_mean"],
                window_features["vibration_y_std"],
                window_features["vibration_y_max"],
                window_features["vibration_z_mean"],
                window_features["vibration_z_std"],
                window_features["vibration_z_max"],
                window_features["vibration_magnitude_mean"],
                window_features["vibration_magnitude_std"],
                window_features["acceleration_x_mean"],
                window_features["acceleration_x_std"],
                window_features["acceleration_x_max"],
                window_features["acceleration_y_mean"],
                window_features["acceleration_y_std"],
                window_features["acceleration_y_max"],
                window_features["acceleration_z_mean"],
                window_features["acceleration_z_std"],
                window_features["acceleration_z_max"],
                window_features["acceleration_magnitude_mean"],
                window_features["acceleration_magnitude_std"],
                window_features["temperature_mean"],
                window_features["temperature_std"],
                window_features["humidity_mean"],
                window_features["humidity_std"],
                window_features["pressure_mean"],
                window_features["pressure_std"]
            ]
            
            sensor_features.append(feature_vector)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{num_windows} windows")
        
        sensor_features = np.array(sensor_features)
        
        # Save features
        np.save(self.sensor_features_file, sensor_features)
        self.feature_metadata["sensor_features"]["last_update"] = datetime.now().isoformat()
        self.save_feature_metadata()
        
        logger.info(f"Extracted sensor features: {sensor_features.shape}")
        return sensor_features
    
    def combine_features(self, vision_features=None, sensor_features=None):
        """
        Combine vision and sensor features into a single feature matrix.
        
        Args:
            vision_features (np.ndarray): Vision features
            sensor_features (np.ndarray): Sensor features
        
        Returns:
            np.ndarray: Combined features
        """
        if vision_features is None:
            if os.path.exists(self.vision_features_file):
                vision_features = np.load(self.vision_features_file)
            else:
                vision_features = np.array([])
        
        if sensor_features is None:
            if os.path.exists(self.sensor_features_file):
                sensor_features = np.load(self.sensor_features_file)
            else:
                sensor_features = np.array([])
        
        # Handle empty features
        if vision_features.size == 0 and sensor_features.size == 0:
            logger.warning("No features to combine")
            return np.array([])
        elif vision_features.size == 0:
            logger.info("Using sensor features only")
            return sensor_features
        elif sensor_features.size == 0:
            logger.info("Using vision features only")
            return vision_features
        
        # Align feature counts (take minimum)
        min_samples = min(len(vision_features), len(sensor_features))
        vision_aligned = vision_features[:min_samples]
        sensor_aligned = sensor_features[:min_samples]
        
        # Combine features
        combined_features = np.hstack([vision_aligned, sensor_aligned])
        
        # Save combined features
        combined_file = os.path.join(self.processed_dir, "combined_features.npy")
        np.save(combined_file, combined_features)
        
        logger.info(f"Combined features shape: {combined_features.shape}")
        return combined_features
    
    def get_feature_names(self):
        """
        Get all feature names.
        
        Returns:
            list: Feature names
        """
        vision_names = self.feature_metadata["vision_features"]["feature_names"]
        sensor_names = self.feature_metadata["sensor_features"]["feature_names"]
        return vision_names + sensor_names


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Feature Extraction")
    parser.add_argument("--mode", choices=["vision", "sensor", "combined"], required=True,
                       help="Feature extraction mode")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--window-size", type=int, default=30, help="Window size in seconds (sensor mode)")
    
    args = parser.parse_args()
    
    # Create feature extractor
    extractor = FeatureExtractor(data_dir=args.data_dir)
    
    if args.mode == "vision":
        vision_features = extractor.process_vision_data()
        logger.info(f"Vision features extracted: {vision_features.shape}")
    
    elif args.mode == "sensor":
        sensor_features = extractor.process_sensor_data(window_size_seconds=args.window_size)
        logger.info(f"Sensor features extracted: {sensor_features.shape}")
    
    elif args.mode == "combined":
        vision_features = extractor.process_vision_data()
        sensor_features = extractor.process_sensor_data(window_size_seconds=args.window_size)
        combined_features = extractor.combine_features(vision_features, sensor_features)
        logger.info(f"Combined features extracted: {combined_features.shape}")
    
    # Print feature names
    feature_names = extractor.get_feature_names()
    logger.info(f"Feature names ({len(feature_names)}): {feature_names}")


if __name__ == "__main__":
    main()