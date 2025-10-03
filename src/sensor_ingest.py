"""
Sensor Data Ingestion Module
Reads vibration sensor data from CSV files or data streams and aligns with timestamps.
"""

import pandas as pd
import numpy as np
import os
import time
import json
import logging
from datetime import datetime, timedelta
import argparse
import csv
from typing import Optional, List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensorIngest:
    def __init__(self, data_dir="data/raw", sampling_rate=100):
        """
        Initialize sensor ingestion system.
        
        Args:
            data_dir (str): Directory to save sensor data
            sampling_rate (int): Sensor sampling rate in Hz
        """
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.sensor_data_file = os.path.join(data_dir, "sensor_data.csv")
        self.metadata_file = os.path.join(data_dir, "sensor_metadata.json")
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize CSV file with headers if it doesn't exist
        if not os.path.exists(self.sensor_data_file):
            self.init_csv_file()
        
        # Load or initialize metadata
        self.metadata = self.load_metadata()
    
    def init_csv_file(self):
        """Initialize CSV file with headers."""
        headers = [
            'timestamp', 'vibration_x', 'vibration_y', 'vibration_z',
            'acceleration_x', 'acceleration_y', 'acceleration_z',
            'temperature', 'humidity', 'pressure'
        ]
        
        with open(self.sensor_data_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
        
        logger.info(f"Initialized sensor data file: {self.sensor_data_file}")
    
    def load_metadata(self):
        """Load sensor metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "total_samples": 0,
                "start_time": None,
                "last_update": None,
                "sensors": {
                    "vibration": {"units": "m/s²", "range": [-10, 10]},
                    "acceleration": {"units": "m/s²", "range": [-50, 50]},
                    "temperature": {"units": "°C", "range": [-40, 85]},
                    "humidity": {"units": "%", "range": [0, 100]},
                    "pressure": {"units": "kPa", "range": [80, 120]}
                }
            }
    
    def save_metadata(self):
        """Save sensor metadata to file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def generate_synthetic_data(self, duration_minutes=10, with_anomalies=True):
        """
        Generate synthetic sensor data for testing.
        
        Args:
            duration_minutes (int): Duration in minutes
            with_anomalies (bool): Include anomalous readings
        """
        logger.info(f"Generating synthetic sensor data for {duration_minutes} minutes")
        
        # Calculate number of samples
        total_samples = duration_minutes * 60 * self.sampling_rate
        
        # Generate timestamps
        start_time = datetime.now()
        timestamps = [
            start_time + timedelta(seconds=i/self.sampling_rate)
            for i in range(total_samples)
        ]
        
        # Generate base sensor readings with realistic patterns
        data_rows = []
        
        for i, timestamp in enumerate(timestamps):
            # Time-based variations (daily cycles, etc.)
            time_factor = np.sin(2 * np.pi * i / (self.sampling_rate * 3600))  # Hourly cycle
            
            # Base vibration (usually low)
            vibration_base = 0.1
            vibration_noise = np.random.normal(0, 0.05, 3)
            vibration_x = vibration_base + vibration_noise[0] + 0.02 * time_factor
            vibration_y = vibration_base + vibration_noise[1] + 0.02 * time_factor
            vibration_z = vibration_base + vibration_noise[2] + 0.02 * time_factor
            
            # Add anomalies (potential rockfall indicators)
            if with_anomalies and np.random.random() < 0.001:  # 0.1% chance of anomaly
                anomaly_factor = np.random.uniform(5, 15)
                vibration_x *= anomaly_factor
                vibration_y *= anomaly_factor
                vibration_z *= anomaly_factor
                logger.info(f"Added anomaly at {timestamp}")
            
            # Acceleration (correlated with vibration)
            acc_factor = 2.0
            acceleration_x = vibration_x * acc_factor + np.random.normal(0, 0.1)
            acceleration_y = vibration_y * acc_factor + np.random.normal(0, 0.1)
            acceleration_z = vibration_z * acc_factor + np.random.normal(0, 0.1)
            
            # Environmental sensors
            temperature = 25 + 10 * time_factor + np.random.normal(0, 2)
            humidity = 60 + 20 * time_factor + np.random.normal(0, 5)
            pressure = 101.3 + 2 * time_factor + np.random.normal(0, 0.5)
            
            # Create data row
            row = [
                timestamp.isoformat(),
                round(vibration_x, 6),
                round(vibration_y, 6),
                round(vibration_z, 6),
                round(acceleration_x, 6),
                round(acceleration_y, 6),
                round(acceleration_z, 6),
                round(temperature, 2),
                round(humidity, 2),
                round(pressure, 2)
            ]
            
            data_rows.append(row)
        
        # Write to CSV file
        with open(self.sensor_data_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(data_rows)
        
        # Update metadata
        self.metadata["total_samples"] += total_samples
        if self.metadata["start_time"] is None:
            self.metadata["start_time"] = start_time.isoformat()
        self.metadata["last_update"] = datetime.now().isoformat()
        self.save_metadata()
        
        logger.info(f"Generated {total_samples} samples")
        return True
    
    def read_csv_file(self, csv_path, timestamp_column='timestamp'):
        """
        Read sensor data from external CSV file.
        
        Args:
            csv_path (str): Path to CSV file
            timestamp_column (str): Name of timestamp column
        """
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found: {csv_path}")
            return False
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)
            logger.info(f"Reading CSV file: {csv_path}")
            logger.info(f"Loaded {len(df)} rows")
            
            # Validate required columns
            required_columns = ['vibration_x', 'vibration_y', 'vibration_z']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.warning(f"Missing columns: {missing_columns}")
                # Fill missing columns with zeros
                for col in missing_columns:
                    df[col] = 0.0
            
            # Process timestamps
            if timestamp_column in df.columns:
                df['timestamp'] = pd.to_datetime(df[timestamp_column])
            else:
                # Generate timestamps if not present
                start_time = datetime.now()
                df['timestamp'] = [
                    start_time + timedelta(seconds=i/self.sampling_rate)
                    for i in range(len(df))
                ]
            
            # Append to main CSV file
            df.to_csv(self.sensor_data_file, mode='a', header=False, index=False)
            
            # Update metadata
            self.metadata["total_samples"] += len(df)
            if self.metadata["start_time"] is None:
                self.metadata["start_time"] = df['timestamp'].min().isoformat()
            self.metadata["last_update"] = datetime.now().isoformat()
            self.save_metadata()
            
            logger.info(f"Successfully imported {len(df)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            return False
    
    def get_latest_data(self, duration_seconds=60):
        """
        Get the latest sensor data within specified duration.
        
        Args:
            duration_seconds (int): Duration in seconds
        
        Returns:
            pandas.DataFrame: Latest sensor data
        """
        try:
            # Read the CSV file
            df = pd.read_csv(self.sensor_data_file)
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert timestamp column to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter for latest data
            cutoff_time = datetime.now() - timedelta(seconds=duration_seconds)
            latest_data = df[df['timestamp'] >= cutoff_time]
            
            return latest_data.sort_values('timestamp')
            
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
    
    def get_data_statistics(self):
        """
        Get statistical summary of sensor data.
        
        Returns:
            dict: Statistics summary
        """
        try:
            df = pd.read_csv(self.sensor_data_file)
            
            if df.empty:
                return {"error": "No data available"}
            
            # Select numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            stats = {}
            for col in numeric_columns:
                stats[col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "count": int(df[col].count())
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {"error": str(e)}
    
    def simulate_realtime_stream(self, duration_minutes=5):
        """
        Simulate real-time sensor data streaming.
        
        Args:
            duration_minutes (int): Duration to stream in minutes
        """
        logger.info(f"Starting real-time sensor simulation for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        sample_interval = 1.0 / self.sampling_rate
        
        try:
            while time.time() < end_time:
                current_time = datetime.now()
                
                # Generate single sample
                vibration_x = 0.1 + np.random.normal(0, 0.05)
                vibration_y = 0.1 + np.random.normal(0, 0.05)
                vibration_z = 0.1 + np.random.normal(0, 0.05)
                
                # Add random anomalies
                if np.random.random() < 0.001:
                    anomaly_factor = np.random.uniform(5, 10)
                    vibration_x *= anomaly_factor
                    vibration_y *= anomaly_factor
                    vibration_z *= anomaly_factor
                
                acceleration_x = vibration_x * 2 + np.random.normal(0, 0.1)
                acceleration_y = vibration_y * 2 + np.random.normal(0, 0.1)
                acceleration_z = vibration_z * 2 + np.random.normal(0, 0.1)
                
                temperature = 25 + np.random.normal(0, 2)
                humidity = 60 + np.random.normal(0, 5)
                pressure = 101.3 + np.random.normal(0, 0.5)
                
                # Save to CSV
                row = [
                    current_time.isoformat(),
                    round(vibration_x, 6),
                    round(vibration_y, 6),
                    round(vibration_z, 6),
                    round(acceleration_x, 6),
                    round(acceleration_y, 6),
                    round(acceleration_z, 6),
                    round(temperature, 2),
                    round(humidity, 2),
                    round(pressure, 2)
                ]
                
                with open(self.sensor_data_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)
                
                # Update metadata
                self.metadata["total_samples"] += 1
                self.metadata["last_update"] = current_time.isoformat()
                
                time.sleep(sample_interval)
        
        finally:
            self.save_metadata()
            logger.info("Real-time simulation completed")


def main():
    parser = argparse.ArgumentParser(description="Sensor Data Ingestion")
    parser.add_argument("--mode", choices=["synthetic", "csv", "realtime"], required=True,
                       help="Data ingestion mode")
    parser.add_argument("--csv-path", type=str, help="Path to CSV file (if mode=csv)")
    parser.add_argument("--duration", type=int, default=10, help="Duration in minutes")
    parser.add_argument("--sampling-rate", type=int, default=100, help="Sampling rate in Hz")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Data directory")
    parser.add_argument("--with-anomalies", action="store_true", help="Include anomalies in synthetic data")
    
    args = parser.parse_args()
    
    # Create sensor ingestion instance
    sensor_ingest = SensorIngest(data_dir=args.data_dir, sampling_rate=args.sampling_rate)
    
    if args.mode == "synthetic":
        success = sensor_ingest.generate_synthetic_data(
            duration_minutes=args.duration,
            with_anomalies=args.with_anomalies
        )
    elif args.mode == "csv":
        if not args.csv_path:
            logger.error("CSV path is required when mode=csv")
            return
        success = sensor_ingest.read_csv_file(args.csv_path)
    elif args.mode == "realtime":
        sensor_ingest.simulate_realtime_stream(duration_minutes=args.duration)
        success = True
    
    if success:
        logger.info("Sensor ingestion completed successfully")
        
        # Print statistics
        stats = sensor_ingest.get_data_statistics()
        logger.info("Data statistics:")
        for sensor, values in stats.items():
            if isinstance(values, dict):
                logger.info(f"  {sensor}: mean={values['mean']:.4f}, std={values['std']:.4f}")
    else:
        logger.error("Sensor ingestion failed")


if __name__ == "__main__":
    main()