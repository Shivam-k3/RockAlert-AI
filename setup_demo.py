#!/usr/bin/env python3
"""
Setup and Demo Script for RockWatch AI System
Generates sample data, trains models, and demonstrates the complete system.
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description, check=True):
    """Run a shell command with error handling."""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            logger.info(f"Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Error: {e}")
        if e.stderr:
            logger.error(f"Stderr: {e.stderr}")
        return False

def setup_python_environment():
    """Set up Python virtual environment and install dependencies."""
    logger.info("Setting up Python environment...")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        if not run_command("python -m venv venv", "Creating virtual environment"):
            return False
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix-like systems
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    logger.info("Python environment setup complete")
    return True

def generate_sample_data():
    """Generate sample sensor and camera data."""
    logger.info("Generating sample data...")
    
    # Change to src directory
    os.chdir("src")
    
    try:
        # Generate synthetic sensor data
        if not run_command("python sensor_ingest.py --mode synthetic --duration 5 --with-anomalies", 
                          "Generating synthetic sensor data"):
            return False
        
        # Generate dummy camera frames (using OpenCV to create sample images)
        logger.info("Creating sample camera frames...")
        import cv2
        import numpy as np
        
        frames_dir = "../data/raw/frames"
        os.makedirs(frames_dir, exist_ok=True)
        
        for i in range(10):
            # Create a sample image with some noise and patterns
            img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
            
            # Add some geometric patterns to simulate mine walls
            cv2.rectangle(img, (100, 100), (500, 400), (80, 80, 80), -1)
            cv2.rectangle(img, (150, 150), (450, 350), (120, 120, 120), -1)
            
            # Add some noise for texture
            noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
            img = cv2.add(img, noise)
            
            filename = os.path.join(frames_dir, f"sample_frame_{i:03d}.jpg")
            cv2.imwrite(filename, img)
        
        logger.info("Sample camera frames created")
        
    finally:
        os.chdir("..")
    
    return True

def extract_features():
    """Extract features from sample data."""
    logger.info("Extracting features...")
    
    os.chdir("src")
    try:
        if not run_command("python features.py --mode combined", "Extracting combined features"):
            return False
    finally:
        os.chdir("..")
    
    return True

def train_models():
    """Train baseline models."""
    logger.info("Training baseline models...")
    
    os.chdir("src")
    try:
        if not run_command("python train_baseline.py --no-tuning", "Training baseline models"):
            return False
    finally:
        os.chdir("..")
    
    return True

def setup_frontend():
    """Set up React frontend."""
    logger.info("Setting up React frontend...")
    
    os.chdir("frontend")
    try:
        # Check if node_modules exists
        if not os.path.exists("node_modules"):
            if not run_command("npm install", "Installing Node.js dependencies"):
                return False
        
        logger.info("Frontend setup complete")
    finally:
        os.chdir("..")
    
    return True

def start_backend():
    """Start the FastAPI backend server."""
    logger.info("Starting backend server...")
    
    os.chdir("src")
    try:
        # Start backend in background
        if os.name == 'nt':  # Windows
            subprocess.Popen(["python", "inference_service.py"], shell=True)
        else:  # Unix-like systems
            subprocess.Popen(["python3", "inference_service.py"])
        
        logger.info("Backend server started on http://localhost:8000")
        time.sleep(3)  # Give server time to start
        
    finally:
        os.chdir("..")

def start_frontend():
    """Start the React frontend."""
    logger.info("Starting frontend...")
    
    os.chdir("frontend")
    try:
        if os.name == 'nt':  # Windows
            subprocess.Popen(["npm", "start"], shell=True)
        else:  # Unix-like systems
            subprocess.Popen(["npm", "start"])
        
        logger.info("Frontend started on http://localhost:3000")
        
    finally:
        os.chdir("..")

def main():
    """Main setup and demo function."""
    logger.info("="*60)
    logger.info("RockWatch AI System Setup")
    logger.info("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src") or not os.path.exists("frontend"):
        logger.error("Please run this script from the project root directory")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    try:
        # Setup steps
        if not setup_python_environment():
            logger.error("Failed to setup Python environment")
            sys.exit(1)
        
        if not generate_sample_data():
            logger.error("Failed to generate sample data")
            sys.exit(1)
        
        if not extract_features():
            logger.error("Failed to extract features")
            sys.exit(1)
        
        if not train_models():
            logger.error("Failed to train models")
            sys.exit(1)
        
        if not setup_frontend():
            logger.error("Failed to setup frontend")
            logger.warning("You can still run the backend API")
        
        # Start services
        logger.info("\n" + "="*60)
        logger.info("Starting Services")
        logger.info("="*60)
        
        start_backend()
        
        try:
            start_frontend()
        except Exception as e:
            logger.warning(f"Could not start frontend: {e}")
            logger.info("Frontend requires Node.js and npm to be installed")
        
        logger.info("\n" + "="*60)
        logger.info("Setup Complete!")
        logger.info("="*60)
        logger.info("Backend API: http://localhost:8000")
        logger.info("Frontend Dashboard: http://localhost:3000")
        logger.info("API Documentation: http://localhost:8000/docs")
        logger.info("\nPress Ctrl+C to stop all services")
        
        # Keep script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nShutting down services...")
            
    except Exception as e:
        logger.error(f"Setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()