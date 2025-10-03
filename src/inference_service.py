"""
FastAPI Inference Service
Provides REST API and WebSocket endpoints for rockfall prediction and real-time alerts.
"""

import os
import json
import logging
import asyncio
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data models
class PredictionRequest(BaseModel):
    features: List[float]
    feature_names: Optional[List[str]] = None
    timestamp: Optional[str] = None

class PredictionResponse(BaseModel):
    probability: float
    alert_level: str
    confidence: float
    timestamp: str
    model_used: str

class AlertMessage(BaseModel):
    alert_level: str
    probability: float
    message: str
    timestamp: str
    location: Optional[Dict[str, float]] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: List[str]
    uptime: str

class InferenceService:
    def __init__(self, models_dir="models", data_dir="data"):
        """
        Initialize inference service.
        
        Args:
            models_dir (str): Directory containing trained models
            data_dir (str): Data directory
        """
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_metadata = {}
        
        # Alert thresholds
        self.alert_thresholds = {
            "safe": 0.3,
            "warning": 0.7,
            "critical": 1.0
        }
        
        # WebSocket connections
        self.websocket_connections: List[WebSocket] = []
        
        # Service start time
        self.start_time = datetime.now()
        
        # Load models
        self.load_models()
        
        # Default model to use
        self.default_model = "random_forest"
        if self.default_model not in self.models and self.models:
            self.default_model = list(self.models.keys())[0]
    
    def load_models(self):
        """Load trained models from disk."""
        if not os.path.exists(self.models_dir):
            logger.warning(f"Models directory not found: {self.models_dir}")
            return
        
        # Look for model files
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith("_model.joblib")]
        
        for model_file in model_files:
            model_name = model_file.replace("_model.joblib", "")
            
            try:
                # Load model
                model_path = os.path.join(self.models_dir, model_file)
                model = joblib.load(model_path)
                self.models[model_name] = model
                
                # Load scaler
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    self.scalers[model_name] = scaler
                
                # Load label encoder
                encoder_path = os.path.join(self.models_dir, f"{model_name}_label_encoder.joblib")
                if os.path.exists(encoder_path):
                    encoder = joblib.load(encoder_path)
                    self.label_encoders[model_name] = encoder
                
                # Load metadata
                metadata_path = os.path.join(self.models_dir, f"{model_name}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    self.model_metadata[model_name] = metadata
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
        
        if not self.models:
            logger.warning("No models loaded. Using dummy predictions.")
    
    def get_alert_level(self, probability: float) -> str:
        """
        Determine alert level based on probability.
        
        Args:
            probability (float): Rockfall probability
        
        Returns:
            str: Alert level
        """
        if probability >= self.alert_thresholds["warning"]:
            return "critical"
        elif probability >= self.alert_thresholds["safe"]:
            return "warning"
        else:
            return "safe"
    
    def predict(self, features: List[float], model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Make prediction using specified model.
        
        Args:
            features (List[float]): Input features
            model_name (str): Model name to use
        
        Returns:
            dict: Prediction results
        """
        if model_name is None:
            model_name = self.default_model
        
        # Check if model exists
        if model_name not in self.models:
            if not self.models:
                # No models loaded, return dummy prediction
                logger.warning("No models available, returning dummy prediction")
                probability = np.random.uniform(0, 0.3)  # Mostly safe predictions
                return {
                    "probability": float(probability),
                    "alert_level": self.get_alert_level(probability),
                    "confidence": 0.5,
                    "model_used": "dummy"
                }
            else:
                # Use default model
                model_name = self.default_model
        
        try:
            # Get model and scaler
            model = self.models[model_name]
            scaler = self.scalers.get(model_name)
            label_encoder = self.label_encoders.get(model_name)
            
            # Prepare features
            features_array = np.array(features).reshape(1, -1)
            
            # Scale features if scaler available
            if scaler is not None:
                features_array = scaler.transform(features_array)
            
            # Make prediction
            prediction = model.predict(features_array)[0]
            probabilities = model.predict_proba(features_array)[0]
            
            # Handle label encoding for XGBoost
            if label_encoder is not None:
                prediction = label_encoder.inverse_transform([prediction])[0]
                # Map probabilities to original labels
                label_to_prob = dict(zip(label_encoder.classes_, probabilities))
                probability = label_to_prob.get("critical", 0.0) + 0.5 * label_to_prob.get("warning", 0.0)
            else:
                # For models that directly predict alert levels
                if hasattr(model, "classes_"):
                    classes = model.classes_
                    if "critical" in classes:
                        critical_idx = list(classes).index("critical")
                        warning_idx = list(classes).index("warning") if "warning" in classes else -1
                        probability = probabilities[critical_idx]
                        if warning_idx >= 0:
                            probability += 0.5 * probabilities[warning_idx]
                    else:
                        probability = np.max(probabilities)
                else:
                    probability = np.max(probabilities)
            
            # Ensure probability is in [0, 1] range
            probability = np.clip(probability, 0.0, 1.0)
            
            # Calculate confidence (max probability)
            confidence = float(np.max(probabilities))
            
            # Determine alert level
            alert_level = self.get_alert_level(probability)
            
            return {
                "probability": float(probability),
                "alert_level": alert_level,
                "confidence": confidence,
                "model_used": model_name
            }
            
        except Exception as e:
            logger.error(f"Error making prediction with {model_name}: {e}")
            # Return safe prediction as fallback
            return {
                "probability": 0.1,
                "alert_level": "safe",
                "confidence": 0.0,
                "model_used": f"{model_name}_error"
            }
    
    async def broadcast_alert(self, alert: AlertMessage):
        """
        Broadcast alert to all connected WebSocket clients.
        
        Args:
            alert (AlertMessage): Alert message to broadcast
        """
        if not self.websocket_connections:
            return
        
        message = alert.dict()
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            self.websocket_connections.remove(ws)
    
    async def monitor_and_alert(self):
        """
        Background task to monitor data and send alerts.
        This would typically read from live data streams.
        """
        logger.info("Starting monitoring and alert service")
        
        while True:
            try:
                # Simulate reading latest sensor data
                # In practice, this would read from actual sensors
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Generate dummy features for demonstration
                dummy_features = np.random.normal(0, 1, 34).tolist()  # 34 features total
                
                # Add occasional anomalies
                if np.random.random() < 0.1:  # 10% chance of anomaly
                    # Simulate high vibration
                    for i in range(8, 15):  # Vibration feature indices
                        dummy_features[i] *= np.random.uniform(3, 8)
                
                # Make prediction
                prediction = self.predict(dummy_features)
                
                # Send alert if warning or critical
                if prediction["alert_level"] in ["warning", "critical"]:
                    alert = AlertMessage(
                        alert_level=prediction["alert_level"],
                        probability=prediction["probability"],
                        message=f"{prediction['alert_level'].upper()} alert: Rockfall probability {prediction['probability']:.2f}",
                        timestamp=datetime.now().isoformat(),
                        location={"lat": 45.0, "lng": -110.0}  # Dummy location
                    )
                    
                    await self.broadcast_alert(alert)
                    logger.info(f"Sent {prediction['alert_level']} alert to {len(self.websocket_connections)} clients")
                
            except Exception as e:
                logger.error(f"Error in monitoring service: {e}")
                await asyncio.sleep(5)

# Create FastAPI app
app = FastAPI(
    title="Rockfall Prediction API",
    description="AI-based rockfall prediction and alert system for open-pit mines",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize inference service
inference_service = InferenceService()

@app.on_event("startup")
async def startup_event():
    """Start background monitoring task."""
    asyncio.create_task(inference_service.monitor_and_alert())

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    uptime = datetime.now() - inference_service.start_time
    uptime_str = str(uptime).split(".")[0]  # Remove microseconds
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=list(inference_service.models.keys()),
        uptime=uptime_str
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make rockfall prediction based on input features.
    
    Args:
        request (PredictionRequest): Prediction request
    
    Returns:
        PredictionResponse: Prediction results
    """
    try:
        # Validate features
        if not request.features:
            raise HTTPException(status_code=400, detail="Features cannot be empty")
        
        # Make prediction
        prediction = inference_service.predict(request.features)
        
        return PredictionResponse(
            probability=prediction["probability"],
            alert_level=prediction["alert_level"],
            confidence=prediction["confidence"],
            timestamp=request.timestamp or datetime.now().isoformat(),
            model_used=prediction["model_used"]
        )
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "available_models": list(inference_service.models.keys()),
        "default_model": inference_service.default_model,
        "model_metadata": inference_service.model_metadata
    }

@app.get("/alert-thresholds")
async def get_alert_thresholds():
    """Get current alert thresholds."""
    return inference_service.alert_thresholds

@app.post("/alert-thresholds")
async def update_alert_thresholds(thresholds: Dict[str, float]):
    """Update alert thresholds."""
    # Validate thresholds
    required_keys = {"safe", "warning", "critical"}
    if not required_keys.issubset(thresholds.keys()):
        raise HTTPException(status_code=400, detail=f"Missing required keys: {required_keys}")
    
    if not (0 <= thresholds["safe"] <= thresholds["warning"] <= thresholds["critical"] <= 1):
        raise HTTPException(status_code=400, detail="Invalid threshold values")
    
    inference_service.alert_thresholds.update(thresholds)
    return {"message": "Thresholds updated successfully", "thresholds": inference_service.alert_thresholds}

@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts.
    """
    await websocket.accept()
    inference_service.websocket_connections.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(inference_service.websocket_connections)}")
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        inference_service.websocket_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total: {len(inference_service.websocket_connections)}")

@app.get("/statistics")
async def get_statistics():
    """Get system statistics."""
    return {
        "connected_clients": len(inference_service.websocket_connections),
        "models_loaded": len(inference_service.models),
        "uptime": str(datetime.now() - inference_service.start_time).split(".")[0],
        "alert_thresholds": inference_service.alert_thresholds
    }

# Test endpoints for development
@app.post("/test/predict")
async def test_predict():
    """Test prediction with dummy data."""
    # Generate dummy features
    dummy_features = np.random.normal(0, 1, 34).tolist()
    
    request = PredictionRequest(features=dummy_features)
    return await predict(request)

@app.post("/test/alert")
async def test_alert(alert_level: str = "warning"):
    """Send test alert via WebSocket."""
    if alert_level not in ["safe", "warning", "critical"]:
        raise HTTPException(status_code=400, detail="Invalid alert level")
    
    alert = AlertMessage(
        alert_level=alert_level,
        probability=0.5,
        message=f"Test {alert_level} alert",
        timestamp=datetime.now().isoformat(),
        location={"lat": 45.0, "lng": -110.0}
    )
    
    await inference_service.broadcast_alert(alert)
    return {"message": f"Test {alert_level} alert sent to {len(inference_service.websocket_connections)} clients"}

def main():
    """Run the FastAPI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Rockfall Prediction API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--models-dir", type=str, default="models", help="Models directory")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    # Update global inference service paths
    global inference_service
    inference_service = InferenceService(models_dir=args.models_dir, data_dir=args.data_dir)
    
    # Run server
    uvicorn.run(
        "inference_service:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()