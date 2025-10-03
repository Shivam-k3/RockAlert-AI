# RockWatch AI - Rockfall Prediction and Alert System

An AI-based system for predicting and alerting about potential rockfall events in open-pit mines using computer vision, sensor data, and machine learning.

## 🎯 Overview

RockWatch AI combines multiple data sources to provide real-time monitoring and prediction of rockfall events:

- **Camera Analysis**: Optical flow detection and visual pattern recognition
- **Sensor Monitoring**: Vibration, acceleration, and environmental sensors
- **AI Prediction**: Random Forest and XGBoost models for risk assessment
- **Real-time Alerts**: WebSocket-based instant notifications
- **Interactive Dashboard**: React-based monitoring interface

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   AI Pipeline   │    │   Dashboard     │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Cameras       │───▶│ • Feature       │───▶│ • React App     │
│ • Vibration     │    │   Extraction    │    │ • Real-time     │
│ • Environment   │    │ • ML Models     │    │   Alerts        │
│ • Weather       │    │ • Predictions   │    │ • Charts & Maps │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
rockfall-prediction-system/
├── src/                    # Python backend
│   ├── camera_ingest.py   # Camera data ingestion
│   ├── sensor_ingest.py   # Sensor data ingestion
│   ├── features.py        # Feature extraction
│   ├── train_baseline.py  # Model training
│   └── inference_service.py # FastAPI service
├── frontend/              # React frontend
│   ├── src/components/    # React components
│   ├── public/           # Static files
│   └── package.json      # Dependencies
├── data/                 # Data storage
│   ├── raw/             # Raw sensor/camera data
│   └── processed/       # Processed features
├── models/              # Trained ML models
├── requirements.txt     # Python dependencies
└── setup_demo.py       # Setup script
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm (for frontend)
- Git

### Option 1: Automated Setup

Run the setup script to automatically configure everything:

```bash
python setup_demo.py
```

This will:
1. Create Python virtual environment
2. Install dependencies
3. Generate sample data
4. Train baseline models
5. Start both backend and frontend

### Option 2: Manual Setup

#### Backend Setup

1. **Create virtual environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Generate sample data:**
```bash
cd src
python sensor_ingest.py --mode synthetic --duration 10 --with-anomalies
```

4. **Extract features:**
```bash
python features.py --mode combined
```

5. **Train models:**
```bash
python train_baseline.py
```

6. **Start API server:**
```bash
python inference_service.py
```

#### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start development server:**
```bash
npm start
```

## 🖥️ API Endpoints

### Core Endpoints

- `GET /health` - System health check
- `POST /predict` - Make rockfall prediction
- `GET /models` - List available models
- `GET /statistics` - System statistics
- `WS /ws/alerts` - WebSocket for real-time alerts

### Test Endpoints

- `POST /test/predict` - Test prediction with dummy data
- `POST /test/alert` - Send test alert

### Example API Usage

```python
import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Make prediction
features = [0.1, 0.2, 0.3, ...]  # 34 feature values
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": features}
)
prediction = response.json()
print(f"Alert Level: {prediction['alert_level']}")
print(f"Probability: {prediction['probability']:.2f}")
```

## 🎛️ Dashboard Features

### Main Interface

- **System Status**: Real-time system health monitoring
- **Interactive Map**: Mine layout with camera positions and alert locations
- **Live Alerts Panel**: Color-coded alerts with timestamps
- **Sensor Charts**: Real-time vibration, temperature, and pressure data

### Alert Levels

- 🟢 **Safe** (< 30%): Normal conditions
- 🟡 **Warning** (30-70%): Elevated risk, increased monitoring
- 🔴 **Critical** (> 70%): High risk, immediate action required

## 🔧 Configuration

### Alert Thresholds

Update alert thresholds via API:

```python
import requests

new_thresholds = {
    "safe": 0.3,
    "warning": 0.7,
    "critical": 1.0
}

response = requests.post(
    "http://localhost:8000/alert-thresholds",
    json=new_thresholds
)
```

### Model Selection

The system automatically loads all trained models from the `models/` directory. You can specify which model to use for predictions.

## 📊 Data Flow

1. **Data Ingestion**:
   - Cameras capture frames at configurable intervals
   - Sensors record vibration, environmental data
   - All data timestamped and stored in `data/raw/`

2. **Feature Extraction**:
   - Optical flow analysis from camera frames
   - Statistical features from sensor time series
   - Combined feature vectors saved to `data/processed/`

3. **Model Training**:
   - Synthetic labels generated for demonstration
   - RandomForest and XGBoost models trained
   - Models saved with metadata in `models/`

4. **Real-time Inference**:
   - Features extracted from live data
   - Models predict rockfall probability
   - Alerts generated based on thresholds
   - WebSocket broadcasts to connected clients

## 🛠️ Development

### Adding New Features

1. **Sensor Types**: Extend `sensor_ingest.py` with new sensor handlers
2. **Vision Features**: Add new computer vision features in `features.py`
3. **ML Models**: Implement new models in `train_baseline.py`
4. **Frontend Components**: Add React components in `frontend/src/components/`

### Testing

```bash
# Backend tests
cd src
python -m pytest tests/

# Frontend tests
cd frontend
npm test
```

### Code Structure

- **Modular Design**: Each component is independent and well-documented
- **Configuration**: All settings externalized for easy modification
- **Error Handling**: Comprehensive error handling and logging
- **Scalability**: Designed for easy horizontal scaling

## 🔐 Security Considerations

- **API Security**: Add authentication for production use
- **Data Encryption**: Encrypt sensitive data in transit and at rest
- **Access Control**: Implement role-based access control
- **Network Security**: Use HTTPS and secure WebSocket connections

## 📈 Performance Optimization

- **Feature Caching**: Cache computed features for faster inference
- **Model Optimization**: Use model quantization for edge deployment
- **Database**: Consider PostgreSQL for production data storage
- **Load Balancing**: Use multiple inference service instances

## 🚨 Production Deployment

### Docker Deployment

```dockerfile
# Backend Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .
CMD ["python", "inference_service.py"]
```

### Environment Variables

```bash
# Production settings
export API_HOST=0.0.0.0
export API_PORT=8000
export MODELS_DIR=/app/models
export DATA_DIR=/app/data
export LOG_LEVEL=INFO
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the [Issues](https://github.com/your-repo/issues) page
- Review the API documentation at `http://localhost:8000/docs`
- Consult the inline code documentation

---

**Note**: This is a demonstration system. For production use in actual mining operations, additional safety measures, regulatory compliance, and extensive testing are required.