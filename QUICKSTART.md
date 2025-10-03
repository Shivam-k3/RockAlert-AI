# RockAlert AI - Quick Start Guide

## 🚀 Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
git clone https://github.com/Shivam-k3/RockAlert-AI.git
cd RockAlert-AI
python setup_demo.py
```

### Option 2: Manual Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Shivam-k3/RockAlert-AI.git
cd RockAlert-AI
```

2. **Set up Python environment:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

3. **Generate sample data and train models:**
```bash
cd src
python sensor_ingest.py --mode synthetic --duration 10 --with-anomalies
python features.py --mode combined
python train_baseline.py
```

4. **Start the backend:**
```bash
python inference_service.py
```

5. **Start the frontend (new terminal):**
```bash
cd frontend
npm install
npm start
```

## 🌐 Access Points

- **Frontend Dashboard:** http://localhost:3000
- **Backend API:** http://localhost:8000
- **API Documentation:** http://localhost:8000/docs

## 🧪 Testing

Try the test buttons in the dashboard navbar:
- **Test Prediction:** Generates a sample prediction
- **Test Alert:** Sends a test warning/critical alert

## 📚 Full Documentation

See [README.md](README.md) for complete documentation.

## 🆘 Troubleshooting

### Common Issues

1. **Port already in use:**
   - Backend: Change port in `src/inference_service.py`
   - Frontend: Set `PORT=3001` environment variable

2. **Missing dependencies:**
   - Python: `pip install -r requirements.txt`
   - Node.js: `cd frontend && npm install`

3. **Model not found errors:**
   - Run: `cd src && python train_baseline.py`

### System Requirements

- Python 3.8+
- Node.js 14+
- 4GB RAM minimum
- Modern web browser

## 📞 Support

- GitHub Issues: [Report bugs](https://github.com/Shivam-k3/RockAlert-AI/issues)
- Documentation: Check README.md for detailed setup