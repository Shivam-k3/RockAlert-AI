import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import MapView from './components/MapView';
import AlertsPanel from './components/AlertsPanel';
import ChartsPanel from './components/ChartsPanel';
import StatusPanel from './components/StatusPanel';
import './App.css';

function App() {
  const [alerts, setAlerts] = useState([]);
  const [sensorData, setSensorData] = useState([]);
  const [systemStatus, setSystemStatus] = useState({
    status: 'connecting',
    modelsLoaded: [],
    connectedClients: 0,
    uptime: '00:00:00'
  });
  const [websocket, setWebsocket] = useState(null);

  // WebSocket connection for real-time alerts
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket('ws://localhost:8000/ws/alerts');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setSystemStatus(prev => ({ ...prev, status: 'connected' }));
      };
      
      ws.onmessage = (event) => {
        const alertData = JSON.parse(event.data);
        console.log('Received alert:', alertData);
        
        // Add new alert to the list
        setAlerts(prev => [alertData, ...prev.slice(0, 49)]); // Keep last 50 alerts
      };
      
      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setSystemStatus(prev => ({ ...prev, status: 'disconnected' }));
        
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setSystemStatus(prev => ({ ...prev, status: 'error' }));
      };
      
      setWebsocket(ws);
    };

    connectWebSocket();

    return () => {
      if (websocket) {
        websocket.close();
      }
    };
  }, []);

  // Fetch system status periodically
  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const [healthResponse, statsResponse] = await Promise.all([
          fetch('/health'),
          fetch('/statistics')
        ]);
        
        const healthData = await healthResponse.json();
        const statsData = await statsResponse.json();
        
        setSystemStatus({
          status: healthData.status,
          modelsLoaded: healthData.models_loaded,
          connectedClients: statsData.connected_clients,
          uptime: healthData.uptime
        });
      } catch (error) {
        console.error('Error fetching system status:', error);
        setSystemStatus(prev => ({ ...prev, status: 'error' }));
      }
    };

    // Fetch immediately and then every 30 seconds
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  // Generate dummy sensor data for demo
  useEffect(() => {
    const generateDummyData = () => {
      const now = new Date();
      const newDataPoint = {
        timestamp: now.toISOString(),
        vibrationX: 0.1 + Math.random() * 0.2,
        vibrationY: 0.1 + Math.random() * 0.2,
        vibrationZ: 0.1 + Math.random() * 0.2,
        temperature: 25 + Math.random() * 10,
        humidity: 50 + Math.random() * 30,
        pressure: 101 + Math.random() * 2
      };
      
      setSensorData(prev => [newDataPoint, ...prev.slice(0, 99)]); // Keep last 100 points
    };

    // Generate initial data
    for (let i = 0; i < 20; i++) {
      setTimeout(() => generateDummyData(), i * 100);
    }

    // Continue generating data every 5 seconds
    const interval = setInterval(generateDummyData, 5000);
    return () => clearInterval(interval);
  }, []);

  const testPrediction = async () => {
    try {
      const response = await fetch('/test/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('Test prediction result:', result);
        
        // If it's a warning or critical, it might generate an alert
        if (result.alert_level !== 'safe') {
          // Simulate adding an alert
          const alert = {
            alert_level: result.alert_level,
            probability: result.probability,
            message: `Test prediction: ${result.alert_level} (${(result.probability * 100).toFixed(1)}%)`,
            timestamp: new Date().toISOString(),
            location: { lat: 45.0, lng: -110.0 }
          };
          setAlerts(prev => [alert, ...prev.slice(0, 49)]);
        }
      }
    } catch (error) {
      console.error('Error testing prediction:', error);
    }
  };

  const testAlert = async (alertLevel) => {
    try {
      const response = await fetch(`/test/alert?alert_level=${alertLevel}`, {
        method: 'POST',
      });
      
      if (response.ok) {
        console.log(`Test ${alertLevel} alert sent`);
      }
    } catch (error) {
      console.error('Error sending test alert:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Navbar 
        systemStatus={systemStatus}
        onTestPrediction={testPrediction}
        onTestAlert={testAlert}
      />
      
      <main className="container mx-auto px-4 py-6">
        {/* Status Panel */}
        <div className="mb-6">
          <StatusPanel systemStatus={systemStatus} />
        </div>
        
        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Map and Charts */}
          <div className="lg:col-span-2 space-y-6">
            {/* Map View */}
            <div className="bg-white rounded-lg shadow-md">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-800">Mine Overview</h2>
              </div>
              <MapView alerts={alerts} />
            </div>
            
            {/* Charts */}
            <div className="bg-white rounded-lg shadow-md">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-800">Sensor Data Trends</h2>
              </div>
              <ChartsPanel sensorData={sensorData} />
            </div>
          </div>
          
          {/* Right Column - Alerts */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow-md">
              <div className="p-4 border-b border-gray-200">
                <h2 className="text-lg font-semibold text-gray-800">
                  Live Alerts
                  <span className="ml-2 text-sm font-normal text-gray-500">
                    ({alerts.length})
                  </span>
                </h2>
              </div>
              <AlertsPanel alerts={alerts} />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;