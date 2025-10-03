import React from 'react';
import { MapContainer, TileLayer, Marker, Popup, Circle } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

// Fix for default markers in react-leaflet
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

// Custom camera icon
const cameraIcon = L.divIcon({
  html: '📹',
  className: 'custom-camera-icon',
  iconSize: [30, 30],
  iconAnchor: [15, 15],
});

// Alert icons based on level
const getAlertIcon = (alertLevel) => {
  const icons = {
    safe: '🟢',
    warning: '🟡',
    critical: '🔴'
  };
  
  return L.divIcon({
    html: icons[alertLevel] || '⚪',
    className: 'custom-alert-icon',
    iconSize: [20, 20],
    iconAnchor: [10, 10],
  });
};

// Get alert color for circles
const getAlertColor = (alertLevel) => {
  const colors = {
    safe: '#10b981',
    warning: '#f59e0b',
    critical: '#ef4444'
  };
  return colors[alertLevel] || '#6b7280';
};

const MapView = ({ alerts }) => {
  // Mine boundary coordinates (example coordinates for an open-pit mine)
  const mineCenter = [45.0, -110.0];
  const mineBoundary = [
    [44.995, -110.005],
    [44.995, -109.995],
    [45.005, -109.995],
    [45.005, -110.005],
    [44.995, -110.005]
  ];

  // Camera positions around the mine
  const cameraPositions = [
    { id: 1, position: [44.998, -110.002], name: "Camera 1 - North Wall" },
    { id: 2, position: [45.002, -110.002], name: "Camera 2 - South Wall" },
    { id: 3, position: [45.000, -109.998], name: "Camera 3 - East Wall" },
    { id: 4, position: [45.000, -110.002], name: "Camera 4 - West Wall" },
  ];

  // Get recent alerts (last 10)
  const recentAlerts = alerts.slice(0, 10);

  return (
    <div className="w-full h-96 relative">
      <MapContainer
        center={mineCenter}
        zoom={14}
        scrollWheelZoom={true}
        className="w-full h-full rounded-lg"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        {/* Mine boundary */}
        <Circle
          center={mineCenter}
          radius={200}
          pathOptions={{
            color: '#374151',
            fillColor: '#f3f4f6',
            fillOpacity: 0.3,
            weight: 2,
          }}
        />

        {/* Camera positions */}
        {cameraPositions.map((camera) => (
          <Marker
            key={camera.id}
            position={camera.position}
            icon={cameraIcon}
          >
            <Popup>
              <div className="text-center">
                <h3 className="font-semibold">{camera.name}</h3>
                <p className="text-sm text-gray-600">Monitoring active</p>
                <p className="text-xs text-gray-500">
                  Position: {camera.position[0].toFixed(3)}, {camera.position[1].toFixed(3)}
                </p>
              </div>
            </Popup>
          </Marker>
        ))}

        {/* Alert markers */}
        {recentAlerts.map((alert, index) => {
          if (!alert.location) return null;
          
          return (
            <React.Fragment key={index}>
              <Marker
                position={[alert.location.lat, alert.location.lng]}
                icon={getAlertIcon(alert.alert_level)}
              >
                <Popup>
                  <div className="text-center min-w-32">
                    <h3 className={`font-semibold text-${alert.alert_level === 'safe' ? 'green' : alert.alert_level === 'warning' ? 'yellow' : 'red'}-600`}>
                      {alert.alert_level.toUpperCase()} Alert
                    </h3>
                    <p className="text-sm text-gray-700 mb-2">{alert.message}</p>
                    <p className="text-xs text-gray-500">
                      Probability: {(alert.probability * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-gray-500">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </Popup>
              </Marker>
              
              {/* Alert radius circle */}
              <Circle
                center={[alert.location.lat, alert.location.lng]}
                radius={50}
                pathOptions={{
                  color: getAlertColor(alert.alert_level),
                  fillColor: getAlertColor(alert.alert_level),
                  fillOpacity: 0.2,
                  weight: 2,
                  className: alert.alert_level !== 'safe' ? 'alert-pulse' : '',
                }}
              />
            </React.Fragment>
          );
        })}
      </MapContainer>

      {/* Map Legend */}
      <div className="absolute top-4 right-4 bg-white p-3 rounded-lg shadow-md text-sm z-10">
        <h4 className="font-semibold mb-2">Legend</h4>
        <div className="space-y-1">
          <div className="flex items-center">
            <span className="mr-2">📹</span>
            <span>Camera</span>
          </div>
          <div className="flex items-center">
            <span className="mr-2">🟢</span>
            <span>Safe</span>
          </div>
          <div className="flex items-center">
            <span className="mr-2">🟡</span>
            <span>Warning</span>
          </div>
          <div className="flex items-center">
            <span className="mr-2">🔴</span>
            <span>Critical</span>
          </div>
        </div>
      </div>

      {/* Current Alert Count */}
      {recentAlerts.length > 0 && (
        <div className="absolute top-4 left-4 bg-white p-2 rounded-lg shadow-md text-sm z-10">
          <div className="flex items-center">
            <span className="font-semibold mr-2">Active Alerts:</span>
            <span className="bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs">
              {recentAlerts.filter(a => a.alert_level !== 'safe').length}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapView;