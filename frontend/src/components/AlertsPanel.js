import React from 'react';

const AlertsPanel = ({ alerts }) => {
  const getAlertIcon = (alertLevel) => {
    const icons = {
      safe: '✅',
      warning: '⚠️',
      critical: '🚨'
    };
    return icons[alertLevel] || '❓';
  };

  const getAlertBadgeClass = (alertLevel) => {
    const classes = {
      safe: 'bg-green-100 text-green-800 border-green-200',
      warning: 'bg-yellow-100 text-yellow-800 border-yellow-200',
      critical: 'bg-red-100 text-red-800 border-red-200'
    };
    return classes[alertLevel] || 'bg-gray-100 text-gray-800 border-gray-200';
  };

  const getAlertBorderClass = (alertLevel) => {
    const classes = {
      safe: 'border-l-green-500',
      warning: 'border-l-yellow-500',
      critical: 'border-l-red-500'
    };
    return classes[alertLevel] || 'border-l-gray-500';
  };

  const formatTime = (timestamp) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { 
      hour: '2-digit', 
      minute: '2-digit',
      second: '2-digit'
    });
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString();
    }
  };

  const getProbabilityColor = (probability) => {
    if (probability >= 0.7) return 'text-red-600 font-semibold';
    if (probability >= 0.3) return 'text-yellow-600 font-semibold';
    return 'text-green-600 font-semibold';
  };

  // Group alerts by date
  const groupedAlerts = alerts.reduce((groups, alert) => {
    const date = formatDate(alert.timestamp);
    if (!groups[date]) {
      groups[date] = [];
    }
    groups[date].push(alert);
    return groups;
  }, {});

  if (alerts.length === 0) {
    return (
      <div className="p-8 text-center text-gray-500">
        <div className="text-4xl mb-4">🔍</div>
        <h3 className="text-lg font-medium mb-2">No Alerts</h3>
        <p className="text-sm">
          System is monitoring for potential rockfall events.
        </p>
      </div>
    );
  }

  return (
    <div className="h-96 overflow-y-auto custom-scrollbar">
      {Object.entries(groupedAlerts).map(([date, dateAlerts]) => (
        <div key={date} className="mb-4">
          {/* Date Header */}
          <div className="sticky top-0 bg-gray-50 px-4 py-2 text-sm font-medium text-gray-700 border-b">
            {date}
          </div>
          
          {/* Alerts for this date */}
          <div className="space-y-2 p-4">
            {dateAlerts.map((alert, index) => (
              <div
                key={index}
                className={`p-3 border-l-4 bg-white border border-gray-200 rounded-r-lg hover:shadow-md transition-shadow ${getAlertBorderClass(alert.alert_level)}`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3 flex-1">
                    <div className="text-lg flex-shrink-0 mt-0.5">
                      {getAlertIcon(alert.alert_level)}
                    </div>
                    
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className={`inline-block px-2 py-1 text-xs font-medium rounded-full border ${getAlertBadgeClass(alert.alert_level)}`}>
                          {alert.alert_level.toUpperCase()}
                        </span>
                        <span className="text-xs text-gray-500">
                          {formatTime(alert.timestamp)}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-800 mb-2 break-words">
                        {alert.message}
                      </p>
                      
                      <div className="flex items-center justify-between text-xs">
                        <span className={`${getProbabilityColor(alert.probability)}`}>
                          Risk: {(alert.probability * 100).toFixed(1)}%
                        </span>
                        
                        {alert.location && (
                          <span className="text-gray-500">
                            📍 {alert.location.lat.toFixed(3)}, {alert.location.lng.toFixed(3)}
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
      
      {/* Alert Summary */}
      <div className="sticky bottom-0 bg-gray-50 px-4 py-3 border-t text-sm">
        <div className="flex justify-between items-center">
          <span className="font-medium text-gray-700">
            Total Alerts: {alerts.length}
          </span>
          <div className="flex space-x-4 text-xs">
            <span className="flex items-center">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
              Safe: {alerts.filter(a => a.alert_level === 'safe').length}
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-yellow-500 rounded-full mr-1"></div>
              Warning: {alerts.filter(a => a.alert_level === 'warning').length}
            </span>
            <span className="flex items-center">
              <div className="w-2 h-2 bg-red-500 rounded-full mr-1"></div>
              Critical: {alerts.filter(a => a.alert_level === 'critical').length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlertsPanel;