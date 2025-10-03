import React from 'react';

const StatusPanel = ({ systemStatus }) => {
  const getStatusIcon = (status) => {
    switch (status) {
      case 'connected':
      case 'healthy':
        return '✅';
      case 'connecting':
        return '🔄';
      case 'disconnected':
        return '❌';
      case 'error':
        return '⚠️';
      default:
        return '❓';
    }
  };

  const getStatusMessage = (status) => {
    switch (status) {
      case 'connected':
      case 'healthy':
        return 'All systems operational';
      case 'connecting':
        return 'Establishing connection...';
      case 'disconnected':
        return 'Connection lost - attempting to reconnect';
      case 'error':
        return 'System error detected';
      default:
        return 'Unknown status';
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
      {/* System Status */}
      <div className="bg-white p-4 rounded-lg shadow border-l-4 border-blue-500">
        <div className="flex items-center">
          <div className="text-2xl mr-3">{getStatusIcon(systemStatus.status)}</div>
          <div>
            <p className="text-sm font-medium text-gray-600">System Status</p>
            <p className="text-lg font-semibold text-gray-900">
              {systemStatus.status.charAt(0).toUpperCase() + systemStatus.status.slice(1)}
            </p>
            <p className="text-xs text-gray-500">{getStatusMessage(systemStatus.status)}</p>
          </div>
        </div>
      </div>

      {/* Models Loaded */}
      <div className="bg-white p-4 rounded-lg shadow border-l-4 border-green-500">
        <div className="flex items-center">
          <div className="text-2xl mr-3">🤖</div>
          <div>
            <p className="text-sm font-medium text-gray-600">AI Models</p>
            <p className="text-lg font-semibold text-gray-900">{systemStatus.modelsLoaded.length}</p>
            <p className="text-xs text-gray-500">
              {systemStatus.modelsLoaded.length > 0 ? 'Active' : 'None loaded'}
            </p>
          </div>
        </div>
      </div>

      {/* Connected Clients */}
      <div className="bg-white p-4 rounded-lg shadow border-l-4 border-purple-500">
        <div className="flex items-center">
          <div className="text-2xl mr-3">👥</div>
          <div>
            <p className="text-sm font-medium text-gray-600">Connected Clients</p>
            <p className="text-lg font-semibold text-gray-900">{systemStatus.connectedClients}</p>
            <p className="text-xs text-gray-500">Real-time monitoring</p>
          </div>
        </div>
      </div>

      {/* System Uptime */}
      <div className="bg-white p-4 rounded-lg shadow border-l-4 border-orange-500">
        <div className="flex items-center">
          <div className="text-2xl mr-3">⏱️</div>
          <div>
            <p className="text-sm font-medium text-gray-600">Uptime</p>
            <p className="text-lg font-semibold text-gray-900">{systemStatus.uptime}</p>
            <p className="text-xs text-gray-500">Continuous operation</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StatusPanel;